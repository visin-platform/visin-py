"""The transport: one session, retries that know which requests may repeat, and
the response envelope undone.

Every vision-service endpoint answers with ``{"success": bool, "data": ...,
"message": str}``. Unwrapping that in one place is what keeps the rest of the
package free of ``data.get("data", {}).get("_id")`` chains, which is where the
copied-around integrations spent most of their length.
"""

from __future__ import annotations

import json as jsonlib
import logging
import os
import platform
import random
import time
from collections.abc import Mapping
from typing import Any, Callable, Tuple, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.exceptions import ConnectTimeoutError, MaxRetryError, NewConnectionError

from .._version import __version__
from ..errors import ApiError, TransportError
from .serialize import to_jsonable

logger = logging.getLogger("visin")

Timeout = Union[float, Tuple[float, float]]

# (connect, read). Connecting is quick or it is not happening; reading allows
# for a large epoch body and a busy server.
DEFAULT_TIMEOUT: Timeout = (5.0, 30.0)
# Uploads move bytes, and the read deadline covers sending the body too.
UPLOAD_TIMEOUT: Timeout = (5.0, 300.0)
# A download's read deadline applies to each chunk, not the whole body.
DOWNLOAD_TIMEOUT: Timeout = (5.0, 60.0)
DOWNLOAD_CHUNK = 8 * 2**20

# What Visin documents as worth retrying: the rate limit, and a gateway or
# service that did not answer in time. A 500 is not on the list; repeating it
# gets the same 500.
RETRY_STATUSES = frozenset({429, 502, 503, 504})

# Refusals that happen before any handler runs, so repeating even a POST that
# carries no id of its own cannot store it twice: the rate limiter answers 429,
# and required auth answers 503 when it cannot reach its database.
REFUSED_BEFORE_HANDLING = frozenset({429, 503})

# Transport failures worth another attempt. TLS failures are not among them:
# a certificate problem does not fix itself between attempts.
_RETRYABLE_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)

MAX_WAIT = 60.0


def never_sent(exc: BaseException) -> bool:
    """Whether a failed request certainly never reached the server.

    True for a connection that could not be opened: DNS, a refused port, a
    connect timeout. Those are safe to repeat for any request. A connection that
    dropped mid-request is not, because the server may have acted on it.
    """
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return True
    if isinstance(exc, requests.exceptions.SSLError):
        return False
    if isinstance(exc, requests.exceptions.ConnectionError):
        reason = exc.args[0] if exc.args else None
        if isinstance(reason, MaxRetryError):
            reason = reason.reason
        return isinstance(reason, (NewConnectionError, ConnectTimeoutError))
    return False


def worth_retrying_later(exc: BaseException) -> bool:
    """Whether a failure is the kind that may go away: unreachable or overloaded.

    These are the failures worth keeping a report on disk for. Anything else,
    a 400 naming a bad field or a 403 for a token outside its project, will be
    refused again however long it waits.
    """
    if isinstance(exc, TransportError):
        return True
    return isinstance(exc, ApiError) and exc.status in RETRY_STATUSES


def _retry_after(response: requests.Response) -> float | None:
    headers = getattr(response, "headers", None) or {}
    raw = headers.get("Retry-After") or headers.get("RateLimit-Reset")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:  # an HTTP date; the backoff will do
        return None


class HttpClient:
    """A retrying JSON client for one Visin instance."""

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        *,
        timeout: Timeout = DEFAULT_TIMEOUT,
        retries: int = 4,
        backoff: float = 0.5,
        verify: bool = True,
        session: requests.Session | None = None,
        upload_session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        # Tolerate both "https://host" and "https://host/api": the old scripts
        # hard-coded the latter and people will paste either.
        base = base_url.rstrip("/")
        self.base_url = base if base.endswith("/api") else base + "/api"
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.verify = verify
        self._sleep = sleep
        agent = f"visin-py/{__version__} python/{platform.python_version()}"

        self.session = session or requests.Session()
        if session is None:
            # Retrying is done in ``request``, not by urllib3, because whether a
            # request may be repeated depends on the request, and urllib3 only
            # knows its method.
            adapter = HTTPAdapter(max_retries=0)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
        if token:
            # A project token, a user API key and a user JWT are all presented
            # this way; the server tells them apart itself.
            self.session.headers["Authorization"] = f"Bearer {token}"
        self.session.headers["User-Agent"] = agent
        self.session.headers["Accept"] = "application/json"

        # Signed uploads and downloads go straight to file-service with the
        # signature as the credential. A session of their own means the
        # Authorization header above can never ride along to a storage endpoint
        # that has no use for it.
        self._uploads = upload_session or requests.Session()
        self._uploads.headers["User-Agent"] = agent

    # ------------------------------------------------------------------ JSON API

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        timeout: Timeout | None = None,
        idempotent: bool | None = None,
        retries: int | None = None,
    ) -> Any:
        """Send one request and return the unwrapped ``data`` field.

        ``idempotent`` says whether repeating the request is harmless. It
        defaults to true for everything but POST. A POST that carries its own
        UUID, which the server answers with 409 on a repeat, should pass True:
        that is what lets a dropped connection mid-request be retried at all.

        Raises ``ApiError`` for a refusal and ``TransportError`` for a request
        that never completed. A 409 arrives as ``ApiError(status=409)`` so the
        caller can treat "already recorded" as success. ``retries`` overrides
        the client's own budget for this one request.
        """
        budget = self.retries if retries is None else retries
        method = method.upper()
        if idempotent is None:
            idempotent = method != "POST"
        url = f"{self.base_url}/{path.lstrip('/')}"
        body = None if json is None else jsonlib.dumps(to_jsonable(json), allow_nan=False)
        headers = {"Content-Type": "application/json"} if body is not None else None

        attempt = 0
        while True:
            try:
                response = self.session.request(
                    method,
                    url,
                    data=body,
                    headers=headers,
                    params=params,
                    timeout=timeout or self.timeout,
                    verify=self.verify,
                )
            except requests.exceptions.RequestException as exc:
                retryable = isinstance(exc, _RETRYABLE_EXCEPTIONS) and not isinstance(
                    exc, requests.exceptions.SSLError
                )
                if retryable and attempt < budget and (idempotent or never_sent(exc)):
                    self._pause(attempt, None, f"{method} {path}: {exc}")
                    attempt += 1
                    continue
                raise TransportError(f"{method} {url} failed: {exc}") from exc

            status = response.status_code
            if (
                status in RETRY_STATUSES
                and attempt < budget
                and (idempotent or status in REFUSED_BEFORE_HANDLING)
            ):
                self._pause(attempt, _retry_after(response), f"{method} {path} answered {status}")
                attempt += 1
                continue
            return self._unwrap(method, url, response)

    def _pause(self, attempt: int, retry_after: float | None, why: str) -> None:
        if retry_after is not None:
            wait = retry_after
        else:
            # Exponential, with jitter so a fleet of workers that lost the
            # server together does not come back in lockstep.
            wait = self.backoff * (2**attempt) * (0.5 + random.random() / 2)
        wait = min(wait, MAX_WAIT)
        logger.debug("visin: %s; retrying in %.1fs", why, wait)
        self._sleep(wait)

    @staticmethod
    def _unwrap(method: str, url: str, response: requests.Response) -> Any:
        status = response.status_code
        payload: Any = None
        if response.content:
            try:
                payload = response.json()
            except ValueError:
                payload = None

        if status >= 400:
            # The server's message names the offending field or record, and it
            # is the difference between a five-minute fix and an afternoon. The
            # rate limiter's 429 is plain text, so fall back to the body.
            message = payload.get("message") if isinstance(payload, dict) else None
            text = str(message or response.text or "").strip()
            raise ApiError(
                f"{method} {url} returned {status}" + (f": {text[:300]}" if text else ""),
                status=status,
                body=(response.text or "")[:1000],
            )

        if payload is None:
            if response.content:
                raise ApiError(f"{method} {url} returned non-JSON", status=status)
            return None

        # Endpoints answer with the envelope; a bare body is passed through so
        # this client keeps working if one ever doesn't.
        if isinstance(payload, dict) and "success" in payload:
            if not payload.get("success"):
                raise ApiError(
                    payload.get("message") or f"{method} {url} was refused",
                    status=status,
                    body=str(payload)[:1000],
                )
            return payload.get("data")
        return payload

    # ------------------------------------------------------------------ uploads

    def put_file(self, url: str, path: str, content_type: str, *, retries: int | None = None) -> None:
        """PUT a file at a signed URL, retrying as a PUT may be retried."""
        budget = self.retries if retries is None else retries
        attempt = 0
        while True:
            try:
                with open(path, "rb") as handle:
                    response = self._uploads.put(
                        url,
                        data=handle,
                        headers={"Content-Type": content_type},
                        timeout=UPLOAD_TIMEOUT,
                        verify=self.verify,
                    )
            except _RETRYABLE_EXCEPTIONS as exc:
                if attempt < budget and not isinstance(exc, requests.exceptions.SSLError):
                    self._pause(attempt, None, f"upload: {exc}")
                    attempt += 1
                    continue
                raise TransportError(f"upload to signed URL failed: {exc}") from exc
            except requests.exceptions.RequestException as exc:
                raise TransportError(f"upload to signed URL failed: {exc}") from exc
            if response.status_code in RETRY_STATUSES and attempt < budget:
                self._pause(attempt, _retry_after(response), f"upload answered {response.status_code}")
                attempt += 1
                continue
            if response.status_code >= 400:
                raise ApiError(
                    f"upload returned {response.status_code}: {(response.text or '')[:300]}",
                    status=response.status_code,
                    body=(response.text or "")[:500],
                )
            return

    # ------------------------------------------------------------------ downloads

    def download_file(
        self,
        url: str,
        path: str,
        *,
        size: int | None = None,
        progress: Callable[[int, int | None], None] | None = None,
        retries: int | None = None,
    ) -> None:
        """GET a signed URL into ``path``, resuming a partial download.

        The body is written to ``path + ".part"`` and renamed to ``path`` only
        once complete, and with ``size``, only at that length: an interrupted
        download leaves the part to resume from, never a truncated file that
        looks finished. ``progress(done, total)`` is called as bytes arrive.
        """
        budget = self.retries if retries is None else retries
        part = path + ".part"
        attempt = 0
        while True:
            done = os.path.getsize(part) if os.path.exists(part) else 0
            if size is not None and done >= size:
                break
            headers = {"Range": f"bytes={done}-"} if done else {}
            try:
                with self._uploads.get(
                    url, headers=headers, stream=True, timeout=DOWNLOAD_TIMEOUT, verify=self.verify
                ) as response:
                    status = response.status_code
                    if status in RETRY_STATUSES and attempt < budget:
                        self._pause(attempt, _retry_after(response), f"download answered {status}")
                        attempt += 1
                        continue
                    if status == 416 and done:  # the part is already the whole file
                        break
                    if status >= 400:
                        raise ApiError(
                            f"download returned {status}: {(response.text or '')[:300]}",
                            status=status,
                            body=(response.text or "")[:500],
                        )
                    # 200 to a range request: the server sent the whole body again
                    mode = "ab" if done and status == 206 else "wb"
                    done = done if mode == "ab" else 0
                    total = size
                    with open(part, mode) as handle:
                        for chunk in response.iter_content(DOWNLOAD_CHUNK):
                            handle.write(chunk)
                            done += len(chunk)
                            if progress is not None:
                                progress(done, total)
            except _RETRYABLE_EXCEPTIONS as exc:
                if attempt < budget and not isinstance(exc, requests.exceptions.SSLError):
                    self._pause(attempt, None, f"download: {exc}")
                    attempt += 1
                    continue
                raise TransportError(f"download from signed URL failed: {exc}") from exc
            except requests.exceptions.RequestException as exc:
                raise TransportError(f"download from signed URL failed: {exc}") from exc
            if size is not None and done < size and attempt < budget:
                # The stream ended early without an error; ask for the rest
                self._pause(attempt, None, f"download ended at {done} of {size} bytes")
                attempt += 1
                continue
            break

        got = os.path.getsize(part) if os.path.exists(part) else 0
        if size is not None and got != size:
            raise TransportError(f"downloaded {got} of {size} bytes; run again to resume")
        os.replace(part, path)

    def close(self) -> None:
        self.session.close()
        self._uploads.close()
