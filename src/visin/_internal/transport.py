"""The transport: one session, retries that know which requests may repeat, and
the response envelope undone.

Every vision-service endpoint answers with ``{"success": bool, "data": ...,
"message": str}``. Unwrapping that in one place is what keeps the rest of the
package free of ``data.get("data", {}).get("_id")`` chains, which is where the
copied-around integrations spent most of their length.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json as jsonlib
import logging
import os
import platform
import random
import re
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
# How much of a file is read at a time to hash it, so a checkpoint never has to fit in memory.
HASH_BLOCK = 8 * 2**20
# A proxy in front of Visin commonly limits one request body to about 100 MB, so a large file goes up in
# pieces
# well under it: the same size the web app uses.
UPLOAD_CHUNK = 64 * 2**20

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


def file_sha256(path: str) -> str:
    """The SHA-256 of a file, read in pieces so a checkpoint does not have to fit in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for piece in iter(lambda: handle.read(HASH_BLOCK), b""):
            digest.update(piece)
    return digest.hexdigest()


def _representation_digest(response: Any) -> str | None:
    """A whole-file SHA-256 from Repr-Digest, even for a partial response."""
    header = getattr(response, "headers", {}).get("Repr-Digest", "")
    match = re.search(r"(?:^|,)\s*sha-256\s*=\s*:([^:]*):", header)
    if not match:
        if "sha-256" in header:
            raise TransportError("the response has an invalid Repr-Digest SHA-256")
        return None
    try:
        digest = base64.b64decode(match[1], validate=True)
    except binascii.Error as exc:
        raise TransportError("the response has an invalid Repr-Digest SHA-256") from exc
    if len(digest) != 32:
        raise TransportError("the response has an invalid Repr-Digest SHA-256")
    return digest.hex()


def _stored_size(response: Any) -> int | None:
    """The bytes file-service says it holds, from its reply to a piece, when it says."""
    try:
        size = response.json().get("size")
    except (ValueError, AttributeError):
        return None
    return size if isinstance(size, int) and not isinstance(size, bool) and size >= 0 else None


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
        # hard-coded the latter and people will paste either. A host with no
        # scheme is https, and a token pasted with its newline is the token.
        base = base_url.strip().rstrip("/")
        if "://" not in base:
            base = "https://" + base
        token = token.strip() if token else token
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
            # An API key and a user JWT are both
            # presented this way; the server tells them apart itself.
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
        anonymous: bool = False,
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
        headers: dict[str, str | None] = {"Content-Type": "application/json"} if body is not None else {}
        if anonymous:
            headers["Authorization"] = None

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

    def put_file_chunked(
        self,
        url: str,
        path: str,
        content_type: str,
        *,
        chunk_bytes: int = UPLOAD_CHUNK,
        progress: Callable[[int, int], None] | None = None,
        retries: int | None = None,
    ) -> None:
        """PUT a file at a signed URL in ``Content-Range`` pieces, resuming where the server says it is.

        A file of ``chunk_bytes`` or less is one request, as :meth:`put_file`. A larger one goes up a
        piece at a
        time, which is what gets it past a proxy's request-size limit and what makes it resumable: a
        piece that
        dies is sent again on its own, and when the server turns out to hold a different amount than assumed
        (answering 409 with the bytes it has) the upload continues from there rather than starting over.
        ``progress(done, total)`` is called as pieces land. The failures tolerated in a row are the client's
        retry budget; any piece that lands resets the count.
        """
        if chunk_bytes < 1:
            raise ValueError("chunk_bytes must be positive")
        total = os.path.getsize(path)
        if total <= chunk_bytes:
            self.put_file(url, path, content_type, retries=retries)
            if progress is not None:
                progress(total, total)
            return
        budget = self.retries if retries is None else retries
        offset = 0
        failures = 0
        with open(path, "rb") as handle:
            while offset < total:
                end = min(offset + chunk_bytes, total)
                handle.seek(offset)
                piece = handle.read(end - offset)
                headers = {"Content-Type": content_type, "Content-Range": f"bytes {offset}-{end - 1}/{total}"}
                try:
                    response = self._uploads.put(
                        url,
                        data=io.BytesIO(piece),
                        headers=headers,
                        timeout=UPLOAD_TIMEOUT,
                        verify=self.verify,
                    )
                except _RETRYABLE_EXCEPTIONS as exc:
                    if failures < budget and not isinstance(exc, requests.exceptions.SSLError):
                        self._pause(failures, None, f"upload at byte {offset}: {exc}")
                        failures += 1
                        continue
                    raise TransportError(f"upload to signed URL failed: {exc}") from exc
                except requests.exceptions.RequestException as exc:
                    raise TransportError(f"upload to signed URL failed: {exc}") from exc

                status = response.status_code
                stored = _stored_size(response)
                if stored is not None and stored > total:
                    raise TransportError(f"the server reports {stored} bytes for an upload of {total}")
                if status == 409 and stored is not None and stored != offset:
                    failures += 1
                    if failures > budget:
                        raise TransportError(
                            "upload could not resynchronise with the server "
                            f"(it holds {stored} of {total} bytes)"
                        )
                    offset = min(stored, total)
                    continue
                if 200 <= status < 300:
                    landed = stored if stored is not None else end
                    if landed <= offset:
                        failures += 1
                        if failures > budget:
                            raise TransportError(f"upload made no progress past byte {offset} of {total}")
                        continue
                    offset = min(landed, total)
                    failures = 0
                    if progress is not None:
                        progress(offset, total)
                    continue
                retryable = status in RETRY_STATUSES or status >= 500 or status == 409
                if retryable and failures < budget:
                    self._pause(failures, _retry_after(response), f"upload answered {status}")
                    failures += 1
                    continue
                raise ApiError(
                    f"upload returned {status}: {(response.text or '')[:300]}",
                    status=status,
                    body=(response.text or "")[:500],
                )

    # ------------------------------------------------------------------ downloads

    def download_file(
        self,
        url: str,
        path: str,
        *,
        size: int | None = None,
        sha256: str | None = None,
        authenticated: bool = False,
        progress: Callable[[int, int | None], None] | None = None,
        retries: int | None = None,
    ) -> None:
        """GET a URL into ``path``, resuming a partial download.

        The body is written to ``path + ".part"`` and renamed to ``path`` only
        once complete, and with ``size``, only at that length: an interrupted
        download leaves the part to resume from, never a truncated file that
        looks finished. With ``sha256`` the finished file must also have that
        digest; a file that does not is deleted and refused. ``progress(done,
        total)`` is called as bytes arrive.

        A signed URL is its own credential. With ``authenticated`` the request is
        made as this client, for an address on Visin's own API that needs one.
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
                if authenticated:
                    # The session's JSON `Accept` would have a file server answer 406.
                    opened = self.session.request(
                        "GET",
                        url,
                        headers={**headers, "Accept": "*/*"},
                        stream=True,
                        timeout=DOWNLOAD_TIMEOUT,
                        verify=self.verify,
                    )
                else:
                    opened = self._uploads.get(
                        url, headers=headers, stream=True, timeout=DOWNLOAD_TIMEOUT, verify=self.verify
                    )
                with opened as response:
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
                    declared = _representation_digest(response)
                    if declared:
                        if sha256 and declared != sha256.lower():
                            raise TransportError("Repr-Digest does not match the model's recorded SHA-256")
                        sha256 = declared
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
        if sha256 is not None:
            digest = file_sha256(part)
            if digest != sha256.lower():
                os.unlink(part)
                raise TransportError(
                    f"the downloaded file's SHA-256 is {digest}, not the {sha256} Visin recorded; "
                    "it was deleted"
                )
        os.replace(part, path)

    def close(self) -> None:
        self.session.close()
        self._uploads.close()
