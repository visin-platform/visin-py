"""Stand-ins for the network, shared by every test.

``FakeSession`` replaces the ``requests`` session behind the JSON API: it records
each request and answers from a queue of canned responses. ``FakeUploads`` does
the same for the signed-URL uploads that go straight to file-service.
"""

import json
import types
from pathlib import Path

BASE = "https://visin.example.test"


class FakeResponse:
    def __init__(self, status=200, payload=None, text="", headers=None):
        self.status_code = status
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")
        self.content = self.text.encode()
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def ok(data=None, status=200):
    return FakeResponse(status, {"success": True, "data": data if data is not None else {}})


def refused(status, message="refused", error="Error"):
    return FakeResponse(status, {"success": False, "error": error, "message": message})


class FakeSession:
    """Records requests and replays canned answers, keyed by (method, url fragment).

    A queued answer may be an exception, which is raised instead. With no
    answer queued, the default is an empty success.
    """

    def __init__(self):
        self.headers = {}
        self.calls = []
        self.routes = {}
        self.default = ok()

    def mount(self, *_args, **_kwargs):
        pass

    def close(self):
        pass

    def route(self, method, contains, *responses):
        self.routes.setdefault((method.upper(), contains), []).extend(responses)

    def request(self, method, url, data=None, headers=None, params=None, timeout=None, verify=None):
        body = json.loads(data) if data is not None else None
        self.calls.append({"method": method.upper(), "url": url, "json": body, "params": params})
        # Most specific fragment first, so "/trainings/uuid/" wins over "/trainings".
        for (verb, contains), queued in sorted(self.routes.items(), key=lambda item: -len(item[0][1])):
            if verb == method.upper() and contains in url and queued:
                answer = queued.pop(0)
                if isinstance(answer, BaseException):
                    raise answer
                return answer
        return self.default

    def paths(self, method=None):
        return [
            call["url"].split("/api", 1)[1]
            for call in self.calls
            if method is None or call["method"] == method.upper()
        ]

    def bodies(self, fragment, method="POST"):
        return [c["json"] for c in self.calls if c["method"] == method and c["url"].endswith(fragment)]


class FakeStream:
    """A streamed download response: a status and a body, served in chunks."""

    def __init__(self, status=200, body=b"", text="", fail_after=None):
        self.status_code = status
        self.body = body
        self.text = text
        self.fail_after = fail_after  # raise this exception after the first chunk

    def iter_content(self, chunk_size):
        for start in range(0, len(self.body), max(1, min(chunk_size, 1024))):
            yield self.body[start : start + 1024]
            if self.fail_after is not None:
                raise self.fail_after

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class FakeUploads:
    """Stands in for the signed-URL session: records what was PUT, and serves
    ``files`` ({url: bytes}) to GETs, honouring Range."""

    def __init__(self):
        self.headers = {}
        self.puts = []
        self.answers = []
        self.files = {}
        self.gets = []
        self.get_answers = []

    def get(self, url, headers=None, stream=False, timeout=None, verify=None):
        self.gets.append({"url": url, "headers": headers or {}})
        if self.get_answers:
            answer = self.get_answers.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            return answer
        body = self.files.get(url)
        if body is None:
            return FakeStream(404, text="no such file")
        requested = (headers or {}).get("Range", "")
        if requested.startswith("bytes="):
            start = int(requested[len("bytes=") :].rstrip("-"))
            if start >= len(body):
                return FakeStream(416)
            return FakeStream(206, body[start:])
        return FakeStream(200, body)

    def put(self, url, data=None, headers=None, timeout=None, verify=None):
        self.puts.append({"url": url, "bytes": data.read(), "headers": headers})
        if self.answers:
            answer = self.answers.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            return answer
        return FakeResponse(200, None, text="")

    def close(self):
        pass


class FakeAtexit:
    def __init__(self):
        self.callbacks = []

    def register(self, callback):
        self.callbacks.append(callback)

    def unregister(self, callback):
        self.callbacks = [known for known in self.callbacks if known != callback]


class FakeHubApi:
    def __init__(self, hub):
        self.hub = hub

    def create_repo(self, **kwargs):
        self.hub.created.append(kwargs)

    def upload_file(self, **kwargs):
        if self.hub.failure_on == kwargs["path_in_repo"]:
            raise RuntimeError("refused")
        return self.hub.commit("file", kwargs)

    def file_exists(self, **kwargs):
        return kwargs["filename"] in self.hub.existing

    def upload_folder(self, **kwargs):
        kwargs["files"] = sorted(
            path.relative_to(kwargs["folder_path"]).as_posix()
            for path in Path(kwargs["folder_path"]).rglob("*")
            if path.is_file()
        )
        return self.hub.commit("folder", kwargs)


class FakeHub(types.ModuleType):
    """Stands in for ``huggingface_hub``: records what was asked and writes a small repo."""

    COMMIT = "3f2a1c9d8e7b6a5f4e3d2c1b0a99887766554433"

    def __init__(self):
        super().__init__("huggingface_hub")
        self.downloads = []
        self.uploads = []
        self.created = []
        self.existing = set()
        self.failure = None
        self.failure_on = None

    def snapshot_download(self, **kwargs):
        self.downloads.append(kwargs)
        if self.failure:
            raise self.failure
        root = Path(kwargs["local_dir"])
        (root / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
        (root / "images").mkdir(exist_ok=True)
        (root / "images" / "1.png").write_bytes(b"png")
        (root / "README.md").write_text(kwargs["revision"])
        return str(root)

    def HfApi(self):
        return FakeHubApi(self)

    def commit(self, kind, kwargs):
        if self.failure:
            raise self.failure
        self.uploads.append({"kind": kind, **kwargs})
        return types.SimpleNamespace(oid=self.COMMIT)
