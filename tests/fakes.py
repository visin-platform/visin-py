"""Stand-ins for the network, shared by every test.

``FakeSession`` replaces the ``requests`` session behind the JSON API: it records
each request and answers from a queue of canned responses. ``FakeUploads`` does
the same for the signed-URL uploads that go straight to file-service.
"""

import json

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


class FakeUploads:
    """Stands in for the signed-URL session: records what was PUT."""

    def __init__(self):
        self.headers = {}
        self.puts = []
        self.answers = []

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
