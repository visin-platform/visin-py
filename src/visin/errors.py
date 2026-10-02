"""Errors this package raises.

Almost nothing here reaches a training loop. The whole point of the package is
that reporting failures are survivable: a run that trained for nine hours and
lost its last metric POST is a run you still want. So the transport layer
raises these internally and `Run` swallows them unless `strict=True`, which
exists for tests and for the smoke test, where a silent failure would defeat
the purpose.
"""


class VisinError(Exception):
    """Base class, so a caller can catch everything this package raises."""


class ConfigurationError(VisinError):
    """Something the environment had to supply is missing or unusable."""


class SyncInProgressError(VisinError):
    """Another ``visin sync`` is already sending this directory's reports."""


class ApiError(VisinError):
    """The server answered, and the answer was a refusal.

    `status` is the HTTP status; `body` is the response text, truncated. Both
    are kept because a Zod validation failure names the offending field in the
    body and nowhere else, and that message is the difference between a
    five-minute fix and an afternoon.
    """

    def __init__(self, message: str, status: "int | None" = None, body: "str | None" = None):
        super().__init__(message)
        self.status = status
        self.body = body


class TransportError(VisinError):
    """The request never got an answer — DNS, TCP, TLS, timeout.

    Separate from `ApiError` because this is the class of failure that is worth
    retrying, and on an HPC compute node it is also the common one.
    """
