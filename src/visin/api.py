"""Reading what Visin holds: projects, runs, and their epochs and results.

For notebooks and analysis scripts. Visin's documentation recommends a user API
key with read scopes for this (``vsn_live_…``, made under **Account → API
keys**); a pipeline key works too, for its one project.

    from visin import Api

    api = Api()                        # VISIN_URL and VISIN_TOKEN
    for run in api.trainings(project="road-seg", status="completed"):
        print(run.name, run.status)
    frame = api.epochs_frame(run)      # one row per epoch, needs pandas
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any
from urllib.parse import quote

from ._internal.config import read_settings
from ._internal.transport import HttpClient
from .errors import ConfigurationError
from .models import Benchmark, Epoch, Project, TestResult, Training

# The server's cap on one page.
MAX_PAGE_SIZE = 1000

# Sort keys are Python names here; the server spells them its own way.
_SORT_KEYS = {
    "updated_at": "updatedAt",
    "created_at": "createdAt",
    "start_time": "startTime",
    "end_time": "endTime",
    "name": "name",
    "status": "status",
}


def flatten(results: Mapping[str, Any], sep: str = ".", prefix: str = "") -> dict[str, Any]:
    """Nested results as one level: ``{"val": {"loss": 1}}`` → ``{"val.loss": 1}``.

    Lists are kept whole, as values: they are rarely metrics, and exploding
    them into numbered columns makes a frame nobody can read.
    """
    flat: dict[str, Any] = {}
    for key, value in results.items():
        name = f"{prefix}{sep}{key}" if prefix else str(key)
        if isinstance(value, Mapping) and value:
            flat.update(flatten(value, sep, name))
        else:
            flat[name] = value
    return flat


class Api:
    """A read client for one Visin instance.

    Everything it returns is a model from :mod:`visin.models`. Anything a
    method takes as a run accepts a :class:`~visin.models.Training` or a run's
    UUID.
    """

    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
        *,
        verify_ssl: bool | None = None,
        client: HttpClient | None = None,
    ):
        if client is None:
            settings = read_settings(url=url, token=token, verify_ssl=verify_ssl)
            if not settings.url:
                raise ConfigurationError("no Visin to read from: pass url= or set VISIN_URL")
            client = HttpClient(settings.url, settings.token, verify=settings.verify_ssl)
        self._client = client

    def close(self) -> None:
        """Release the connection. ``with Api() as api`` does this for you."""
        self._client.close()

    def __enter__(self) -> Api:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def _get(self, path: str, **params: Any) -> Any:
        return self._client.request("GET", path, params={k: v for k, v in params.items() if v is not None})

    def _pages(
        self, path: str, key: str, params: Mapping[str, Any], page_size: int
    ) -> Iterator[dict[str, Any]]:
        page = 1
        size = max(1, min(page_size, MAX_PAGE_SIZE))
        while True:
            data = self._get(path, **params, page=page, limit=size) or {}
            items = data.get(key) or []
            yield from items
            pages = (data.get("pagination") or {}).get("pages")
            if not items or (pages is not None and page >= pages) or (pages is None and len(items) < size):
                return
            page += 1

    # ---------------------------------------------------------------- projects

    def projects(self, search: str | None = None) -> list[Project]:
        """The projects this credential can see."""
        return [Project.from_json(item) for item in self._get("/projects", search=search) or []]

    def project(self, id_or_slug: str) -> Project:
        """One project, by its id or slug."""
        return Project.from_json(self._get(f"/projects/{quote(id_or_slug, safe='')}") or {})

    # ---------------------------------------------------------------- runs

    def trainings(
        self,
        *,
        project: str | None = None,
        status: str | None = None,
        tags: Iterable[str] | str | None = None,
        search: str | None = None,
        dataset: str | None = None,
        sort_by: str = "updated_at",
        order: str = "desc",
        limit: int | None = None,
        page_size: int = 100,
    ) -> Iterator[Training]:
        """Runs, newest first, fetched a page at a time as you iterate.

        ``sort_by`` is one of ``updated_at``, ``created_at``, ``start_time``,
        ``end_time``, ``name`` or ``status``. ``limit`` stops after that many;
        without it, every matching run.
        """
        if sort_by not in _SORT_KEYS:
            raise ValueError(f"sort_by must be one of {', '.join(_SORT_KEYS)}, not {sort_by!r}")
        params = {
            "projectId": project,
            "status": status,
            "tags": tags if isinstance(tags, str) or tags is None else ",".join(tags),
            "search": search,
            "datasetId": dataset,
            "sortBy": _SORT_KEYS[sort_by],
            "order": order,
        }
        params = {key: value for key, value in params.items() if value is not None}
        if limit is not None:
            page_size = min(page_size, limit)
        for count, training in enumerate(self._pages("/trainings", "trainings", params, page_size)):
            if limit is not None and count >= limit:
                return
            yield Training.from_json(training)

    def training(self, ref: str | Training) -> Training:
        """One run, by its UUID, or a run you already have."""
        if isinstance(ref, Training):
            return ref
        return Training.from_json(self._get(f"/trainings/uuid/{quote(ref, safe='')}") or {})

    def epochs(self, ref: str | Training) -> list[Epoch]:
        """Every epoch of a run, in epoch order."""
        training = self.training(ref)
        pages = self._pages(
            f"/epochs/training/{training.id}", "epochs", {"sortBy": "epoch", "order": "asc"}, 1000
        )
        return [Epoch.from_json(item) for item in pages]

    def test_results(self, ref: str | Training) -> list[TestResult]:
        """The run's test results, newest first."""
        uuid = self._uuid(ref)
        pages = self._pages("/test-results", "testResults", {"training_uuid": uuid}, 500)
        return [TestResult.from_json(item) for item in pages]

    def benchmarks(
        self,
        ref: str | Training | None = None,
        *,
        project: str | None = None,
    ) -> list[Benchmark]:
        """Benchmarks of a run, or of a whole project, newest first."""
        params: dict[str, Any] = {}
        if ref is not None:
            params["training_uuid"] = self._uuid(ref)
        if project is not None:
            params["projectId"] = project
        return [Benchmark.from_json(item) for item in self._pages("/benchmarks", "benchmarks", params, 500)]

    @staticmethod
    def _uuid(ref: str | Training) -> str:
        return ref.uuid if isinstance(ref, Training) else ref

    # ---------------------------------------------------------------- frames

    def epochs_frame(self, ref: str | Training, *, sep: str = ".") -> Any:
        """A run's epochs as a pandas DataFrame, one row per epoch.

        Results are flattened into columns (``val.loss``, ``val.car.iou``), and
        ``learning_rate`` and ``epoch_time`` sit beside them. A metric an epoch
        did not report is missing (NaN), as it is in Visin's charts.
        """
        try:
            import pandas as pd  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("epochs_frame needs pandas: pip install 'visin[pandas]'") from exc
        rows = []
        for epoch in self.epochs(ref):
            row: dict[str, Any] = {
                "epoch": epoch.epoch,
                "timestamp": epoch.timestamp,
                "learning_rate": epoch.learning_rate,
                "epoch_time": epoch.epoch_time,
            }
            row.update(flatten(epoch.results, sep))
            rows.append(row)
        frame = pd.DataFrame(rows)
        if "timestamp" in frame:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        return frame.set_index("epoch") if "epoch" in frame else frame
