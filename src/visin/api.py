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

import os
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ._internal.config import HOSTED_URL, read_settings
from ._internal.reports import digest_of
from ._internal.transport import HttpClient
from .errors import ConfigurationError
from .models import (
    Benchmark,
    Comparison,
    Configuration,
    Epoch,
    Evaluation,
    Finding,
    Leaderboard,
    LeaderboardEntry,
    Project,
    Suite,
    Summary,
    TestResult,
    Training,
    Visualization,
)
from .suites import parse_suite_ref, protocol_of

# The server's cap on one page.
MAX_PAGE_SIZE = 1000
# The server's cap on one page of leaderboard rows.
LEADERBOARD_PAGE = 100
# The server's cap on one page of findings.
FINDINGS_PAGE = 200

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
                raise ConfigurationError(
                    f"no Visin to read from: pass url= or set VISIN_URL (hosted Visin: {HOSTED_URL})"
                )
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

    def _get_public(self, path: str, **params: Any) -> Any:
        return self._client.request(
            "GET", path, params={k: v for k, v in params.items() if v is not None}, anonymous=True
        )

    def _pages(
        self,
        path: str,
        key: str,
        params: Mapping[str, Any],
        page_size: int,
        *,
        public: bool = False,
        cap: int = MAX_PAGE_SIZE,
    ) -> Iterator[dict[str, Any]]:
        read = self._get_public if public else self._get
        page = 1
        size = max(1, min(page_size, cap))
        while True:
            data = read(path, **params, page=page, limit=size) or {}
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

    def find(
        self,
        name: str | None = None,
        *,
        project: str | None = None,
        tags: Iterable[str] | str | None = None,
        status: str | None = None,
    ) -> Training | None:
        """The newest run called exactly ``name`` that matches the rest, or ``None``.

        Without ``name``, the newest run that matches the filters. For a script
        that must pick up where an earlier one left off without having kept
        its UUID.
        """
        for training in self.trainings(project=project, status=status, tags=tags, search=name):
            if name is None or training.name == name:
                return training
        return None

    def comparisons(
        self,
        *,
        project: str | None = None,
        type: str | None = None,  # noqa: A002 - the server's own word for what a comparison compares
        search: str | None = None,
        limit: int | None = None,
        page_size: int = MAX_PAGE_SIZE,
    ) -> Iterator[Comparison]:
        """Saved comparisons, a page at a time, optionally of one ``type`` (trainings, tests, ...)."""
        params = {"projectId": project, "type": type, "search": search}
        params = {key: value for key, value in params.items() if value is not None}
        for count, item in enumerate(self._pages("/comparisons", "comparisons", params, page_size)):
            if limit is not None and count >= limit:
                return
            yield Comparison.from_json(item)

    def findings(
        self, *, project: str | None = None, run: str | Training | None = None, limit: int | None = None
    ) -> Iterator[Finding]:
        """Recorded findings, newest first: a project's, or those about one run (as its subject or cited)."""
        if limit is not None and limit <= 0:
            return
        training = self.training(run).id if run is not None else None
        before: str | None = None
        yielded = 0
        while True:
            size = FINDINGS_PAGE if limit is None else max(1, min(FINDINGS_PAGE, limit - yielded))
            items = (
                self._get("/findings", project=project, training=training, limit=size, before=before) or []
            )
            for item in items:
                yield Finding.from_json(item)
                yielded += 1
                if limit is not None and yielded >= limit:
                    return
            if len(items) < size:
                return
            last = items[-1]
            before = f"{last['createdAt']}_{last['_id']}"

    def summary(self, ref: str | Training) -> Summary:
        """How a run did, in one call: its state, models and provenance, and per result the best epoch.

        ``summary.metric("val.mean_iou")`` gives the best value, the epoch it was reached at and the last
        value, for the direction the project says is better. A run's last epoch is not its result, so both
        are there. Read ``direction_from`` on a metric: ``default`` means Visin guessed the direction.
        """
        training = self.training(ref)
        return Summary.from_json(self._get(f"/trainings/{quote(training.id, safe='')}/summary") or {})

    def tags(self) -> list[str]:
        """Every tag on a run this credential can see."""
        return [str(tag) for tag in self._get("/trainings/tags") or []]

    def config(self, ref: str | Training) -> Configuration | None:
        """What the run was launched with, or ``None`` if it logged no config."""
        training = self.training(ref)
        found = (self._get(f"/trainings/{quote(training.id, safe='')}/configs") or {}).get("configs") or []
        return Configuration.from_json(found[0]) if found else None

    def visualizations(self, ref: str | Training, kind: str | None = None) -> list[Visualization]:
        """The frames a run stored, newest first, each with a signed link to its file.

        ``kind`` keeps one kind, as given to ``upload_visualization``.
        """
        params = {"type": kind, "includeUrls": "true"}
        pages = self._pages(
            f"/visualizations/training/{quote(self._uuid(ref), safe='')}", "visualizations", params, 100
        )
        return [Visualization.from_json(item) for item in pages]

    def download_visualization(self, visualization: Visualization, directory: str | os.PathLike[str]) -> Path:
        """Save a frame under ``directory``, by its file name, and return the path."""
        if not (visualization.url and visualization.filename):
            raise ConfigurationError(f"visualization {visualization.uuid} has no link to download from")
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / Path(visualization.filename).name
        self._client.download_file(visualization.url, str(path))
        return path

    def epochs(self, ref: str | Training) -> list[Epoch]:
        """Every epoch of a run, in epoch order."""
        training = self.training(ref)
        pages = self._pages(
            f"/epochs/training/{training.id}", "epochs", {"sortBy": "epoch", "order": "asc"}, 1000
        )
        return [Epoch.from_json(item) for item in pages]

    def test_results(self, ref: str | Training) -> list[TestResult]:
        """What the run's checkpoints scored, newest first: its test results, with or without a suite."""
        uuid = self._uuid(ref)
        pages = self._pages("/evaluations", "evaluations", {"trainingUuid": uuid, "include": "results"}, 500)
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

    # ---------------------------------------------------------------- suites and evaluations

    def suites(
        self, *, slug: str | None = None, project: str | None = None, include_archived: bool = False
    ) -> list[Suite]:
        """The suites this credential can read, newest version first within a name.

        A suite is readable by whoever can read its project, and by anyone when it is public. Archived suites
        (retired: no new results) are left out unless ``include_archived`` is true.
        """
        params: dict[str, Any] = {"slug": slug, "projectId": project}
        if include_archived:
            params["includeArchived"] = "true"
        return [Suite.from_json(item) for item in self._pages("/suites", "suites", params, 100)]

    def suite(self, ref: str, version: int | str | None = None) -> Suite:
        """One suite, by ``"slug@1"``, by slug and version, or by slug alone for the latest."""
        slug, found = parse_suite_ref(ref) if version is None else (ref, str(version))
        return Suite.from_json(self._get(f"/suites/{quote(slug, safe='')}/{quote(found, safe='')}") or {})

    def evaluations(
        self,
        *,
        project: str | None = None,
        suite: str | None = None,
        state: str | None = None,
        status: str | None = None,
        checkpoint_key: str | None = None,
    ) -> list[Evaluation]:
        """Evaluations this credential can read, newest first, without their results and provenance.

        ``suite`` is ``"slug@version"``. ``state`` keeps one verdict (``eligible``, ``incomplete``,
        ``incompatible``, ``exploratory``, ``legacy-unverified``); open an evaluation with
        :meth:`evaluation` for its results.
        """
        params = {
            "projectId": project,
            "suite": suite,
            "state": state,
            "status": status,
            "checkpointKey": checkpoint_key,
        }
        return [
            Evaluation.from_json(item) for item in self._pages("/evaluations", "evaluations", params, 100)
        ]

    def evaluation(self, ref: str | Evaluation, *, project: str | None = None) -> Evaluation:
        """One evaluation, with its results and provenance, by id or (with ``project``) by your own uuid.

        Use the uuid form to recover after a lost response: it finds what an earlier attempt stored.
        """
        if isinstance(ref, Evaluation):
            ref = ref.id or ""
        if project is not None:
            return Evaluation.from_json(
                self._get(f"/evaluations/uuid/{quote(ref, safe='')}", projectId=project) or {}
            )
        return Evaluation.from_json(self._get(f"/evaluations/{quote(ref, safe='')}") or {})

    def check_protocol(self, source: str | Path | Mapping[str, Any]) -> str:
        """The digest Visin gives a protocol file or dict, without publishing it."""
        return digest_of(self._client, protocol_of(source))

    def leaderboard(
        self,
        ref: str,
        version: int | str | None = None,
        *,
        page: int | None = None,
        limit: int | None = None,
        unranked_page: int | None = None,
        all_pages: bool = False,
        observed: bool = False,
    ) -> Leaderboard:
        """The ranking of a suite version over the evaluations this credential can read.

        One entry per checkpoint, from its latest ranked attempt, never its best. ``candidates`` says how many
        evaluations were in the pool: a rank is a position in that pool, and ranks are global, so page 2 does
        not restart at 1. ``pagination.total`` is how many checkpoints were selected. What has attempts but
        none ranked is in ``unranked``, with the reasons, and pages apart from the ranking
        (``unranked_page``).

        ``page`` and ``limit`` (at most 100) choose one page. ``all_pages=True`` reads every page of both
        lists and returns them together with ``complete`` set; it cannot be combined with ``page``.

        ``observed=True`` ranks only the results whose evaluator sent complete evidence that matched; the
        pool, its counts and the ranks are then those of that subset, and ``reported`` and promoted results
        are left out.
        """
        slug, found = parse_suite_ref(ref) if version is None else (ref, str(version))
        path = f"/suites/{quote(slug, safe='')}/{quote(found, safe='')}/leaderboard"
        evidence = "observed" if observed else None
        if all_pages:
            return self._whole_board(path, page, unranked_page, limit, public=False, evidence=evidence)
        return Leaderboard.from_json(
            self._get(path, page=page, limit=limit, unrankedPage=unranked_page, evidence=evidence) or {}
        )

    def public_leaderboards(
        self, *, page: int | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """One page of the leaderboards with something published, anonymously: the same list whoever asks.

        The server sends 100 at most per page; ``iter_public_leaderboards`` reads them all.
        """
        data = self._get_public("/public/leaderboards", page=page, limit=limit) or {}
        return [dict(item) for item in data.get("leaderboards") or ()]

    def iter_public_leaderboards(self, *, page_size: int = LEADERBOARD_PAGE) -> Iterator[dict[str, Any]]:
        """Every public leaderboard, anonymously, reading page after page."""
        return self._pages(
            "/public/leaderboards", "leaderboards", {}, page_size, public=True, cap=LEADERBOARD_PAGE
        )

    def public_leaderboard(
        self,
        ref: str,
        version: int | str | None = None,
        *,
        page: int | None = None,
        limit: int | None = None,
        all_pages: bool = False,
        observed: bool = False,
    ) -> Leaderboard:
        """One public leaderboard, anonymously: only published results, and a version number is required.

        Paged and global-ranked like ``leaderboard``; there is no ``unranked`` list on the public view.
        ``observed=True`` ranks only the results whose evaluator sent complete evidence that matched.
        """
        slug, found = parse_suite_ref(ref) if version is None else (ref, str(version))
        if found == "latest":
            raise ValueError("a public leaderboard is addressed by a version, such as road-test@1")
        path = f"/public/leaderboards/{quote(slug, safe='')}/{quote(found, safe='')}"
        evidence = "observed" if observed else None
        if all_pages:
            return self._whole_board(path, page, None, limit, public=True, evidence=evidence)
        return Leaderboard.from_json(
            self._get_public(path, page=page, limit=limit, evidence=evidence) or {}, public=True
        )

    def _whole_board(
        self,
        path: str,
        page: int | None,
        unranked_page: int | None,
        limit: int | None,
        *,
        public: bool,
        evidence: str | None = None,
    ) -> Leaderboard:
        """Every page of a ranking, and of its unranked list when it has one, merged into one board.

        Rows are de-duplicated by evaluation id, since a write between two requests can move a row across a
        page boundary.
        """
        if page is not None or unranked_page is not None:
            raise ValueError("all_pages reads every page: leave page and unranked_page out")
        read = self._get_public if public else self._get
        size = max(1, min(limit or LEADERBOARD_PAGE, LEADERBOARD_PAGE))
        first = Leaderboard.from_json(read(path, page=1, limit=size, evidence=evidence) or {}, public=public)
        entries = {entry.evaluation_id: entry for entry in first.entries}
        raw_entries = list(first.raw.get("entries") or ())
        for number in range(2, (first.pagination.pages if first.pagination else 0) + 1):
            more = read(path, page=number, limit=size, evidence=evidence) or {}
            for item in more.get("entries") or ():
                entry = LeaderboardEntry.from_json(item)
                if entry.evaluation_id not in entries:
                    entries[entry.evaluation_id] = entry
                    raw_entries.append(item)
        unranked = list(first.unranked)
        seen = {row.get("evaluationId") for row in unranked}
        pages = first.unranked_pagination.pages if first.unranked_pagination else 0
        beyond = (first.pagination.pages if first.pagination else 0) + 1
        for number in range(2, pages + 1):
            more = read(path, page=beyond, limit=size, unrankedPage=number, evidence=evidence) or {}
            for row in more.get("unranked") or ():
                if row.get("evaluationId") not in seen:
                    seen.add(row.get("evaluationId"))
                    unranked.append(dict(row))
        return replace(
            first,
            entries=tuple(entries.values()),
            unranked=tuple(unranked),
            complete=True,
            raw={**first.raw, "entries": raw_entries, "unranked": unranked},
        )

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

    def compare_frame(self, refs: Iterable[str | Training], metric: str, *, sep: str = ".") -> Any:
        """One metric of several runs side by side: a row per epoch, a column per run.

        ``metric`` is a flattened name such as ``val.mean_iou``. Columns are run
        names; two runs with the same name are told apart by the start of their
        UUID. A run that never reported the metric is a column of NaN.
        """
        try:
            import pandas as pd  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("compare_frame needs pandas: pip install 'visin[pandas]'") from exc
        runs = [self.training(ref) for ref in refs]
        names = [run.name for run in runs]
        columns = {}
        for run in runs:
            label = run.name if names.count(run.name) == 1 else f"{run.name} ({run.uuid[:8]})"
            frame = self.epochs_frame(run, sep=sep)
            columns[label] = frame[metric] if metric in frame else pd.Series(dtype=float)
        return pd.DataFrame(columns).sort_index()
