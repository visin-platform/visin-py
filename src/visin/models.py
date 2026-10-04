"""What Visin returns, as objects with Python names.

The server spells fields its own way (``_id``, ``projectId``, ``training_uuid``).
These classes are the translation, so a script reads ``run.project_id`` and a
server rename changes this file, not every script. ``raw`` keeps the server's
JSON for anything not modelled here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ._internal.providers import HUB


def _pick(data: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if data.get(name) is not None:
            return data[name]
    return None


def _str(value: Any) -> str | None:
    return None if value is None else str(value)


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class Project:
    """A project, as Visin lists it."""

    id: str
    name: str
    slug: str | None = None
    description: str | None = None
    visibility: str | None = None
    taxonomy: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def direction(self, metric: str) -> str | None:
        """``higher`` or ``lower``: which way is better for a result, as the project says, else ``None``.

        ``metric`` is a path such as ``val.mean_iou``; the project names a result by that path or by its
        last part. ``None`` means the project has not said, so any direction you assume is a guess.
        """
        leaf = metric.rsplit(".", 1)[-1]
        metrics = self.taxonomy.get("metrics") or ()
        for key in (metric, leaf):
            for item in metrics:
                if isinstance(item, Mapping) and item.get("key") == key and item.get("direction"):
                    return str(item["direction"])
        return None

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Project:
        """The Project the server described in ``data``."""
        return cls(
            id=str(_pick(data, "_id", "id") or ""),
            name=str(data.get("name") or ""),
            slug=_str(data.get("slug")),
            description=_str(data.get("description")),
            visibility=_str(data.get("visibility")),
            taxonomy=_dict(data.get("taxonomy")),
            created_at=_str(data.get("createdAt")),
            updated_at=_str(data.get("updatedAt")),
            raw=dict(data),
        )


@dataclass(frozen=True)
class Finding:
    """A conclusion someone (or an assistant) recorded about a project's runs."""

    id: str
    project_id: str
    title: str
    body: str
    recommendations: str | None = None
    training_id: str | None = None
    training_ids: tuple[str, ...] = ()
    author_kind: str | None = None
    author_label: str | None = None
    created_at: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Finding:
        """The Finding the server described in ``data``."""
        return cls(
            id=str(_pick(data, "_id", "id") or ""),
            project_id=str(data.get("projectId") or ""),
            title=str(data.get("title") or ""),
            body=str(data.get("body") or ""),
            recommendations=_str(data.get("recommendations")),
            training_id=_str(data.get("trainingId")),
            training_ids=tuple(str(item) for item in data.get("trainingIds") or ()),
            author_kind=_str(data.get("authorKind")),
            author_label=_str(data.get("authorLabel")),
            created_at=_str(data.get("createdAt")),
            raw=dict(data),
        )


@dataclass(frozen=True)
class Comparison:
    """A saved comparison of runs, test results, benchmarks or epochs."""

    id: str
    uuid: str
    name: str
    type: str | None = None
    description: str | None = None
    item_ids: tuple[str, ...] = ()
    project_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Comparison:
        """The Comparison the server described in ``data``."""
        return cls(
            id=str(_pick(data, "_id", "id") or ""),
            uuid=str(data.get("uuid") or ""),
            name=str(data.get("name") or ""),
            type=_str(data.get("type")),
            description=_str(data.get("description")),
            item_ids=tuple(str(item) for item in data.get("itemIds") or ()),
            project_id=_str(data.get("projectId")),
            created_at=_str(data.get("createdAt")),
            updated_at=_str(data.get("updatedAt")),
            raw=dict(data),
        )


@dataclass(frozen=True)
class Training:
    """A run. ``uuid`` is the one you chose; ``id`` is the database's."""

    id: str
    uuid: str
    name: str
    status: str | None = None
    description: str | None = None
    project_id: str | None = None
    dataset_id: str | None = None
    config_id: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    start_time: str | None = None
    end_time: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    models: tuple[dict[str, Any], ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Training:
        """The Training the server described in ``data``."""
        return cls(
            id=str(_pick(data, "_id", "id") or ""),
            uuid=str(data.get("uuid") or ""),
            name=str(data.get("name") or ""),
            status=_str(data.get("status")),
            description=_str(data.get("description")),
            project_id=_str(data.get("projectId")),
            dataset_id=_str(data.get("datasetId")),
            config_id=_str(data.get("configId")),
            tags=tuple(str(t) for t in data.get("tags") or ()),
            metadata=_dict(data.get("metadata")),
            start_time=_str(data.get("startTime")),
            end_time=_str(data.get("endTime")),
            created_at=_str(data.get("createdAt")),
            updated_at=_str(data.get("updatedAt")),
            metrics=_dict(data.get("metrics")),
            models=tuple(_dict(model) for model in data.get("models") or ()),
            provenance=_dict(data.get("provenance")),
            notes=_str(data.get("notes")),
            raw=dict(data),
        )


@dataclass(frozen=True)
class MetricSummary:
    """One result of a run: its path in an epoch, which way is better, and the best epoch beside the last.

    ``direction_from`` is ``taxonomy`` when the project said which way is better, and ``default`` when Visin
    guessed from the name (a loss or a latency is lower-is-better, the rest higher): check it before trusting
    a guess.
    """

    path: str
    direction: str
    direction_from: str
    best_value: float
    best_epoch: int
    last_value: float
    last_epoch: int

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> MetricSummary:
        """The MetricSummary the server described in ``data``."""
        best, last = _dict(data.get("best")), _dict(data.get("last"))
        return cls(
            path=str(data.get("path") or ""),
            direction=str(data.get("direction") or "higher"),
            direction_from=str(data.get("directionFrom") or "default"),
            best_value=float(best.get("value", 0.0)),
            best_epoch=int(best.get("epoch", 0)),
            last_value=float(last.get("value", 0.0)),
            last_epoch=int(last.get("epoch", 0)),
        )


@dataclass(frozen=True)
class Summary:
    """How a run did, in one call: the run, its models and provenance, and the best epoch of every result."""

    training: Training
    epoch_count: int
    last_epoch: int | None
    metrics: tuple[MetricSummary, ...] = ()
    models: tuple[dict[str, Any], ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def metric(self, path: str) -> MetricSummary | None:
        """The summary of one result by path (such as ``val.mean_iou``), or ``None`` if never reported."""
        return next((metric for metric in self.metrics if metric.path == path), None)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Summary:
        """The Summary the server described in ``data``."""
        last_epoch = data.get("lastEpoch")
        return cls(
            training=Training.from_json(_dict(data.get("training"))),
            epoch_count=int(data.get("epochCount") or 0),
            last_epoch=None if last_epoch is None else int(last_epoch),
            metrics=tuple(MetricSummary.from_json(_dict(item)) for item in data.get("metrics") or ()),
            models=tuple(_dict(model) for model in data.get("models") or ()),
            provenance=_dict(data.get("provenance")),
            raw=dict(data),
        )


@dataclass(frozen=True)
class Epoch:
    """One epoch of a run, with everything it reported in ``results``."""

    epoch: int | float
    uuid: str | None = None
    training_uuid: str | None = None
    results: dict[str, Any] = field(default_factory=dict)
    timestamp: str | None = None
    learning_rate: float | None = None
    epoch_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Epoch:
        """The Epoch the server described in ``data``."""
        return cls(
            epoch=_pick(data, "epoch") or 0,
            uuid=_str(_pick(data, "epoch_uuid", "epochUuid")),
            training_uuid=_str(_pick(data, "training_uuid", "trainingUuid")),
            results=_dict(data.get("results")),
            timestamp=_str(data.get("timestamp")),
            learning_rate=data.get("learning_rate"),
            epoch_time=data.get("epoch_time"),
            metadata=_dict(data.get("metadata")),
            raw=dict(data),
        )


@dataclass(frozen=True)
class TestResult:
    """Scores on held-out data, recorded against an epoch: an evaluation a run reported."""

    __test__ = False  # not a pytest class

    test_uuid: str | None = None
    epoch_uuid: str | None = None
    training_uuid: str | None = None
    results: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> TestResult:
        """The TestResult the server described in ``data``: an evaluation, or a test result in older words."""
        source = _dict(data.get("source")) if isinstance(data.get("source"), Mapping) else {}
        run = _dict(data.get("run")) if isinstance(data.get("run"), Mapping) else {}
        return cls(
            id=_str(_pick(data, "_id", "id")),
            test_uuid=_str(_pick(data, "test_uuid", "testUuid", "uuid")),
            epoch_uuid=_str(_pick(data, "epoch_uuid", "epochUuid") or source.get("epochUuid")),
            training_uuid=_str(_pick(data, "training_uuid", "trainingUuid") or run.get("uuid")),
            results=_dict(_pick(data, "test_results", "results", "metrics")),
            created_at=_str(_pick(data, "createdAt", "created_at")),
            raw=dict(data),
        )


@dataclass(frozen=True)
class Benchmark:
    """How a model ran on a machine: ``results`` per setting, and the ``system_info`` it ran on."""

    id: str
    results: Any = None
    system_info: dict[str, Any] = field(default_factory=dict)
    training_uuid: str | None = None
    epoch_uuid: str | None = None
    created_at: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Benchmark:
        """The Benchmark the server described in ``data``."""
        return cls(
            id=str(_pick(data, "_id", "id") or ""),
            results=_pick(data, "results", "metrics"),
            system_info=_dict(_pick(data, "system_info", "systemInfo")),
            training_uuid=_str(_pick(data, "training_uuid", "trainingUuid")),
            epoch_uuid=_str(_pick(data, "epoch_uuid", "epochUuid")),
            created_at=_str(_pick(data, "createdAt", "created_at")),
            raw=dict(data),
        )


@dataclass(frozen=True)
class Configuration:
    """What a run was launched with. ``config`` is the dict that was logged."""

    id: str
    uuid: str | None = None
    name: str | None = None
    training_id: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Configuration:
        """The Configuration the server described in ``data``."""
        return cls(
            id=str(_pick(data, "_id", "id") or ""),
            uuid=_str(data.get("uuid")),
            name=_str(data.get("name")),
            training_id=_str(data.get("trainingId")),
            config=_dict(_pick(data, "config", "config_data", "configData")),
            created_at=_str(_pick(data, "createdAt", "created_at")),
            raw=dict(data),
        )


@dataclass(frozen=True)
class Visualization:
    """A frame stored against an epoch. ``url`` is a signed link, present when it was asked for."""

    uuid: str
    epoch_uuid: str | None = None
    training_uuid: str | None = None
    kind: str | None = None
    filename: str | None = None
    url: str | None = None
    created_at: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Visualization:
        """The Visualization the server described in ``data``."""
        return cls(
            uuid=str(_pick(data, "visualization_uuid", "visualizationUuid", "_id") or ""),
            epoch_uuid=_str(_pick(data, "epoch_uuid", "epochUuid")),
            training_uuid=_str(_pick(data, "training_uuid", "trainingUuid")),
            kind=_str(data.get("type")),
            filename=_str(data.get("filename")),
            url=_str(_pick(data, "signedUrl", "url")),
            created_at=_str(_pick(data, "createdAt", "created_at")),
            raw=dict(data),
        )


@dataclass(frozen=True)
class Dataset:
    """A dataset on Visin. ``size`` is the zip's size in bytes, when it has one."""

    id: str
    name: str
    size: int | None = None
    filename: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def source(self) -> dict[str, Any] | None:
        """The Hugging Face repo and commit this dataset lives at, when it is kept there."""
        source = _dict(self.raw.get("source"))
        return source if source.get("repo") and source.get("revision") else None

    @property
    def downloadable(self) -> bool:
        """Whether there is something to download: a zip on Visin, or a Hub repo."""
        return bool(self.id and (self.raw.get("archive") or self.source))

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Dataset:
        """The Dataset the server described in ``data``."""
        archive = _dict(data.get("archive"))
        return cls(
            id=str(_pick(data, "_id", "id") or ""),
            name=str(data.get("name") or ""),
            size=archive.get("size"),
            filename=_str(archive.get("filename")),
            raw=dict(data),
        )


@dataclass(frozen=True)
class Reason:
    """One thing the server found wrong with a result, or worth saying about it.

    ``code`` is stable (``missing-condition``, ``protocol-mismatch``) and each is explained, with how to fix
    it, in Visin's documentation under *Why is my result unranked?*. ``detail`` says what it is about: a
    condition such as ``night``, or ``night/mIoU``.
    """

    code: str
    detail: str | None = None

    def __str__(self) -> str:
        return f"{self.code}({self.detail})" if self.detail else self.code

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Reason:
        """The Reason the server described in ``data``."""
        return cls(code=str(data.get("code") or ""), detail=_str(data.get("detail")))


@dataclass(frozen=True)
class Verdict:
    """Whether a result can be ranked on its suite, and why not.

    ``state`` is ``eligible`` (it can be ranked), ``incomplete`` (something the suite requires is missing),
    ``incompatible`` (it measured something other than the suite measures), ``exploratory`` (no suite was
    named) or ``legacy-unverified``. ``scores`` holds the numbers the ranking uses, for an eligible result.
    ``evidence`` labels the result, and never decides whether it is ranked: ``observed`` (the evaluator sent
    the data, protocol and evaluator it ran, and each matched), ``reported`` (ranked on the submitter's word,
    with less than that), ``attested`` (a manager vouched for a promoted result) or ``none`` (no suite). It
    is always the submitter's word.
    """

    state: str
    reasons: tuple[Reason, ...] = ()
    warnings: tuple[Reason, ...] = ()
    scores: dict[str, Any] | None = None
    evidence: str | None = None

    @property
    def ranked(self) -> bool:
        """Whether the result can be ranked."""
        return self.state == "eligible"

    def __str__(self) -> str:
        why = ", ".join(str(reason) for reason in self.reasons)
        return f"{self.state}: {why}" if why else self.state

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Verdict:
        """The Verdict the server described in ``data``."""
        scores = data.get("scores")
        return cls(
            state=str(data.get("state") or ""),
            reasons=tuple(
                Reason.from_json(item) for item in data.get("reasons") or () if isinstance(item, Mapping)
            ),
            warnings=tuple(
                Reason.from_json(item) for item in data.get("warnings") or () if isinstance(item, Mapping)
            ),
            scores=dict(scores) if isinstance(scores, Mapping) else None,
            evidence=_str(data.get("evidence")),
        )


@dataclass(frozen=True)
class Evaluation:
    """One checkpoint scored on one suite version, and the server's verdict on it.

    ``stored`` is false for a dry run, which has a verdict but no id. ``queued`` is true when the result
    could not be sent and waits on disk for ``visin sync``: it has no verdict yet.
    """

    id: str | None = None
    uuid: str | None = None
    project_id: str | None = None
    checkpoint: dict[str, Any] = field(default_factory=dict)
    checkpoint_key: str | None = None
    suite: str | None = None
    suite_digest: str | None = None
    status: str | None = None
    verdict: Verdict | None = None
    sample_counts: dict[str, int] = field(default_factory=dict)
    received_at: str | None = None
    published_at: str | None = None
    pending_approval: bool = False
    hidden: dict[str, Any] | None = None
    superseded_by: str | None = None
    results: dict[str, Any] = field(default_factory=dict, repr=False)
    stored: bool = True
    queued: bool = False
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def ranked(self) -> bool:
        """Whether it can be ranked. False while it is queued, since nothing has judged it yet."""
        return bool(self.verdict and self.verdict.ranked)

    @property
    def published(self) -> bool:
        """Whether a manager put it on its suite's public leaderboard."""
        return self.published_at is not None

    @property
    def public(self) -> bool:
        """Whether anyone can see it now: published, not waiting for approval and not hidden."""
        return self.published and not self.pending_approval and self.hidden is None

    @classmethod
    def from_json(cls, data: Mapping[str, Any], *, stored: bool = True) -> Evaluation:
        """The Evaluation the server described in ``data``."""
        suite = data.get("suite") if isinstance(data.get("suite"), Mapping) else None
        counts = data.get("sampleCounts")
        validation = data.get("validation")
        return cls(
            id=_str(_pick(data, "_id", "id")),
            uuid=_str(data.get("uuid")),
            project_id=_str(data.get("projectId")),
            checkpoint=_dict(data.get("checkpoint")),
            checkpoint_key=_str(data.get("checkpointKey")),
            suite=f"{suite['slug']}@{suite['version']}" if suite else None,
            suite_digest=_str(suite.get("digest")) if suite else None,
            status=_str(data.get("status")),
            verdict=Verdict.from_json(validation) if isinstance(validation, Mapping) else None,
            sample_counts={str(k): int(v) for k, v in counts.items()} if isinstance(counts, Mapping) else {},
            received_at=_str(data.get("receivedAt")),
            published_at=_str(data.get("publishedAt")),
            pending_approval=bool(data.get("pendingApproval")),
            hidden=_dict(data.get("hidden")) if isinstance(data.get("hidden"), Mapping) else None,
            superseded_by=_str(data.get("supersededById")),
            results=_dict(data.get("results")),
            stored=stored,
            raw=dict(data),
        )


@dataclass(frozen=True)
class Suite:
    """A published scoring protocol: results on one suite version can be compared.

    A published version never changes. ``digest`` is the hash of ``protocol``, so two suites with one digest
    score identically.
    """

    id: str
    slug: str
    version: int
    name: str
    description: str | None = None
    project_id: str | None = None
    visibility: str | None = None
    digest: str | None = None
    archived_at: str | None = None
    protocol: dict[str, Any] = field(default_factory=dict, repr=False)
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def ref(self) -> str:
        """``slug@version``, how a result names its suite."""
        return f"{self.slug}@{self.version}"

    @property
    def headline(self) -> dict[str, Any]:
        """The metric a ranking sorts by: its ``key``, ``direction`` (``max`` or ``min``) and ``unit``."""
        for metric in self.protocol.get("metrics") or ():
            if isinstance(metric, Mapping) and metric.get("headline"):
                return dict(metric)
        return {}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Suite:
        """The Suite the server described in ``data``."""
        return cls(
            id=str(_pick(data, "_id", "id") or ""),
            slug=str(data.get("slug") or ""),
            version=int(data.get("version") or 0),
            name=str(data.get("name") or ""),
            description=_str(data.get("description")),
            project_id=_str(data.get("projectId")),
            visibility=_str(data.get("visibility")),
            digest=_str(data.get("digest")),
            archived_at=_str(data.get("archivedAt")),
            protocol=_dict(data.get("protocol")),
            raw=dict(data),
        )


@dataclass(frozen=True)
class Pagination:
    """Where a page sits among the rows the server selected: ``total`` counts rows, ``pages`` counts pages."""

    page: int
    limit: int
    total: int
    pages: int

    @classmethod
    def from_json(cls, data: object) -> Pagination | None:
        """The pagination the server described in ``data``, or None when it sent none."""
        if not isinstance(data, Mapping):
            return None
        return cls(
            page=int(data.get("page") or 1),
            limit=int(data.get("limit") or 0),
            total=int(data.get("total") or 0),
            pages=int(data.get("pages") or 0),
        )


def _hub_name(checkpoint: Mapping[str, Any]) -> str:
    return f"{checkpoint.get('repo')} @ {str(checkpoint.get('commit', ''))[:7]}"


def _local_name(checkpoint: Mapping[str, Any]) -> str:
    return str(checkpoint.get("label") or "unknown")


_CHECKPOINT_NAMES = {HUB: _hub_name, "local": _local_name}


@dataclass(frozen=True)
class LeaderboardEntry:
    """One ranked checkpoint: its rank, headline score, where it is weakest, and the evidence."""

    rank: int
    evaluation_id: str
    checkpoint: dict[str, Any]
    headline: float
    worst_condition: str
    worst_value: float
    gap: float
    attempts: int
    project: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def name(self) -> str:
        """What a person calls the checkpoint: its Hub repo and commit, or its label."""
        name = _CHECKPOINT_NAMES.get(str(self.checkpoint.get("kind")), _local_name)
        return name(self.checkpoint)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> LeaderboardEntry:
        """The entry the server described in ``data``, as the app's ranking or the public one."""
        summary = data.get("summary") if isinstance(data.get("summary"), Mapping) else None
        if summary:
            headline = float((summary.get("headline") or {}).get("value", 0.0))
            worst = summary.get("worst") or {}
            gap = float(summary.get("gap", 0.0))
        else:
            headline = float(data.get("headline", 0.0))
            worst = data.get("worst") or {}
            gap = float(data.get("gap", 0.0))
        project = data.get("project")
        return cls(
            rank=int(data.get("rank") or 0),
            evaluation_id=str(data.get("evaluationId") or ""),
            checkpoint=_dict(data.get("checkpoint")),
            headline=headline,
            worst_condition=str(worst.get("condition") or ""),
            worst_value=float(worst.get("value", 0.0)),
            gap=gap,
            attempts=int(data.get("attempts") or 0),
            project=_str(project.get("name")) if isinstance(project, Mapping) else None,
            raw=dict(data),
        )


@dataclass(frozen=True)
class Leaderboard:
    """The ranking of one suite version.

    One entry per checkpoint, from its latest ranked attempt and never its best. ``candidates`` is how many
    evaluations were in the pool a rank is a position in (attempts of any outcome, not everything ever
    run), while ``pagination.total`` is how many checkpoints were selected from it: the two differ whenever
    a checkpoint has more than one attempt. Ranks are global, so page 2 starts at the rank the pool gives
    it, never at 1.

    ``unranked`` holds what has attempts but none ranked, with the reasons, paged apart from the ranking in
    ``unranked_pagination``: only the signed-in view has it, never the public one. ``complete`` says
    ``entries`` and ``unranked`` cover every page rather than the one that was asked for.
    """

    suite: str
    headline_key: str
    direction: str
    candidates: int
    entries: tuple[LeaderboardEntry, ...] = ()
    unranked: tuple[dict[str, Any], ...] = ()
    public: bool = False
    pagination: Pagination | None = None
    unranked_pagination: Pagination | None = None
    complete: bool = False
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_json(cls, data: Mapping[str, Any], *, public: bool = False) -> Leaderboard:
        """The Leaderboard the server described in ``data``."""
        suite = data.get("suite") or {}
        headline = suite.get("headline") or {}
        return cls(
            suite=f"{suite.get('slug')}@{suite.get('version')}",
            headline_key=str(headline.get("key") or ""),
            direction=str(headline.get("direction") or ""),
            candidates=int((data.get("scope") or {}).get("candidates") or 0),
            entries=tuple(
                LeaderboardEntry.from_json(item)
                for item in data.get("entries") or ()
                if isinstance(item, Mapping)
            ),
            unranked=tuple(dict(item) for item in data.get("unranked") or () if isinstance(item, Mapping)),
            public=public,
            pagination=Pagination.from_json(data.get("pagination")),
            unranked_pagination=Pagination.from_json(data.get("unrankedPagination")),
            raw=dict(data),
        )
