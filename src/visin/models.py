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
    """Scores on held-out data, recorded against an epoch."""

    __test__ = False  # not a pytest class

    test_uuid: str | None = None
    epoch_uuid: str | None = None
    training_uuid: str | None = None
    results: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> TestResult:
        """The TestResult the server described in ``data``."""
        return cls(
            test_uuid=_str(_pick(data, "test_uuid", "testUuid")),
            epoch_uuid=_str(_pick(data, "epoch_uuid", "epochUuid")),
            training_uuid=_str(_pick(data, "training_uuid", "trainingUuid")),
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
