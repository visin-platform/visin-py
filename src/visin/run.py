"""``Run``, the one object a training script touches, and ``init``, the one call
that makes it."""

from __future__ import annotations

import atexit
import functools
import logging
import os
import time
import uuid as uuidlib
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from . import system as _system
from ._internal.config import Settings, read_settings
from ._internal.inputs import as_mapping, default_name, epoch_number, merge_results, now
from ._internal.process import ProcessHooks, rank
from ._internal.reports import DeliveryContext, deliver, discard_staged
from ._internal.sender import Sender
from ._internal.serialize import to_jsonable
from ._internal.spool import Spool, sync_spool
from ._internal.transport import HttpClient, worth_retrying_later
from .errors import ConfigurationError, VisinError

logger = logging.getLogger("visin")

# Server-side caps. The Zod schemas accept any string; the Mongoose model
# rejects an over-long one, so an untruncated name fails as a 400 halfway
# through a run rather than at the call that set it.
MAX_NAME = 200
MAX_DESCRIPTION = 1000

STATUSES = ("pending", "running", "completed", "failed")

# What Visin stores as a visualization, by extension. The server checks the
# extension against the type and the bytes against both, so an unsupported
# file is refused here, at the call, rather than after a queued upload.
VISUALIZATION_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".pdf": "application/pdf",
}

# The API refuses a benchmark whose system_info lacks any of these.
BENCHMARK_SYSTEM_FIELDS = ("cpu_count", "cpu_count_logical", "memory_total_gb")

# How long a run that lost the server waits before trying it again.
CATCH_UP_INTERVAL = 60.0

# Retries for the requests that make a caller wait. Creating the run blocks the
# start of training, and a catch-up blocks every report queued behind it; both
# fall back to keeping reports on disk, so a short budget loses nothing. With
# the client's full budget, a server that drops packets rather than refusing
# held init() for close to a minute.
CREATE_RETRIES = 1
CATCH_UP_RETRIES = 0

# Fixed namespace, so the same (run, epoch) always names the same epoch on
# every machine and in every process. This is what makes a retried POST
# idempotent: the server answers the second one with 409 instead of recording
# the epoch twice.
_EPOCH_NS = uuidlib.uuid5(uuidlib.NAMESPACE_DNS, "epochs.visin")


def epoch_uuid_for(training_uuid: str, epoch: int | float) -> str:
    """The UUID an epoch of this run will always have."""
    return str(uuidlib.uuid5(_EPOCH_NS, f"{training_uuid}:{epoch}"))


class Run:
    """A training run, as Visin sees it.

    Make one with :func:`visin.init` (or :meth:`create`) to register a run, or
    with :meth:`attach` to report into one that already exists. A run is in one of three
    modes:

    * ``online`` sends reports as they are made, from a background thread. If
      Visin stops answering, reports are kept on disk and sent once it answers
      again, or by ``visin sync``.
    * ``offline`` keeps every report on disk for ``visin sync``, for machines
      with no route to Visin. Set ``VISIN_MODE=offline``.
    * ``disabled`` reports nothing: no server configured, or a process that is
      not rank zero of a distributed job. The same script still runs.

    Starting a run raises when Visin refuses it outright (a bad token, a
    project the key does not cover): that is a setup mistake, and the person
    starting the job is there to read it. Being unreachable does not raise; the
    run carries on and keeps its reports on disk. After that nothing on this
    class raises by default: a metrics backend is not worth a training job, so
    failures are logged and counted, and :meth:`finish` reports the tally. Pass
    ``strict=True`` to get exceptions there too.
    """

    def __init__(
        self,
        *,
        training_uuid: str | None,
        client: HttpClient | None = None,
        spool: Spool | None = None,
        strict: bool = False,
        training_id: str | None = None,
        system_metrics: bool = False,
        name: str | None = None,
        mark_status: bool = True,
    ):
        self.training_uuid = training_uuid
        self.name = name
        self._client = client
        self._spool = spool
        self._strict = strict
        self._system_metrics = system_metrics
        self._mark_status = mark_status
        self._context = DeliveryContext(files=spool.files if spool else None)
        if training_id and training_uuid:
            self._context.training_ids[training_uuid] = training_id
        self._sender = Sender() if client else None
        self._finished = False
        self._spooling = False
        self._next_catch_up = 0.0
        self._sent = 0
        self._warned_nonfinite = False
        self._warned_repeat = False
        self._resumed: bool | None = None
        # Epochs this process has delivered, to tell a retry from a re-log.
        self._delivered_epochs: set[str] = set()
        self.last_epoch: int | float | None = None
        if self.mode != "disabled":
            ProcessHooks.install()
            # A run killed by an exception, or cut off by the scheduler, should
            # still leave behind the epochs it managed to produce.
            atexit.register(self._at_exit)

    # ---------------------------------------------------------------- construction

    @classmethod
    def create(
        cls,
        name: str,
        *,
        project: str | None = None,
        dataset: str | None = None,
        model: str | None = None,
        config: Any = None,
        config_id: str | None = None,
        description: str | None = None,
        tags: Iterable[str] | str | None = None,
        metadata: Mapping[str, Any] | None = None,
        training_uuid: str | None = None,
        url: str | None = None,
        token: str | None = None,
        mode: str | None = None,
        directory: str | os.PathLike[str] | None = None,
        strict: bool = False,
        system_metrics: bool = False,
    ) -> Run:
        """Register a new run and return it.

        ``project`` is the project's id or slug. With a pipeline key (an API key
        limited to a project) it can be left out: the server puts the run in the
        key's project, whatever ``project`` says. ``model`` travels
        under ``metadata``, and ``dataset`` fills ``datasetId`` as well.

        With ``training_uuid`` (or ``VISIN_TRAINING_UUID``) naming a run that
        already exists, that run is resumed rather than duplicated, so a
        restarted job carries on where it was; :attr:`resumed` says which
        happened.

        Raises :class:`~visin.errors.ApiError` when Visin refuses the run.
        """
        full_name = (name or "").strip()
        if not full_name:
            raise ConfigurationError("a run needs a name")
        settings = read_settings(url=url, token=token, project=project, mode=mode, directory=directory)
        if settings.effective_mode == "disabled" or rank() != 0:
            return cls._disabled(settings)

        run_uuid = training_uuid or settings.training_uuid or str(uuidlib.uuid4())
        combined: dict[str, Any] = dict(metadata or {})
        if model:
            combined["model"] = model
        if dataset:
            combined["dataset"] = dataset
        payload: dict[str, Any] = {
            "uuid": run_uuid,
            "name": full_name[:MAX_NAME],
            "status": "running",
            "startTime": now(),
        }
        if combined:
            payload["metadata"] = combined
        if dataset:
            payload["datasetId"] = dataset
        if settings.project:
            payload["projectId"] = settings.project
        if config_id:
            payload["configId"] = config_id
        if tags:
            payload["tags"] = [tags] if isinstance(tags, str) else list(tags)
        # Callers pass a config summary as the name often enough that
        # truncating it would lose the only description of the run; keep the
        # full text rather than dropping it.
        if not description and len(full_name) > MAX_NAME:
            description = full_name
        if len(full_name) > MAX_NAME:
            logger.warning("visin: run name truncated to %d characters", MAX_NAME)
        if description:
            if len(description) > MAX_DESCRIPTION:
                logger.warning("visin: description truncated to %d characters", MAX_DESCRIPTION)
            payload["description"] = description[:MAX_DESCRIPTION]

        run = cls._build(settings, run_uuid, strict=strict, system_metrics=system_metrics, name=full_name)
        try:
            body = run._jsonable(payload, "run")
        except TypeError as exc:
            run._handle(exc, "create run")
            run._disable()
            return run
        run._start(body)
        if config is not None and run.enabled:
            run.log_config(config)
        return run

    @classmethod
    def attach(
        cls,
        training_uuid: str | None = None,
        *,
        url: str | None = None,
        token: str | None = None,
        mode: str | None = None,
        directory: str | os.PathLike[str] | None = None,
        strict: bool = False,
        system_metrics: bool = False,
        mark_status: bool = True,
    ) -> Run:
        """Report into a run that already exists, without registering it.

        ``training_uuid`` defaults to ``VISIN_TRAINING_UUID``, the variable an
        orchestrator exports for the processes it launches; with neither, this
        raises :class:`~visin.errors.ConfigurationError`. A disabled setup
        returns a run that reports nothing, so the script still runs.

        Pass ``mark_status=False`` from a script that adds to a run it does not
        own, such as a test or benchmark script run after training: its
        ``finish``, and its exit, crash included, then leave the run's status
        alone instead of marking the training completed or failed.
        """
        settings = read_settings(url=url, token=token, mode=mode, directory=directory)
        if settings.effective_mode == "disabled" or rank() != 0:
            return cls._disabled(settings)
        training_uuid = training_uuid or settings.training_uuid
        if not training_uuid:
            raise ConfigurationError(
                "no run to attach to: pass a training UUID or set VISIN_TRAINING_UUID. "
                "Use visin.init(...) to register a run instead."
            )
        return cls._build(
            settings, training_uuid, strict=strict, system_metrics=system_metrics, mark_status=mark_status
        )

    @classmethod
    def disabled(cls) -> Run:
        """A run that reports nothing. Useful in tests and offline scripts."""
        return cls(training_uuid=None)

    @classmethod
    def _disabled(cls, settings: Settings) -> Run:
        if rank() != 0:
            logger.debug("visin: rank %d of a distributed job; only rank 0 reports", rank())
        else:
            if settings.mode == "disabled":
                logger.info("visin: VISIN_MODE=disabled, so reporting is off")
            else:
                logger.info("visin: VISIN_URL/VISIN_TOKEN unset, so reporting is disabled")
        return cls.disabled()

    @classmethod
    def _build(
        cls,
        settings: Settings,
        training_uuid: str,
        *,
        strict: bool,
        system_metrics: bool,
        name: str | None = None,
        mark_status: bool = True,
    ) -> Run:
        try:
            spool: Spool | None = Spool(settings.directory, training_uuid)
        except ConfigurationError:
            if settings.effective_mode == "offline":
                raise
            spool = None  # online without a safety net, rather than not at all
        client = None
        if settings.effective_mode == "online":
            assert settings.url
            assert settings.token
            client = HttpClient(settings.url, settings.token, verify=settings.verify_ssl)
        return cls(
            training_uuid=training_uuid,
            client=client,
            spool=spool,
            strict=strict,
            system_metrics=system_metrics,
            name=name,
            mark_status=mark_status,
        )

    def _start(self, payload: dict[str, Any]) -> None:
        """Register the run, synchronously, so it exists before its first epoch."""
        op = {"op": "create_run", "body": payload}
        if self._client is None:
            assert self._spool is not None
            self._spool.append(op)
            logger.info(
                "visin: run %s is offline; reports are kept in %s for `visin sync`",
                self.training_uuid,
                self._spool.path,
            )
            return
        try:
            self._context.retries = CREATE_RETRIES
            try:
                deliver(self._client, op, self._context)
            finally:
                self._context.retries = None
            self._resumed = self.training_uuid in self._context.existing
            if self._resumed:
                logger.info("visin: resumed run %s (%s)", payload["name"][:60], self.training_uuid)
                # An earlier attempt may have left it completed or failed; it
                # is running again, and should say so while it trains.
                self._emit(
                    {"op": "update", "training_uuid": self.training_uuid, "body": {"status": "running"}}
                )
            else:
                logger.info("visin: created run %s (%s)", payload["name"][:60], self.training_uuid)
        except VisinError as exc:
            if self._spool is not None and worth_retrying_later(exc):
                self._spool.append(op)
                self._start_spooling()
                logger.warning(
                    "visin: could not reach Visin to create run %s (%s); reports are kept in %s and "
                    "will be sent when it answers",
                    self.training_uuid,
                    exc,
                    self._spool.path,
                )
                return
            # Refused outright: a bad token, or a project the token does not
            # cover. Every epoch would be refused the same way, and the person
            # starting a job is there to read this, so say it now rather than
            # let the run train unreported.
            self._disable()
            raise

    def _disable(self) -> None:
        logger.warning("visin: nothing more will be reported for run %s", self.training_uuid)
        atexit.unregister(self._at_exit)
        self._client = None
        self._spool = None
        self._sender = None

    # ---------------------------------------------------------------- state

    @property
    def mode(self) -> str:
        """``online``, ``offline`` or ``disabled``: what this run is doing with its reports."""
        if self._client is not None:
            return "online"
        if self._spool is not None:
            return "offline"
        return "disabled"

    @property
    def enabled(self) -> bool:
        """Whether this run reports anywhere, now or later."""
        return self.mode != "disabled"

    @property
    def resumed(self) -> bool | None:
        """Whether Visin already had this run when it started: a resumed or restarted job.

        ``None`` when that is not known: for an offline or disabled run, one made
        with :meth:`attach`, or one whose creation is waiting on disk.
        """
        return self._resumed

    @property
    def training_id(self) -> str | None:
        """The run's database id, once the server has told us."""
        if not self.training_uuid:
            return None
        return self._context.training_ids.get(self.training_uuid)

    def epoch_uuid(self, epoch: int | float) -> str | None:
        """The UUID this run's ``epoch`` has, whether or not it is sent yet."""
        if not self.training_uuid:
            return None
        return epoch_uuid_for(self.training_uuid, epoch_number(epoch))

    # ---------------------------------------------------------------- reporting

    def log_epoch(
        self,
        epoch: int | float,
        results: Mapping[str, Any] | None = None,
        *,
        train: Mapping[str, Any] | None = None,
        val: Mapping[str, Any] | None = None,
        learning_rate: float | None = None,
        epoch_time: float | None = None,
        metadata: Mapping[str, Any] | None = None,
        timestamp: str | datetime | None = None,
        system: bool | None = None,
    ) -> str | None:
        """Record one epoch. Returns its UUID, which is known before it is sent.

        ``results`` is passed through as given: Visin finds whatever a run chose
        to measure rather than requiring it to be declared. ``train`` and
        ``val`` are shorthand for ``results["train"]`` and ``results["val"]``,
        the two names the run's charts read as the training and validation
        curve. Leave out what an epoch did not measure; a missing metric is a
        gap in the chart, where a 0 would be a real drop.

        ``system=True`` (or ``system_metrics=True`` on the run) adds a snapshot
        of memory and GPU use as ``results["system_info"]``, which fills the
        run's System tab.
        """
        if not self._accepting() or not self.training_uuid:
            return None
        try:
            number = epoch_number(epoch)
            merged = merge_results(results, train, val)
            if system if system is not None else self._system_metrics:
                merged.setdefault("system_info", _system.system_metrics())
            ep_uuid = epoch_uuid_for(self.training_uuid, number)
            payload: dict[str, Any] = {
                "training_uuid": self.training_uuid,
                "epoch_uuid": ep_uuid,
                "epoch": number,
                "results": merged,
                "timestamp": timestamp or now(),
            }
            if learning_rate is not None:
                payload["learning_rate"] = learning_rate
            if epoch_time is not None:
                payload["epoch_time"] = epoch_time
            if metadata:
                payload["metadata"] = dict(metadata)
            body = self._jsonable(payload, f"epoch {number}")
        except (TypeError, ValueError) as exc:
            self._handle(exc, "log epoch")
            return None
        self.last_epoch = number
        self._emit({"op": "epoch", "body": body})
        return ep_uuid

    def log_test_results(
        self,
        epoch: int | float,
        test_results: Mapping[str, Any],
        *,
        test_uuid: str | None = None,
        timestamp: str | datetime | None = None,
        epoch_uuid: str | None = None,
    ) -> str | None:
        """Record scores on held-out data against an epoch of this run.

        ``test_results`` has three levels: conditions (``day``, ``night``, one
        per test set), classes inside them, and metrics inside those. An
        ``overall`` key holds metrics for a whole condition, or at the top level
        for the whole test. Returns the result's UUID.

        ``epoch_uuid`` names the epoch directly, for one this package did not
        log: an epoch recorded by other code, whose UUID a checkpoint's file
        name carries. By default it is the UUID ``log_epoch`` gave ``epoch``.
        """
        if not self._accepting() or not self.training_uuid:
            return None
        try:
            if test_results is None:
                raise ValueError("log_test_results needs test_results")
            number = epoch_number(epoch)
            # Generated here, not by the server, so a retried POST is answered
            # 409 instead of storing the result twice.
            test_uuid = test_uuid or str(uuidlib.uuid4())
            body = self._jsonable(
                {
                    "epoch": number,
                    "epoch_uuid": epoch_uuid or epoch_uuid_for(self.training_uuid, number),
                    "test_uuid": test_uuid,
                    "test_results": test_results,
                    "timestamp": timestamp or now(),
                },
                "test results",
            )
        except (TypeError, ValueError) as exc:
            self._handle(exc, "log test results")
            return None
        self._emit({"op": "test_result", "body": body})
        return test_uuid

    def log_benchmark(
        self,
        results: Mapping[str, Any] | Iterable[Mapping[str, Any]],
        system_info: Mapping[str, Any] | None = None,
        *,
        epoch: int | float | None = None,
        timestamp: str | datetime | None = None,
        epoch_uuid: str | None = None,
    ) -> None:
        """Record how the model runs: time per image, FPS, parameters, memory.

        ``results`` is one measurement or a list of them, usually one per
        device and batch size. ``system_info`` describes the machine; whatever
        it leaves out of what Visin requires is filled in by
        :func:`visin.system_info`. ``epoch`` links it to the checkpoint it
        measured, and ``epoch_uuid`` names that epoch directly, as for
        :meth:`log_test_results`.
        """
        if not self._accepting():
            return
        try:
            rows = [dict(results)] if isinstance(results, Mapping) else [dict(row) for row in results]
            info = dict(system_info or {})
            if any(field not in info for field in BENCHMARK_SYSTEM_FIELDS):
                info = {**_system.system_info(), **info}
            payload: dict[str, Any] = {
                "timestamp": timestamp or now(),
                "system_info": info,
                "results": rows,
            }
            # An explicit epoch UUID names the run too: the server finds it
            # through the epoch. Naming this run as well can only disagree,
            # which the server refuses as conflicting parents.
            if self.training_uuid and not epoch_uuid:
                payload["training_uuid"] = self.training_uuid
            if epoch is not None:
                number = epoch_number(epoch)
                payload["epoch"] = number
                if epoch_uuid:
                    payload["epoch_uuid"] = epoch_uuid
                elif self.training_uuid:
                    payload["epoch_uuid"] = epoch_uuid_for(self.training_uuid, number)
            elif epoch_uuid:
                payload["epoch_uuid"] = epoch_uuid
            body = self._jsonable(payload, "benchmark")
        except (TypeError, ValueError) as exc:
            self._handle(exc, "log benchmark")
            return
        self._emit({"op": "benchmark", "body": body})

    def log_config(
        self,
        config: Any,
        *,
        name: str | None = None,
        summary: str | None = None,
    ) -> None:
        """Store the configuration this run was launched with, and link it to the run.

        Takes a dict, an ``argparse.Namespace``, a dataclass, a pydantic model
        or a Hydra/OmegaConf config.
        """
        if not self._accepting():
            return
        try:
            data = as_mapping(config)
            body: dict[str, Any] = {
                "config_data": data,
                "summary": str(summary or data.get("Summary") or name or self.name or "Config"),
            }
            if name:
                body["config_name"] = name
            body = self._jsonable(body, "config", lenient=True)
        except (TypeError, ValueError, ImportError) as exc:
            self._handle(exc, "log config")
            return
        self._emit({"op": "config", "training_uuid": self.training_uuid, "body": body})

    def upload_visualization(
        self,
        epoch: int | float,
        path: str | os.PathLike[str],
        kind: str = "prediction",
        *,
        metadata: Mapping[str, Any] | None = None,
        mimetype: str | None = None,
        epoch_uuid: str | None = None,
    ) -> None:
        """Store a rendered frame against an epoch: a prediction overlay, a
        segmentation map, a ground-truth comparison.

        ``kind`` is your own name for what it shows; the Visualizations tab
        filters by it, and by file name, so render the same inputs under the
        same names each time. The file is copied at the call, so it may be
        overwritten straight after. ``epoch_uuid`` names the epoch directly,
        as for :meth:`log_test_results`.
        """
        if not self._accepting() or not self.training_uuid:
            return
        source = os.fspath(path)
        extension = os.path.splitext(source)[1].lower()
        content_type = mimetype or VISUALIZATION_TYPES.get(extension)
        try:
            if not os.path.isfile(source):
                raise ConfigurationError(f"no such file: {source}")
            if content_type is None:
                raise ConfigurationError(
                    "Visin stores PNG, JPEG, GIF, WebP, MP4, MOV and PDF files, "
                    f"not {extension or 'a file without an extension'}: {source}"
                )
            body: dict[str, Any] = {
                "epoch_uuid": epoch_uuid or epoch_uuid_for(self.training_uuid, epoch_number(epoch)),
                "filename": os.path.basename(source),
                "type": kind,
                "mimetype": content_type,
            }
            if metadata:
                body["metadata"] = dict(metadata)
            op: dict[str, Any] = {"op": "visualization", "body": self._jsonable(body, "visualization")}
            if self._spool is not None:
                op["staged"] = self._spool.stage(source)
            else:
                op["path"] = os.path.abspath(source)
        except (ConfigurationError, OSError, TypeError, ValueError) as exc:
            self._handle(exc, "upload visualization")
            return
        self._emit(op)

    def update(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        tags: Iterable[str] | str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Change the run's name, description, tags or metadata.

        ``metadata`` replaces what the run has; it is not merged.
        """
        if not self._accepting() or not self.training_uuid:
            return
        body: dict[str, Any] = {}
        if name is not None and name.strip():
            body["name"] = name.strip()[:MAX_NAME]
            self.name = name.strip()
        if description is not None:
            if len(description) > MAX_DESCRIPTION:
                logger.warning("visin: description truncated to %d characters", MAX_DESCRIPTION)
            body["description"] = description[:MAX_DESCRIPTION]
        if tags is not None:
            body["tags"] = [tags] if isinstance(tags, str) else list(tags)
        if metadata is not None:
            body["metadata"] = dict(metadata)
        if not body:
            return
        try:
            body = self._jsonable(body, "update")
        except TypeError as exc:
            self._handle(exc, "update run")
            return
        self._emit({"op": "update", "training_uuid": self.training_uuid, "body": body})

    # ---------------------------------------------------------------- delivery

    def _accepting(self) -> bool:
        if self.mode == "disabled":
            return False
        if self._finished:
            logger.warning("visin: run %s has finished; nothing more is sent", self.training_uuid)
            return False
        return True

    def _jsonable(self, payload: Mapping[str, Any], what: str, *, lenient: bool = False) -> dict[str, Any]:
        nonfinite: list[str] = []
        # A config records what a run was launched with, and holds whatever its
        # author put there: a loss module, a transform. Kept as its repr, that
        # is still worth more than refusing the whole config.
        body: dict[str, Any] = to_jsonable(payload, nonfinite, fallback=repr if lenient else None)
        if nonfinite and not self._warned_nonfinite:
            self._warned_nonfinite = True
            shown = ", ".join(nonfinite[:5]) + (", …" if len(nonfinite) > 5 else "")
            logger.warning(
                "visin: NaN or infinity in %s (%s) sent as null; JSON cannot carry it", what, shown
            )
        return body

    def _emit(self, op: dict[str, Any]) -> None:
        if self._client is not None:
            assert self._sender is not None
            # A partial, not a lambda: a stop that cannot wait hands queued work
            # back, and its op has to be readable to be kept on disk.
            if not self._sender.submit(functools.partial(self._send, op)):
                discard_staged(op, self._context)
        elif self._spool is not None:
            try:
                self._spool.append(op)
            except OSError as exc:
                self._handle(exc, f"keep {op['op']} on disk")

    def _send(self, op: dict[str, Any]) -> None:
        """Deliver one op. Runs on the reporter thread."""
        assert self._client is not None
        if self._spooling and not self._catch_up():
            self._keep(op)
            return
        try:
            deliver(self._client, op, self._context)
        except VisinError as exc:
            if self._spool is not None and worth_retrying_later(exc):
                if not self._spooling:
                    logger.warning(
                        "visin: lost Visin (%s); keeping reports in %s until it answers",
                        exc,
                        self._spool.path,
                    )
                self._keep(op)
                self._start_spooling()
                return
            discard_staged(op, self._context)
            raise
        discard_staged(op, self._context)
        self._sent += 1
        if op.get("op") == "epoch":
            self._note_epoch(str(op["body"].get("epoch_uuid")), op["body"].get("epoch"))

    def _note_epoch(self, epoch_uuid: str, epoch: Any) -> None:
        """Say so, once, when an epoch was recorded before and the new copy dropped."""
        if epoch_uuid in self._context.repeated:
            self._context.repeated.discard(epoch_uuid)
            if epoch_uuid not in self._delivered_epochs and not self._warned_repeat:
                self._warned_repeat = True
                logger.warning(
                    "visin: epoch %s of run %s was already recorded, by an earlier attempt or an earlier "
                    "run of this script, and Visin keeps the first copy. A job that restarts its epoch "
                    "numbering should start a new run.",
                    epoch,
                    self.training_uuid,
                )
        self._delivered_epochs.add(epoch_uuid)

    def _keep(self, op: dict[str, Any]) -> None:
        assert self._spool is not None
        self._spool.append(op)

    def _start_spooling(self) -> None:
        self._spooling = True
        self._next_catch_up = time.monotonic() + CATCH_UP_INTERVAL

    def _catch_up(self, force: bool = False) -> bool:
        """Send what was kept while Visin was away. True once nothing is left."""
        assert self._client is not None
        assert self._spool is not None
        if not force and time.monotonic() < self._next_catch_up:
            return False
        self._context.retries = CATCH_UP_RETRIES
        try:
            result = sync_spool(self._client, self._spool, drop_rejected=True, context=self._context)
        finally:
            self._context.retries = None
        self._sent += result.sent
        for message in result.rejected:
            logger.warning("visin: Visin refused a kept report: %s", message)
        if result.remaining == 0:
            if self._spooling:
                logger.info("visin: Visin answers again; sent %d kept reports", result.sent)
            self._spooling = False
            return True
        self._next_catch_up = time.monotonic() + CATCH_UP_INTERVAL
        return False

    # ---------------------------------------------------------------- lifecycle

    def flush(self, timeout: float = 30.0) -> bool:
        """Wait until every report made so far has been attempted."""
        return self._sender.flush(timeout) if self._sender else True

    def finish(self, status: str = "completed", *, timeout: float = 60.0) -> None:
        """Send what is left and mark the run finished. Safe to call twice."""
        if self._finished or self.mode == "disabled":
            self._finished = True
            return
        if status not in STATUSES:
            self._handle(ValueError(f"status must be one of {', '.join(STATUSES)}, not {status!r}"), "finish")
            status = "completed"
        assert self.training_uuid
        if self._mark_status:
            self._emit(
                {
                    "op": "update",
                    "training_uuid": self.training_uuid,
                    "body": {"status": status, "endTime": now()},
                }
            )
        else:
            status = "done"  # for the summary: the run's own status is not ours to set
        self._finished = True
        atexit.unregister(self._at_exit)

        flushed = True
        if self._sender is not None:
            flushed = self._sender.stop(timeout)
            self._keep_leftovers()
            if flushed and self._spooling:
                # One last try before the process goes away.
                try:
                    self._catch_up(force=True)
                except Exception as exc:
                    logger.warning("visin: could not send kept reports: %s", exc)
        self._summarise(status, flushed)
        if self._client is not None:
            self._client.close()

    def _keep_leftovers(self) -> None:
        """Keep on disk what a timed-out finish could not wait to send."""
        assert self._sender is not None
        leftover = [
            item.args[0]
            for item in self._sender.leftover
            if isinstance(item, functools.partial) and item.args
        ]
        self._sender.leftover = []
        if not leftover:
            return
        if self._spool is None:
            self._sender.dropped += len(leftover)
            return
        for op in leftover:
            try:
                self._spool.append(op)
            except OSError as exc:
                self._sender.dropped += 1
                logger.warning("visin: could not keep a report on disk: %s", exc)

    def fail(self, error: BaseException | str | None = None) -> None:
        """Mark the run failed, keeping whatever it already reported."""
        if error and self.enabled:
            logger.warning("visin: run failed: %s", error)
        self.finish(status="failed")

    def _summarise(self, status: str, flushed: bool) -> None:
        waiting = self._spool.count() if self._spool else 0
        if self.mode == "offline":
            logger.warning(
                "visin: run %s %s offline; %d reports kept in %s. Send them with `visin sync`.",
                self.training_uuid,
                status,
                waiting,
                self._spool.path if self._spool else "-",
            )
            return
        failed = self._sender.failed if self._sender else 0
        dropped = self._sender.dropped if self._sender else 0
        if waiting or failed or dropped or not flushed:
            logger.warning(
                "visin: run %s %s: %d reports sent, %d failed, %d dropped, %d waiting on disk%s",
                self.training_uuid,
                status,
                self._sent,
                failed,
                dropped,
                waiting,
                "; send them with `visin sync`" if waiting else "",
            )
        else:
            logger.info("visin: run %s %s, %d reports sent", self.training_uuid, status, self._sent)

    def _at_exit(self) -> None:
        # Reached when the process ends without finish(): a script that just
        # ran off its end, a crash, or a job the scheduler cut off. Whatever
        # is queued is worth more than the status.
        if self._finished:
            return
        crashed = ProcessHooks.crashed or ProcessHooks.terminated
        self.finish(status="failed" if crashed else "completed", timeout=15.0)

    def _handle(self, exc: BaseException, action: str) -> None:
        if self._strict:
            raise exc
        logger.warning("visin: could not %s: %s", action, exc)

    # ---------------------------------------------------------------- context manager

    def __enter__(self) -> Run:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None or (exc_type is SystemExit and exc.code in (0, None)):
            self.finish()
        elif exc_type is SystemExit:
            # A bare exit code says little: say what ended the process.
            self.fail(
                "terminated by SIGTERM" if ProcessHooks.terminated else f"exited with status {exc.code}"
            )
        else:
            self.fail(exc)

    def __repr__(self) -> str:
        state = self.mode if not self._finished else "finished"
        return f"<visin.Run {self.training_uuid or '-'} {state}>"


def init(
    name: str | None = None,
    *,
    project: str | None = None,
    dataset: str | None = None,
    model: str | None = None,
    config: Any = None,
    description: str | None = None,
    tags: Iterable[str] | str | None = None,
    metadata: Mapping[str, Any] | None = None,
    training_uuid: str | None = None,
    url: str | None = None,
    token: str | None = None,
    mode: str | None = None,
    directory: str | os.PathLike[str] | None = None,
    strict: bool = False,
    system_metrics: bool = False,
) -> Run:
    """Start reporting this script as a Visin run: the one call most scripts need.

    Reads ``VISIN_URL`` and ``VISIN_TOKEN`` (and ``VISIN_PROJECT``) from the
    environment unless given here, and returns a disabled run that reports
    nothing when there is no server to report to. Always registers a run,
    named ``name`` or after the script and the time; with ``training_uuid`` (or
    ``VISIN_TRAINING_UUID``) naming a run that exists, that run is resumed.
    To report into a run without registering it, use :meth:`Run.attach`.
    """
    return Run.create(
        name or default_name(),
        project=project,
        dataset=dataset,
        model=model,
        config=config,
        description=description,
        tags=tags,
        metadata=metadata,
        training_uuid=training_uuid,
        url=url,
        token=token,
        mode=mode,
        directory=directory,
        strict=strict,
        system_metrics=system_metrics,
    )
