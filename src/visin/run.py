"""``Run``, the one object a training script touches, and ``init``, the one call
that makes it."""

from __future__ import annotations

import atexit
import logging
import os
import uuid as uuidlib
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from . import system as _system
from ._internal import hub
from ._internal import provenance as provenance_info
from ._internal.config import HOSTED_URL, Settings, read_settings
from ._internal.delivery import Delivery
from ._internal.images import encode_png, is_image
from ._internal.inputs import as_mapping, default_name, epoch_number, merge_results, now
from ._internal.payloads import benchmark_payload, model_payload, run_payload, update_body
from ._internal.process import ProcessHooks, rank
from ._internal.reports import discard_staged, fetch_model_card
from ._internal.serialize import to_jsonable
from ._internal.spool import Spool
from ._internal.transport import HttpClient, worth_retrying_later
from .errors import ConfigurationError, VisinError

logger = logging.getLogger("visin")

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
        settings: Settings | None = None,
    ):
        self.training_uuid = training_uuid
        self._settings = settings
        self.name = name
        self._delivery = Delivery(client, spool, training_uuid)
        self._strict = strict
        self._system_metrics = system_metrics
        self._mark_status = mark_status
        if training_id and training_uuid:
            self._delivery.context.training_ids[training_uuid] = training_id
        if mark_status and self._delivery.sender is not None:
            self._delivery.sender.start_heartbeat(self._delivery.heartbeat)
        self._finished = False
        self._warned_nonfinite = False
        self._resumed: bool | None = None
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
        dataset: str | Mapping[str, Any] | None = None,
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
        provenance: bool | None = None,
    ) -> Run:
        """Register a new run and return it.

        ``project`` is the project's id or slug. With a pipeline key (an API key
        limited to a project) it can be left out: the server puts the run in the
        key's project, whatever ``project`` says. ``model`` travels
        under ``metadata``. A dataset string fills ``datasetId``; a mapping fills
        the structured ``dataset`` reference, including its pinned revision.

        With ``training_uuid`` (or ``VISIN_TRAINING_UUID``) naming a run that
        already exists, that run is resumed rather than duplicated, so a
        restarted job carries on where it was; :attr:`resumed` says which
        happened.

        The run records what it was started from, so it can be reproduced: the git commit,
        branch and whether the tree was dirty, the command line (credentials redacted), the
        installed packages and the machine. ``provenance=False`` or ``VISIN_PROVENANCE=0`` sends none of it.

        Raises :class:`~visin.errors.ApiError` when Visin refuses the run.
        """
        full_name = (name or "").strip()
        if not full_name:
            raise ConfigurationError("a run needs a name")
        settings = read_settings(url=url, token=token, project=project, mode=mode, directory=directory)
        if settings.effective_mode == "disabled" or rank() != 0:
            return cls._disabled(settings)

        run_uuid = training_uuid or settings.training_uuid or str(uuidlib.uuid4())
        payload = run_payload(
            run_uuid,
            full_name,
            project=settings.project,
            dataset=dataset,
            model=model,
            config_id=config_id,
            provenance=provenance_info.collect() if provenance_info.enabled(provenance) else None,
            description=description,
            tags=tags,
            metadata=metadata,
        )

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
            return _remember(cls._disabled(settings))
        training_uuid = training_uuid or settings.training_uuid
        if not training_uuid:
            raise ConfigurationError(
                "no run to attach to: pass a training UUID or set VISIN_TRAINING_UUID. "
                "Use visin.init(...) to register a run instead."
            )
        return _remember(
            cls._build(
                settings, training_uuid, strict=strict, system_metrics=system_metrics, mark_status=mark_status
            )
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
            elif settings.token and not settings.url:
                logger.warning(
                    "visin: a token is set but VISIN_URL is not, so nothing is reported "
                    "(hosted Visin: %s; or run `visin login`)",
                    HOSTED_URL,
                )
            elif settings.url and not settings.token:
                logger.warning("visin: VISIN_URL is set but VISIN_TOKEN is not, so nothing is reported")
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
            settings=settings,
        )

    def _start(self, payload: dict[str, Any]) -> None:
        """Register the run, synchronously, so it exists before its first epoch."""
        op = {"op": "create_run", "body": payload}
        delivery = self._delivery
        if delivery.client is None:
            delivery.keep(op)
            assert delivery.spool is not None
            logger.info(
                "visin: run %s is offline; reports are kept in %s for `visin sync`",
                self.training_uuid,
                delivery.spool.path,
            )
            return
        try:
            delivery.register(op)
            self._resumed = self.training_uuid in delivery.context.existing
            if self._resumed:
                logger.info(
                    "visin: resumed run %s (%s) %s", payload["name"][:60], self.training_uuid, self.url or ""
                )
                # An earlier attempt may have left it completed or failed; it
                # is running again, and should say so while it trains.
                self._emit(
                    {"op": "update", "training_uuid": self.training_uuid, "body": {"status": "running"}}
                )
            else:
                logger.info(
                    "visin: created run %s (%s) %s", payload["name"][:60], self.training_uuid, self.url or ""
                )
        except VisinError as exc:
            if delivery.spool is not None and worth_retrying_later(exc):
                delivery.keep(op)
                delivery.start_spooling()
                logger.warning(
                    "visin: could not reach Visin to create run %s (%s); reports are kept in %s and "
                    "will be sent when it answers",
                    self.training_uuid,
                    exc,
                    delivery.spool.path,
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
        if self._delivery.sender is not None:
            self._delivery.sender.stop(timeout=5.0)
        self._delivery.close()
        self._delivery = Delivery(None, None, self.training_uuid)

    # ---------------------------------------------------------------- state

    @property
    def mode(self) -> str:
        """``online``, ``offline`` or ``disabled``: what this run is doing with its reports."""
        return self._delivery.mode

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
    def url(self) -> str | None:
        """The run's page in Visin's web app, once the server has told us its id.

        ``None`` for an offline or disabled run, or a deployment whose app address is not known:
        set ``VISIN_APP_URL`` for one that is not the hosted Visin.
        """
        return self._settings.run_link(self.training_id) if self._settings else None

    @property
    def training_id(self) -> str | None:
        """The run's database id, once the server has told us."""
        if not self.training_uuid:
            return None
        return self._delivery.context.training_ids.get(self.training_uuid)

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

        Returns ``None`` when the report was not taken: the run is disabled or
        finished, the results were refused, or the queue was full.

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
        return ep_uuid if self._emit({"op": "epoch", "body": body}) else None

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
        return test_uuid if self._emit({"op": "test_result", "body": body}) else None

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
            payload = benchmark_payload(
                results,
                system_info,
                training_uuid=self.training_uuid,
                epoch=epoch,
                epoch_uuid=epoch_uuid,
                name_epoch=epoch_uuid_for,
                timestamp=timestamp,
            )
            payload["benchmark_uuid"] = str(uuidlib.uuid4())
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
                "config_uuid": str(uuidlib.uuid4()),
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
        path: Any,
        kind: str = "prediction",
        *,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        mimetype: str | None = None,
        epoch_uuid: str | None = None,
    ) -> None:
        """Store a rendered frame against an epoch: a prediction overlay, a
        segmentation map, a ground-truth comparison.

        ``path`` is a file, or an image held in memory: a NumPy array or PyTorch
        tensor (``HxW``, ``HxWx3``, ``HxWx4``, or channels first), a PIL image, or
        a Matplotlib figure. An image is stored as PNG under ``name``, by default
        ``<kind>.png``.

        ``kind`` is your own name for what it shows; the Visualizations tab
        filters by it, and by file name, so render the same inputs under the
        same names each time. The file or image is copied at the call, so it may
        be overwritten straight after. ``epoch_uuid`` names the epoch directly,
        as for :meth:`log_test_results`.
        """
        if not self._accepting() or not self.training_uuid:
            return
        op: dict[str, Any] | None = None
        try:
            filename, content_type, op = self._visualization_source(path, kind, name, mimetype)
            body: dict[str, Any] = {
                "epoch_uuid": epoch_uuid or epoch_uuid_for(self.training_uuid, epoch_number(epoch)),
                "filename": filename,
                "type": kind,
                "mimetype": content_type,
            }
            if metadata:
                body["metadata"] = dict(metadata)
            body["visualization_uuid"] = str(
                uuidlib.uuid5(uuidlib.NAMESPACE_URL, f"visin:{body['epoch_uuid']}:{kind}:{filename}")
            )
            op["body"] = self._jsonable(body, "visualization")
        except (ConfigurationError, OSError, TypeError, ValueError, ImportError) as exc:
            if op is not None:
                discard_staged(op, self._delivery.context)
            self._handle(exc, "upload visualization")
            return
        self._emit(op)

    def _visualization_source(
        self, path: Any, kind: str, name: str | None, mimetype: str | None
    ) -> tuple[str, str, dict[str, Any]]:
        """The file name, type and op (without its body) for what ``upload_visualization`` was given."""
        spool = self._delivery.spool
        if is_image(path):
            if spool is None:
                raise ConfigurationError(
                    "an image in memory needs disk to be kept on: save it to a file instead"
                )
            filename = name or f"{kind}.png"
            if not filename.lower().endswith(".png"):
                filename += ".png"
            return (
                filename,
                "image/png",
                {"op": "visualization", "staged": spool.stage_bytes(encode_png(path), filename)},
            )
        source = os.fspath(path)
        extension = os.path.splitext(source)[1].lower()
        content_type = mimetype or VISUALIZATION_TYPES.get(extension)
        if not os.path.isfile(source):
            raise ConfigurationError(f"no such file: {source}")
        if content_type is None:
            raise ConfigurationError(
                "Visin stores PNG, JPEG, GIF, WebP, MP4, MOV and PDF files, "
                f"not {extension or 'a file without an extension'}: {source}"
            )
        op: dict[str, Any] = {"op": "visualization"}
        if spool is not None:
            op["staged"] = spool.stage(source)
        else:
            op["path"] = os.path.abspath(source)
        return name or os.path.basename(source), content_type, op

    def log_model(
        self,
        path: str | os.PathLike[str],
        repo: str,
        *,
        epoch: int | None = None,
        path_in_repo: str | None = None,
        private: bool = True,
        card: bool = True,
        safetensors: bool = False,
    ) -> str | None:
        """Upload a checkpoint to the Hugging Face Hub and link it to this run.

        ``path`` is a file or a folder; ``repo`` is the model repo, ``org/name``,
        created when it does not exist (private, unless ``private=False``).
        Visin stores only a pointer to the commit this upload made, so the run
        keeps naming exactly these bytes. ``epoch`` says which epoch the
        checkpoint came from, and ``path_in_repo`` where in the repo it goes.

        With ``card=True`` (the default) Visin also writes the repo's README from this
        run: dataset, epochs, results, test scores and speed, as the Hub's
        ``model-index`` so the scores show on the model page. It is added only when
        the repo has no README yet, so one you wrote is never replaced.

        With ``safetensors=True`` a PyTorch checkpoint file that is a plain state dict is also uploaded as
        ``model.safetensors`` beside the original, for tools that prefer the format. It needs ``torch``
        and ``safetensors`` installed, and holds the tensors only: metadata saved beside them stays in
        the original.

        The upload uses your own Hugging Face token (``HF_TOKEN`` or ``huggingface-cli
        login``) and needs ``pip install 'visin[hf]'``. The project must keep its
        storage on Hugging Face (project settings), or Visin refuses the link and
        the checkpoint is on the Hub but not shown on the run. Returns the commit
        hash, or ``None`` when nothing was uploaded; a failure is logged, never
        raised into training.
        """
        if not self._accepting():
            return None
        try:
            source = Path(path).expanduser()
            if not source.exists():
                raise FileNotFoundError(f"no such checkpoint: {source}")
            number = None if epoch is None else epoch_number(epoch)
            if isinstance(number, float):
                raise ValueError("a checkpoint's epoch must be a whole number")
            stored = path_in_repo or (None if source.is_dir() else source.name)
            readme = self._model_card(repo, number) if card else None
            revision = hub.upload_model(
                source,
                repo,
                path_in_repo=path_in_repo,
                private=private,
                card=readme,
                safetensors=safetensors,
            )
        except (VisinError, OSError, TypeError, ValueError) as exc:
            self._handle(exc, "upload model")
            return None
        body = model_payload(repo, revision, path=stored, epoch=number)
        self._emit({"op": "model", "training_uuid": self.training_uuid, "body": body})
        return revision

    def _model_card(self, repo: str, epoch: int | float | None) -> str | None:
        client = self._delivery.client
        if client is None or not self.training_uuid:
            return None
        try:
            self.flush()
            return fetch_model_card(
                client,
                self._delivery.context,
                self.training_uuid,
                repo,
                None if epoch is None else int(epoch),
            )
        except VisinError as exc:
            logger.info("visin: no model card for %s: %s", repo, exc)
            return None

    def update(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        tags: Iterable[str] | str | None = None,
        metadata: Mapping[str, Any] | None = None,
        notes: str | None = None,
    ) -> None:
        """Change the run's name, description, tags, metadata or notes.

        ``metadata`` replaces what the run has; it is not merged. ``notes`` is your own commentary on the
        run (up to 5,000 characters), apart from the description, which says what the run is; an empty
        string removes it.
        """
        if not self._accepting() or not self.training_uuid:
            return
        body = update_body(name=name, description=description, tags=tags, metadata=metadata, notes=notes)
        if name is not None and name.strip():
            self.name = name.strip()
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

    def _emit(self, op: dict[str, Any]) -> bool:
        try:
            return self._delivery.emit(op)
        except OSError as exc:
            self._handle(exc, f"keep {op['op']} on disk")
            return False

    # ---------------------------------------------------------------- lifecycle

    def flush(self, timeout: float = 30.0) -> bool:
        """Wait until every report made so far has been attempted."""
        return self._delivery.flush(timeout)

    def finish(self, status: str = "completed", *, timeout: float = 60.0) -> None:
        """Send what is left and mark the run finished. Safe to call twice.

        A ``status`` that is not one of ``pending``, ``running``, ``completed``
        or ``failed`` is reported and the run is marked failed: a run that
        ended on a mistake has not shown that it completed.
        """
        if self._finished or self.mode == "disabled":
            self._finished = True
            return
        if status not in STATUSES:
            self._handle(ValueError(f"status must be one of {', '.join(STATUSES)}, not {status!r}"), "finish")
            status = "failed"
        assert self.training_uuid
        if self._mark_status:
            self._emit(
                {
                    "op": "update",
                    "training_uuid": self.training_uuid,
                    "body": {"status": status, "endTime": now()},
                }
            )
        self._finished = True
        atexit.unregister(self._at_exit)

        flushed = self._delivery.stop(timeout)
        self._delivery.summarise(status if self._mark_status else "done", flushed)
        self._delivery.close()

    def fail(self, error: BaseException | str | None = None) -> None:
        """Mark the run failed, keeping whatever it already reported."""
        if error and self.enabled:
            logger.warning("visin: run failed: %s", error)
        self.finish(status="failed")

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


_current: list[Run] = []


def get_run() -> Run:
    """The run most recently made by :func:`init` or :meth:`Run.attach` in this process.

    A disabled run when there is none, so code that logs through it still runs
    where nothing was started. A finished run is returned as it is, and warns
    when it is used, which is more use than silence.
    """
    return _current[-1] if _current else Run.disabled()


def _remember(run: Run) -> Run:
    _current[:] = [run]
    return run


def init(
    name: str | None = None,
    *,
    project: str | None = None,
    dataset: str | Mapping[str, Any] | None = None,
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
    provenance: bool | None = None,
) -> Run:
    """Start reporting this script as a Visin run: the one call most scripts need.

    Reads ``VISIN_URL`` and ``VISIN_TOKEN`` (and ``VISIN_PROJECT``) from the
    environment unless given here, and returns a disabled run that reports
    nothing when there is no server to report to. Always registers a run,
    named ``name`` or after the script and the time; with ``training_uuid`` (or
    ``VISIN_TRAINING_UUID``) naming a run that exists, that run is resumed.
    To report into a run without registering it, use :meth:`Run.attach`.
    """
    return _remember(
        Run.create(
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
            provenance=provenance,
        )
    )
