"""Hugging Face ``Trainer``: report each epoch to Visin.

    from visin.integrations.huggingface import VisinCallback

    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val,
                      callbacks=[VisinCallback(name="segformer b2", project="road-seg")])

The ``Trainer`` logs by step, and Visin's charts are per epoch, so one epoch
report is made at each epoch's end from the last values logged during it:
``loss`` and the rest of the training log become the training curve, and the
``eval_`` metrics the validation curve. With ``eval_strategy="epoch"`` the
evaluation runs *after* the epoch-end event, so an epoch is held until that
evaluation has attached to it, and sent when the next epoch starts or training
ends. An evaluation by step is attached to the epoch it falls in.

``trainer.predict`` metrics (``test_``) become a test result on the last epoch.
Because training ends the run this callback made, pass your own ``run`` when a
``predict`` follows ``train``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

try:
    from transformers import TrainerCallback as _Callback
except ImportError as exc:
    raise ImportError("visin.integrations.huggingface needs transformers: pip install transformers") from exc

from ..run import Run, init
from ._metrics import split_metrics

logger = logging.getLogger("visin")

_NOISE = ("_runtime", "_per_second", "total_flos")


def _metrics(logs: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    kept = {
        name: value
        for name, value in (logs or {}).items()
        if isinstance(value, (int, float)) and not any(noise in name for noise in _NOISE)
    }
    return split_metrics(kept)


class VisinCallback(_Callback):  # type: ignore[misc]
    """Report each epoch of ``Trainer`` training.

    Pass ``run`` to report into a run you made. Otherwise one is started with
    :func:`visin.init` when training begins, given any other keyword arguments
    (``name``, ``project``, ``tags``...), with the ``TrainingArguments`` as its
    config, and finished when training ends, so a second ``train`` starts a new run.
    """

    def __init__(self, run: Run | None = None, **init_kwargs: Any):
        super().__init__()
        self.run = run
        self._owns_run = run is None
        self._init_kwargs = init_kwargs
        self._train: dict[str, Any] = {}
        self._rate: float | None = None
        self._val: dict[str, Any] = {}
        self._pending: dict[str, Any] | None = None
        self._started: float | None = None
        self._last_epoch = 0

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if not getattr(state, "is_world_process_zero", True):
            return
        if self.run is None:
            self.run = init(**self._init_kwargs)
            if self.run.enabled:
                self.run.log_config(args.to_dict() if hasattr(args, "to_dict") else args)

    def on_epoch_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._flush()
        self._started = time.monotonic()

    def on_log(
        self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        logs = logs or {}
        if "loss" not in logs and "learning_rate" not in logs:
            return
        self._train.update(_metrics(logs).get("train", {}))
        if isinstance(logs.get("learning_rate"), (int, float)):
            self._rate = float(logs["learning_rate"])

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        epoch = max(1, round(state.epoch or 0))
        self._pending = {
            "epoch": epoch,
            "train": dict(self._train),
            "val": dict(self._val),
            "learning_rate": self._rate,
            "epoch_time": time.monotonic() - self._started if self._started is not None else None,
        }
        self._train, self._val = {}, {}

    def on_evaluate(
        self, args: Any, state: Any, control: Any, metrics: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        val = _metrics(metrics).get("val", {})
        if self._pending is not None:
            self._pending["val"].update(val)
            self._flush()
        else:
            self._val.update(val)

    def on_predict(
        self, args: Any, state: Any, control: Any, metrics: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        self._flush()
        scores = _metrics(metrics).get("test", {})
        if self.run is not None and scores:
            self.run.log_test_results(self._last_epoch or 1, {"overall": scores})

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self._pending is None and (self._train or self._val) and self.run is not None:
            self._pending = {
                "epoch": max(1, round(state.epoch or 0)),
                "train": dict(self._train),
                "val": dict(self._val),
                "learning_rate": self._rate,
                "epoch_time": None,
            }
        self._flush()
        if self._owns_run and self.run is not None:
            self.run.finish()
            self.run = None

    def _flush(self) -> None:
        pending, self._pending = self._pending, None
        if pending is None or self.run is None:
            return
        if not pending["train"] and not pending["val"]:
            return
        self._last_epoch = pending["epoch"]
        self.run.log_epoch(
            pending["epoch"],
            train=pending["train"] or None,
            val=pending["val"] or None,
            learning_rate=pending["learning_rate"],
            epoch_time=pending["epoch_time"],
        )
