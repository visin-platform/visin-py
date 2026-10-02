"""Keras: report each epoch of ``model.fit`` to Visin.

    from visin.integrations.keras import VisinCallback

    model.fit(x, y, validation_data=(vx, vy), epochs=20,
              callbacks=[VisinCallback(name="unet baseline", project="road-seg")])

``val_`` metrics become the validation curve and the rest the training curve,
so ``loss`` and ``val_loss`` land on one loss chart.
"""

from __future__ import annotations

import logging
import time
from typing import Any

try:
    import keras

    _Callback: Any = keras.callbacks.Callback
except ImportError:  # Keras 2, inside TensorFlow
    try:
        from tensorflow import keras  # type: ignore[no-redef]

        _Callback = keras.callbacks.Callback
    except ImportError as exc:
        raise ImportError("visin.integrations.keras needs Keras: pip install keras") from exc

from .._internal.serialize import to_jsonable
from ..run import Run, init
from ._metrics import split_metrics

logger = logging.getLogger("visin")


def _learning_rate(model: Any) -> float | None:
    try:
        rate = model.optimizer.learning_rate
        if callable(rate):  # a schedule
            rate = rate(model.optimizer.iterations)
        return float(to_jsonable(rate))
    except Exception:
        return None


class VisinCallback(_Callback):  # type: ignore[misc]
    """Report each epoch of training.

    Pass ``run`` to report into a run you made. Otherwise one is started with
    :func:`visin.init` when training begins, given any other keyword arguments
    (``name``, ``project``, ``tags``...), and finished when training ends.
    Epochs are numbered from ``first_epoch``, 1 by default, as Keras prints them.
    ``fit(initial_epoch=...)`` carries on from there. A run this callback made is
    finished when training ends, so calling ``fit`` again starts a new one.
    """

    def __init__(self, run: Run | None = None, *, first_epoch: int = 1, **init_kwargs: Any):
        super().__init__()
        self.run = run
        self._owns_run = run is None
        self._init_kwargs = init_kwargs
        self._first_epoch = first_epoch
        self._started: float | None = None

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        if self.run is None:
            self.run = init(**self._init_kwargs)

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        self._started = time.monotonic()

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if self.run is None:
            return
        logs = dict(logs or {})
        rate = logs.get("learning_rate", logs.get("lr"))
        if rate is None:
            rate = _learning_rate(self.model)
        groups = split_metrics(logs)
        if not groups.get("train") and not groups.get("val"):
            return
        self.run.log_epoch(
            epoch + self._first_epoch,
            train=groups.get("train"),
            val=groups.get("val"),
            learning_rate=rate,
            epoch_time=time.monotonic() - self._started if self._started is not None else None,
        )

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        if self._owns_run and self.run is not None:
            self.run.finish()
            self.run = None
