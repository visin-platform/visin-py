"""PyTorch Lightning: report each epoch of ``trainer.fit`` to Visin.

    from visin.integrations.lightning import VisinCallback

    trainer = L.Trainer(max_epochs=50, callbacks=[VisinCallback(name="segformer b2", project="road-seg")])
    trainer.fit(model, datamodule=data)

Metrics logged with ``self.log`` arrive split by prefix: ``val_loss`` or
``val/loss`` becomes the validation curve's ``loss``, ``train_loss`` the
training curve's. The module's hyperparameters become the run's config, and
``trainer.test`` metrics become a test result on the last epoch.
Only global rank zero reports.

Lightning keeps the last logged value of every metric, so the validation of an
earlier epoch would be reported again on an epoch that did not validate (with
``check_val_every_n_epoch=5``, say). The validation curve is therefore sent only
on an epoch whose validation ran. The learning rate is the first optimizer's
first parameter group's.

A run started here is started by ``trainer.fit`` only: a ``trainer.test`` or
``validate`` on its own does not make a run. Calling ``fit`` again, to resume,
starts a new one.
"""

from __future__ import annotations

import logging
import time
from typing import Any

try:
    from lightning.pytorch.callbacks import Callback
except ImportError:
    try:
        from pytorch_lightning.callbacks import Callback  # type: ignore[no-redef]
    except ImportError as exc:
        raise ImportError("visin.integrations.lightning needs Lightning: pip install lightning") from exc

from ..run import Run, init
from ._metrics import split_metrics

logger = logging.getLogger("visin")


def _learning_rate(trainer: Any) -> float | None:
    try:
        return float(trainer.optimizers[0].param_groups[0]["lr"])
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None


class VisinCallback(Callback):  # type: ignore[misc]
    """Report each training epoch, and test results.

    Pass ``run`` to report into a run you made. Otherwise one is started with
    :func:`visin.init` at setup, given any other keyword arguments (``name``,
    ``project``, ``tags``...), and finished after ``finish_after``: ``"fit"``
    by default, or ``"test"`` when ``trainer.test`` follows ``trainer.fit``
    and its results should reach the run too.
    """

    def __init__(
        self,
        run: Run | None = None,
        *,
        first_epoch: int = 1,
        finish_after: str = "fit",
        log_hyperparameters: bool = True,
        **init_kwargs: Any,
    ):
        super().__init__()
        if finish_after not in ("fit", "test", "never"):
            raise ValueError("finish_after must be 'fit', 'test' or 'never'")
        self.run = run
        self._owns_run = run is None
        self._init_kwargs = init_kwargs
        self._first_epoch = first_epoch
        self._finish_after = finish_after
        self._log_hyperparameters = log_hyperparameters
        self._started: float | None = None
        self._validated_epoch: int | None = None
        self._finished = False

    def setup(self, trainer: Any, pl_module: Any, stage: str) -> None:
        if not trainer.is_global_zero or stage != "fit" or not self._owns_run:
            return
        if self.run is not None and not self._finished:
            return
        self._finished = False
        self._validated_epoch = None
        self.run = init(**self._init_kwargs)
        hparams = getattr(pl_module, "hparams", None)
        if self._log_hyperparameters and hparams:
            self.run.log_config(dict(hparams), name=type(pl_module).__name__)

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        self._started = time.monotonic()

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        # Lightning validates inside the training epoch, so by now the
        # callback metrics hold this epoch's validation results too.
        if self.run is None or trainer.sanity_checking or not trainer.is_global_zero:
            return
        groups = split_metrics(trainer.callback_metrics)
        if self._validated_epoch != trainer.current_epoch:
            groups.pop("val", None)
        if not groups.get("train") and not groups.get("val"):
            return
        self.run.log_epoch(
            trainer.current_epoch + self._first_epoch,
            train=groups.get("train"),
            val=groups.get("val"),
            learning_rate=_learning_rate(trainer),
            epoch_time=time.monotonic() - self._started if self._started is not None else None,
        )

    def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
        if not trainer.sanity_checking:
            self._validated_epoch = trainer.current_epoch

    def on_test_end(self, trainer: Any, pl_module: Any) -> None:
        if self.run is None or not trainer.is_global_zero:
            return
        test = split_metrics(trainer.callback_metrics).get("test")
        if test:
            if self.run.last_epoch is None:
                # A test result belongs to an epoch, and this run reported none.
                logger.info("visin: test metrics, but no epoch to attach them to; not sent")
            else:
                self.run.log_test_results(self.run.last_epoch, {"overall": test})
        if self._finish_after == "test":
            self._finish()

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        if self._finish_after == "fit":
            self._finish()

    def on_exception(self, trainer: Any, pl_module: Any, exception: BaseException) -> None:
        if self._owns_run and self.run is not None:
            self.run.fail(exception)

    def _finish(self) -> None:
        if self._owns_run and self.run is not None and not self._finished:
            self.run.finish()
            self._finished = True
