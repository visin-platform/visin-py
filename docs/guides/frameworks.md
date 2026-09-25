# Keras and Lightning

Each integration is a callback that reports every epoch. Metric names are sorted into Visin's two
curves by prefix: `val_loss` and `val/loss` become the validation curve's `loss`, and `loss` or
`train_loss` become the training curve's.

Without `run=`, a callback starts its own run with `visin.init` and passes it any keyword you give
(`name`, `project`, `tags`, `system_metrics`…). It also finishes that run when training ends. A run
you pass in is yours to finish.

## Keras

```python
from visin.integrations.keras import VisinCallback

model.fit(
    x,
    y,
    validation_data=(vx, vy),
    epochs=20,
    callbacks=[VisinCallback(name="unet baseline", project="road-seg")],
)
```

Epochs are numbered from 1, as Keras prints them. The learning rate is read from the logs, or from the
optimizer, schedules included. Keras names its `MeanIoU` metric `mean_io_u`; it is sent as
`mean_iou`, the name Visin's chart reads.

Works with Keras 3, and with Keras 2 inside TensorFlow.

## PyTorch Lightning

```python
import lightning as L
from visin.integrations.lightning import VisinCallback

trainer = L.Trainer(max_epochs=50, callbacks=[VisinCallback(name="segformer b2", project="road-seg")])
trainer.fit(model, datamodule=data)
```

- Whatever you log with `self.log` is reported at the end of each training epoch, together with that
  epoch's validation metrics. Lightning's `_step` copies are left out, and its `_epoch` copies lose
  the suffix.
- The module's `hparams` become the run's config.
- `test_` metrics from `trainer.test` become a test result on the last epoch. The run is finished at
  the end of `fit`, so when a test follows, pass `finish_after="test"`.
- Only global rank zero reports, and the sanity-check pass is skipped.

Works with both `lightning` and `pytorch_lightning`.

## Anything else

For any other loop, call `run.log_epoch` where the epoch ends. See
[What a run records](reporting.md).
