# Keras, Lightning and Hugging Face

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

Epochs are numbered from 1, as Keras prints them, and `fit(initial_epoch=…)` carries on from there. A
callback that made its run finishes it when training ends, so a second `fit` starts a new run. The learning rate is read from the logs, or from the
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
- Lightning keeps the last logged value of every metric. The validation curve is sent only for an
  epoch whose validation ran, so with `check_val_every_n_epoch=5` the other epochs show a gap, not
  the same number five times.
- Only `trainer.fit` starts a run. A `trainer.test` on its own makes none, and a second `fit`, to
  resume, starts a new run.
- The learning rate is the first optimizer's first parameter group's.

Works with both `lightning` and `pytorch_lightning`.

## Hugging Face Trainer

```python
from transformers import Trainer
from visin.integrations.huggingface import VisinCallback

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train,
    eval_dataset=val,
    callbacks=[VisinCallback(name="segformer b2", project="road-seg")],
)
trainer.train()
```

The `Trainer` logs by step, and Visin's charts are per epoch, so one epoch is reported at the end of
each, from the last values logged during it.

- The training log (`loss`, `grad_norm`, …) becomes the training curve, and the `eval_` metrics the
  validation curve. Runtimes and throughput (`eval_runtime`, `train_samples_per_second`) are left out.
- With `eval_strategy="epoch"` the evaluation runs after the epoch ends, so the epoch is held until it
  has been attached, then sent. An evaluation by step goes to the epoch it falls in.
- The `TrainingArguments` become the run's config, and the learning rate comes from the training log.
- `trainer.predict` metrics (`test_`) become a test result on the last epoch. The run is finished when
  training ends, so when a `predict` follows `train`, make the run yourself and pass it with
  `VisinCallback(run=run)`.
- Only the main process reports.
- A run that stops mid-epoch (`max_steps`) still sends its last, partial epoch.

## Anything else

For any other loop, call `run.log_epoch` where the epoch ends. See
[What a run records](reporting.md).
