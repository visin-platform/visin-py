# visin

`visin` sends a training run to [Visin](https://github.com/visin-platform/visin): its epochs, test
results, benchmarks, prediction frames and config. It can also read runs back for analysis.

```sh
pip install visin
```

```python
import visin

with visin.init("unet baseline", project="road-seg") as run:
    for epoch in range(1, 21):
        train_loss, val_loss, val_iou = train_one_epoch()
        run.log_epoch(epoch, train={"loss": train_loss}, val={"loss": val_loss, "mean_iou": val_iou})
```

## Why use it rather than `requests`

Visin's HTTP API is small, and the [quickstart](https://github.com/visin-platform/visin) posts to it
with `requests`. This package does the same, plus the things a long training job needs:

- **It never stops your training.** Reports are sent from a background thread. A failure is logged
  and counted, never raised into your loop.
- **Retries are safe.** Each epoch and test result carries an id of its own, so a retried request
  is recognised and never stored twice. Only requests that are safe to repeat get repeated.
- **It survives losing the network.** When Visin stops answering, reports wait on disk and are sent
  once it answers again. On a machine with no route to Visin at all, run offline and send everything
  later with `visin sync`.
- **Crashed and cancelled runs keep their epochs.** A run that crashes, or that the scheduler
  terminates, still sends what it has and is marked failed.
- **Metrics arrive as they are.** NumPy and PyTorch scalars are converted. NaN becomes a gap in the
  chart instead of an error that loses the whole epoch.
- **The same script runs anywhere.** With no server configured, every call does nothing.

## Where to next

- [Getting started](getting-started.md): a token, a project and your first run, in five minutes.
- [What a run records](guides/reporting.md): epochs, test results, benchmarks, frames and configs.
- [Offline and unreliable networks](guides/offline.md): clusters, compute nodes and `visin sync`.
- [Keras and Lightning](guides/frameworks.md): one callback, no loop changes.
- [Reading runs back](guides/reading.md): runs and their epochs as dicts or a pandas DataFrame.
