# What a run records

Everything below is a method on the `Run` that `visin.init` returns. Every method returns
immediately, and the report is sent in the background, in the order you made it.

## Epochs

```python
run.log_epoch(
    12,
    train={"loss": 0.41},
    val={"loss": 0.47, "mean_iou": 0.58, "car": {"iou": 0.71}, "pedestrian": {"iou": 0.42}},
    learning_rate=1e-4,
    epoch_time=812.5,  # seconds; Visin adds these up, and prices them if the project has rates
)
```

`results` is yours to shape, and anything you send is stored. `train=` and `val=` are shorthand for
`results["train"]` and `results["val"]`. Visin's charts read these names:

| Chart | Reads |
| --- | --- |
| Loss | `train.loss`, `val.loss` |
| Mean IoU | `train.mean_iou`, `val.mean_iou` |
| Pixel accuracy, mean accuracy, Dice | `pixel_accuracy`, `mean_accuracy`, `dice_score` |
| Per class | any key under `train` or `val` whose value holds `iou`, `precision`, `recall`, `f1` or `ap` |

Classes need no setup. The first epoch that reports `car` adds a `car` line.

`log_epoch` returns the epoch's UUID straight away. It is derived from the run and the epoch number,
so the same epoch always has the same UUID, even from another process. That is what makes a retry,
or a restarted job that logs the epoch again, harmless: the server keeps the first copy.

!!! tip "NumPy, PyTorch and NaN"
    Metrics can be NumPy scalars, zero-dimensional tensors or `Decimal`s, and they are converted for
    you. NaN and infinity have no JSON form, so they are sent as `null`, which Visin charts as a gap.
    A warning names them the first time.

### The System tab

`visin.init(..., system_metrics=True)`, or `log_epoch(..., system=True)` for a single epoch, adds a
snapshot of memory and GPU use to each epoch. That fills the run's **System** tab:

- memory used, and the process's peak
- per GPU: memory used, reserved and peak, temperature, power and fan speed

GPU figures come from PyTorch if your script has already imported it, and from `nvidia-smi`
otherwise.

## Test results

Scores on held-out data, attached to an epoch, usually the checkpoint you would ship:

```python
run.log_test_results(
    12,
    {
        "day": {"overall": {"pixel_accuracy": 0.93}, "car": {"iou": 0.74}, "pedestrian": {"iou": 0.47}},
        "night": {"overall": {"pixel_accuracy": 0.86}, "car": {"iou": 0.61}, "pedestrian": {"iou": 0.29}},
    },
)
```

The top-level keys are conditions (one per test set), the keys inside them are classes, and the
numbers inside those are metrics. `overall` holds metrics for a whole condition.

## Benchmarks

How fast the model runs and how big it is, on one or more devices:

```python
run.log_benchmark(
    [
        {
            "device": "cuda",
            "device_type": "RTX 4090",
            "batch_size": 1,
            "mean_time_ms": 8.4,
            "fps": 119.0,
            "total_parameters_m": 27.4,
            "flops_giga": 62.1,
        },
        {"device": "cpu", "batch_size": 1, "mean_time_ms": 142.0, "fps": 7.0},
    ],
    epoch=12,
)
```

Visin requires a description of the machine with each benchmark. Any part of it you leave out is
filled in by `visin.system_info()`: CPU cores, memory, GPU name, memory and driver.

## Prediction frames

```python
run.upload_visualization(12, "renders/overlay_0001.png", kind="overlay")
```

PNG, JPEG, GIF, WebP, MP4, MOV and PDF files are accepted. The file is copied when you call, so you
can render the next epoch's frame over it straight away. `kind` is your own name for what the image
shows, and the Visualizations tab filters by it. Render the same inputs under the same file names
each epoch, and you can follow one image through training.

## Config

```python
run.log_config(args)  # argparse.Namespace, dict, dataclass, pydantic model or Hydra config
```

The config is stored and linked to the run, and shown on its page. Values JSON cannot hold, such as a
loss module, are kept as their `repr`. `visin.init(..., config=cfg)` does the same in one call.

## Changing the run

```python
run.update(tags=["best"], metadata={"best_epoch": 12, "best_val_iou": 0.61})
```

`metadata` replaces what the run had rather than merging with it.

## From a separate test or benchmark script

Research code often trains in one script and tests, renders and benchmarks in others, run afterwards
as separate processes. Each later script reports into the training's run with
`Run.attach(training_uuid, mark_status=False)`:

```python
run = visin.Run.attach(training_uuid, mark_status=False)
with run:
    run.log_test_results(epoch, results, epoch_uuid=epoch_uuid)
```

- **`mark_status=False`** leaves the run's status to the training. Without it, the test script's
  `finish`, or its crash, would mark the training completed or failed.
- **`epoch_uuid=`** names the checkpoint's epoch directly. The training script can put it in the
  checkpoint's file name: `log_epoch` returns it, and `visin.epoch_uuid_for(training_uuid, epoch)`
  gives the same UUID anywhere. `log_test_results`, `log_benchmark` and `upload_visualization` all
  take it.

## Resuming

A job that restarts with the same `training_uuid` reports into its existing run, which
`run.resumed` tells you. The run is marked running again. Epochs it logs again keep their first
recorded values, and visin warns once when that happens. A job whose epoch numbering starts over
should start a new run instead.

## Finishing

`with visin.init(...) as run:` finishes the run for you: completed when the block ends normally,
failed when it raises. Without `with`:

```python
run.finish()  # or run.finish("failed"), or run.fail(exc)
```

A script that ends without calling `finish` is finished at exit: completed if it ended normally,
failed if it crashed or was terminated. `SIGTERM`, which is what a scheduler sends before it kills a
job, is turned into an ordinary exit so this still happens. If your framework already handles
`SIGTERM`, as Lightning does on SLURM, its handler is left alone.

## When things go wrong

Two things raise, both at startup, where you are there to read them: `init` raises `ApiError` when
Visin refuses the run outright (a bad token, a project the key does not cover), and `Run.attach`
raises `ConfigurationError` when it has no run to attach to. Visin being unreachable does not raise;
reports wait on disk.

After that nothing raises into your loop. A report that fails is logged as a warning, and `finish`
prints a summary. It reads like this:

```text
visin: run 5f0c… completed: 118 reports sent, 2 failed, 0 dropped, 0 waiting on disk
```

Pass `strict=True` to `init` to get exceptions instead, which is useful in tests.
