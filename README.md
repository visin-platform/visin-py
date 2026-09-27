# visin

[![CI](https://github.com/visin-platform/visin-py/actions/workflows/ci.yml/badge.svg)](https://github.com/visin-platform/visin-py/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/visin)](https://pypi.org/project/visin/)
[![Python](https://img.shields.io/pypi/pyversions/visin)](https://pypi.org/project/visin/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/visin-platform/visin-py/blob/main/LICENSE)

Send training runs to [Visin](https://github.com/visin-platform/visin): epochs, test results,
benchmarks, prediction frames and configs. Read them back for analysis.

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

Point it at your Visin with two environment variables. Without them, the same script runs and
reports nothing.

```sh
export VISIN_URL=https://vision-api.example.com
export VISIN_TOKEN=…            # a pipeline key, from the project's Settings → Pipeline keys
visin check --write             # confirm this machine can report
```

## What it does for you

- **Never stops your training.** Reports go out from a background thread, and failures are logged,
  never raised.
- **Retries safely.** Epochs and test results carry their own ids, so a retried request is never
  stored twice.
- **Survives the network.** Reports wait on disk while Visin is unreachable, and are sent when it
  answers. With `VISIN_MODE=offline`, a compute node with no route to Visin keeps everything for
  `visin sync`.
- **Keeps crashed runs.** A run that crashes, or that the scheduler terminates, still sends its
  epochs and is marked failed.
- **Takes metrics as they are.** NumPy and PyTorch scalars are converted, and NaN becomes a gap in
  the chart.
- **Reports once in distributed jobs.** Only rank zero reports.

## More than epochs

```python
run.log_test_results(12, {"day": {"car": {"iou": 0.74}}, "night": {"car": {"iou": 0.61}}})
run.log_benchmark({"device": "cuda", "batch_size": 1, "fps": 119.0})  # machine details filled in
run.upload_visualization(12, "renders/overlay_0001.png", kind="overlay")
run.log_config(args)  # argparse, dataclass, pydantic, Hydra…
```

## Frameworks

```python
from visin.integrations.lightning import VisinCallback  # or visin.integrations.keras

trainer = L.Trainer(callbacks=[VisinCallback(name="segformer b2", project="road-seg")])
```

## Reading runs back

```python
api = visin.Api()
for run in api.trainings(project="road-seg", status="completed"):
    frame = api.epochs_frame(run)  # pip install 'visin[pandas]'
```

## Datasets

Datasets on Visin can be listed and downloaded, once, into `VISIN_DATA_DIR`:

```bash
export VISIN_DATASET_URL=https://dataset-api.example.com
visin datasets
visin download zod     # resumes if interrupted; prints the folder
```

`visin.Datasets` does the same from Python.

## Documentation

**[visin-platform.github.io/visin-py](https://visin-platform.github.io/visin-py/)** has the
getting-started guide, guides to offline use and each framework, and the API reference.

## Contributing

See [CONTRIBUTING.md](https://github.com/visin-platform/visin-py/blob/main/CONTRIBUTING.md). In
short: `make install`, then `make check` before a pull request. Changes are listed in the
[changelog](https://github.com/visin-platform/visin-py/blob/main/CHANGELOG.md).

## License

MIT
