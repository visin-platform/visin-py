# Reading runs back

`visin.Api` reads what Visin holds, for notebooks and analysis scripts. Use a user API key with read
scopes (**Account → API keys**). A pipeline key also works, but sees only its own project.

```python
from visin import Api

api = Api()  # VISIN_URL and VISIN_TOKEN, or Api(url=..., token=...)

for run in api.trainings(project="road-seg", status="completed", tags=["ablation"], limit=50):
    print(run.name, run.uuid)
```

`trainings` fetches one page at a time as you iterate, newest first.

## One run in depth

```python
run = api.training("5f0c6a2e-…")  # by UUID
epochs = api.epochs(run)  # every epoch, in order
tests = api.test_results(run)
benchmarks = api.benchmarks(run)
```

Each call returns typed objects (`Training`, `Epoch`, `TestResult`, `Benchmark`, `Project`) with
Python field names: `run.project_id`, `epoch.learning_rate`. `.raw` on each is the server's JSON, for
anything the class does not model. Methods that take a run accept a `Training` or its UUID.

## Finding a run

```python
run = api.find("unet baseline", project="road-seg")  # the newest run with exactly that name, or None
run = api.find(tags=["ablation"], status="completed")  # or the newest that matches filters
api.tags()  # every tag in use
```

A later script can pick up an earlier run by name, without having kept its UUID.

## How a run did, in one call

```python
summary = api.summary(run)
iou = summary.metric("val.mean_iou")
iou.best_value, iou.best_epoch, iou.last_value  # the best epoch beside the last
summary.models, summary.provenance  # linked Hub models; the code, command and machine
```

A run's last epoch is not its result, so every result comes with its best epoch and its last. "Best" follows
the project's metric directions; read `iou.direction_from`: `taxonomy` means the project said which way is
better, `default` means Visin guessed from the name (a loss or a latency is lower-is-better, the rest higher).

## Findings, comparisons and which way a metric is better

```python
for finding in api.findings(project="road-seg", limit=10):  # newest first; run="<uuid>" for one run's
    print(finding.title, finding.author_kind)

for comparison in api.comparisons(project="road-seg", type="trainings"):
    print(comparison.name, comparison.item_ids)

project = api.project("road-seg")
project.direction("val.mean_iou")  # "higher", "lower", or None when the project has not said
```

Visin does not know whether a result is better high or low unless the project says so in its taxonomy. Ask
`project.direction(...)` before ranking runs, and treat `None` as "you are guessing".

## What a run was launched with, and its frames

```python
config = api.config(run)  # the dict logged with log_config, or None
config.config["lr"]

for frame in api.visualizations(run, kind="overlay", epoch=99):  # newest first, each with a signed link
    api.download_visualization(frame, "frames/")  # saved under its file name
```

`kind` and `epoch` are optional filters. Signed links expire an hour after listing, so download soon after.

## As a DataFrame

With `pip install 'visin[pandas]'`:

```python
frame = api.epochs_frame(run)
frame[["train.loss", "val.loss"]].plot()
best = frame["val.mean_iou"].idxmax()
```

Several runs side by side, one metric each:

```python
api.compare_frame([run_a, run_b, "5f0c6a2e-…"], "val.mean_iou").plot()
```

There is one row per epoch, indexed by epoch number, with results flattened into columns such as
`val.loss` and `val.car.iou`. A metric an epoch did not report is `NaN`, just as it is a gap in
Visin's charts. `visin.flatten` does the flattening, if you want it on its own.

## Models a run kept

```python
run = api.training("7c8564a3-bde1-4771-8376-6708002a50ac")
for model in run.stored_models:  # or api.models(run)
    print(model.format, model.size, model.status, model.license.get("id"))

path = api.download_model(run, file_format="onnx", directory="models/")
outputs = api.model_outputs(run, file_format="pth")
print(outputs.epoch, outputs.totals)  # e.g. 99 {'mask': 120, 'segment': 120, 'confidence': 120}
folder = api.download_model(run, file_format="pth", directory="models/", outputs=True)
```

`stored_models` are the checkpoints saved with `Run.save_model`; a model the run only links on the Hub is in
`run.models`. `download_model` checks the file against its recorded SHA-256, resumes an interrupted
download, and refuses a model that is not offered (`model.refusal` says why). `model_outputs` describes what the
model produced on its run's frames: the checkpoint and its licence, the data it was trained and run on, how to read
each kind of output, and the epoch's results (all of it in `outputs.raw`). With `outputs=True`, `download_model`
saves that answer as `outputs.json` beside the frames of the model's epoch, in `mask/`, `segment/` and
`confidence/`, and its `progress` counts frames. `trainings(dataset=...)` finds
the runs trained on a dataset, so `for run in api.trainings(dataset=dataset_id): run.stored_models` lists
what came out of it.

