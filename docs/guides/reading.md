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

## As a DataFrame

With `pip install 'visin[pandas]'`:

```python
frame = api.epochs_frame(run)
frame[["train.loss", "val.loss"]].plot()
best = frame["val.mean_iou"].idxmax()
```

There is one row per epoch, indexed by epoch number, with results flattened into columns such as
`val.loss` and `val.car.iou`. A metric an epoch did not report is `NaN`, just as it is a gap in
Visin's charts. `visin.flatten` does the flattening, if you want it on its own.

