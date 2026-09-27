# Reading runs back

`visin.Api` reads what Visin holds, for notebooks and analysis scripts. Use a user API key with read
scopes (**Account → API keys**). A pipeline key also works, but sees only its own project.

```python
from visin import Api

api = Api()  # VISIN_URL and VISIN_TOKEN, or Api(url=..., token=...)

for run in api.trainings(project="road-seg", status="completed", tags=["ablation"], limit=50):
    print(run["name"], run["uuid"])
```

`trainings` fetches one page at a time as you iterate, newest first.

## One run in depth

```python
run = api.training("5f0c6a2e-…")  # by UUID, or by its 24-character id
epochs = api.epochs(run)  # every epoch, in order
tests = api.test_results(run)
benchmarks = api.benchmarks(run)
```

Each call returns plain dicts, shaped as the [API reference](https://github.com/visin-platform/visin)
describes them.

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

## Anything else

`api.get(path, **params)` calls any `GET` endpoint under `/api` and returns its unwrapped `data`.
