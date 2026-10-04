# How it is built

This page is for people working on the package. It covers where each thing lives and why it works
the way it does.

## The files

```text
src/visin/
├── __init__.py          the public API: everything a user imports comes from here
├── run.py               Run and init: everything a training script writes
├── api.py               Api: reading runs back
├── models.py            Training, Epoch, TestResult, Benchmark, Project, Dataset: what Api returns
├── datasets.py          Datasets: list and download
├── offline.py           sync() and pending(): sending reports kept on disk
├── system.py            system_info() and system_metrics()
├── errors.py            VisinError and its subclasses
├── cli.py, __main__.py  the `visin` command
├── integrations/
│   ├── keras.py         Keras callback
│   ├── lightning.py     PyTorch Lightning callback
│   └── _metrics.py      sorting flat metric names into train/val/test
├── _internal/           machinery; not public, may change in any release
│   ├── config.py        settings from the environment and arguments
│   ├── inputs.py        epoch numbers, results and configs in the shapes callers pass them
│   ├── process.py       the process's rank, and learning how it ended (crash, SIGTERM)
│   ├── transport.py     HTTP: retries, response unwrapping, signed uploads
│   ├── serialize.py     NumPy/tensors/NaN into JSON Visin accepts
│   ├── delivery.py      one run's client, spool and sender: live sends, spooling, catch-up
│   ├── payloads.py      the request bodies a run builds from what a caller passed
│   ├── reports.py       what each kind of report is, and how it is delivered
│   ├── sender.py        the background thread reports are sent from
│   └── spool.py         reports on disk: the file format and syncing it
└── _version.py          the version, and the only place it is written
```

The rule of thumb: a module at the top level is something a user touches, and `_internal` is how it
works.

## One report, from call to server

```text
run.log_epoch(...)                       run.py: validate, convert to JSON, build an op
   │   {"op": "epoch", "body": {...}}
   ▼
Sender queue ──► background thread       sender.py: one thread, so order is kept
   │
   ▼
deliver(client, op)                      reports.py: POST /epochs/upload; a 409 counts as delivered
   │
   ├─ delivered ──────────────► done
   ├─ refused (4xx, 500) ─────► logged and counted; retrying would not help
   └─ unreachable (5xx/down) ─► spool.py: appended to ~/.visin/runs/<uuid>.jsonl,
                                  sent by the run once Visin answers, or by `visin sync`
```

Every write is first an **op**: a small JSON dict naming its kind and carrying its request body.
Because an op is data, the same op can be sent at once, written to disk, and sent from disk a week
later by `visin sync`, and all three go through the one `deliver` function. There is a single code
path for sending.

## Decisions worth knowing

**Nothing raises into the training loop.** A metrics backend is not worth a training job. Failures
are logged and counted, and `finish` summarises them. `strict=True` exists for tests. The one
exception is starting a run: a refusal there (bad token, wrong project) raises, because every report
after it would fail the same way and the person launching the job is there to read it.

**Creating and attaching are separate.** `init` always registers a run, resuming it when its UUID is
known; `Run.attach` reports into one that exists. Which one you get never depends on whether you
passed a name.

**Retries depend on the request, not on its method.** `transport.py` retries on its own terms
instead of leaving it to urllib3, because a POST is only safe to repeat if it carries its own id.
Runs, epochs and test results do: epoch UUIDs are `uuid5(run, epoch)`, and a repeat of a run or an epoch is answered 409. A test result is an evaluation, so a repeat of one is answered with the stored one (200), and only a different result under its uuid is refused (409).
Benchmarks and configs do not, so they are retried only when the request provably never reached
the server: the connection failed, or the rate limiter or auth refused it before any handler ran.
500 is never retried, as Visin documents.

**One sender thread.** Writes depend on each other: a run is created before its epochs, and an
epoch before its test results. Keeping them in order costs less than recovering when they arrive
out of order.

**The run is created synchronously.** `init` returns only once the server has the run, or once it
is known to be unreachable and the creation has been kept on disk. So an epoch can never arrive
before its run.

**Files are copied at the call.** Scripts render each epoch's frames under the same names, so the
original may hold the next epoch's image by the time a queued upload runs.

**SIGTERM becomes SystemExit**, but only if nobody else handles it. Otherwise a scheduler's
termination skips `atexit`, and the run loses its queued epochs and stays "running" forever.

**No default server.** Unset means disabled. A hard-coded address once sent a copied script's runs
to somebody else's Visin.

## The API contract

`tests/contract/test_openapi.py` runs every public entry point against a fake server and checks each
request against Visin's published OpenAPI spec. A request that fails its schema, hits a path that no
longer exists, or carries a field the server would silently strip, fails the test. CI runs it
against Visin's main branch weekly, so an API change that would break this package shows up without
anyone having to remember it.
