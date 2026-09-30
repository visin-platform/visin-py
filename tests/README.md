# Tests

```sh
make test        # everything
make coverage    # everything, failing under the coverage floor in pyproject.toml
make contract    # only the API contract, which needs Visin's spec
pytest tests/internal/test_spool.py -k resume     # one file, or one test
```

No test touches the network, your environment or `~/.visin`. `conftest.py` clears every `VISIN_`
variable, points `VISIN_DIR` at a temporary directory, and stops runs from installing signal
handlers or `atexit` hooks in the test process. Requests go to the fakes in `fakes.py`, which record
what was sent and answer from a queue of canned responses.

Test names are sentences that say what the package promises, such as
`test_a_repeated_epoch_is_accepted_not_an_error`. A failure then reads as the promise it broke.

## Layout

The layout mirrors `src/visin/`: a test file sits where the module it tests sits.

```text
tests/
├── conftest.py           fixtures: the isolated environment, the fake client, the `server` fixture
├── fakes.py              FakeSession, FakeUploads, FakeResponse, ok() and refused()
│
├── test_run.py           visin.init and Run: what each log_* call sends, epoch identity,
│                         argument checking, creating and attaching, finishing, crashes and
│                         SIGTERM in a real subprocess, losing the server mid-run and catching up
├── test_offline.py       offline mode end to end: a run written to disk, then visin.sync()
├── test_api.py           Api and the models it returns: paging, finding runs, epochs, test results, DataFrames
├── test_system.py        system_info() and system_metrics(): nvidia-smi, PyTorch, psutil, /proc
├── test_cli.py           the visin command: check, sync, runs, version
│
├── internal/             visin._internal, one file per module
│   ├── test_config.py    environment variables, modes, overrides
│   ├── test_transport.py which failures are retried for which requests, errors, uploads
│   ├── test_inputs.py    epoch numbers, merging train/val, the shapes a config comes in
│   ├── test_process.py   rank variables, and the crash and SIGTERM hooks
│   ├── test_serialize.py NumPy, tensors, NaN and other values into JSON
│   ├── test_sender.py    the background thread: order, failures, a full queue, flush
│   ├── test_reports.py   delivering each kind of report, and repeats counted as delivered
│   └── test_spool.py     the file on disk, and syncing it: order, resuming, refusals, torn lines
│
├── integrations/
│   └── test_callbacks.py metric-name sorting, and the Keras and Lightning callbacks driven the
│                         way each framework calls them, against stand-in framework modules
│
├── contract/
│   └── test_openapi.py   every request the package sends, checked against Visin's OpenAPI spec
│
└── tooling/
    └── test_release.py   scripts/release.py: the version it picks and the changelog it writes
```

## The contract test

`contract/test_openapi.py` drives every public entry point once against the fake server, then
checks each recorded request against the spec Visin publishes
(`apps/frontend/landing-front/public/openapi/vision.json` in the Visin repository). It fails on:

- a method and path the API does not have;
- a body that does not satisfy the endpoint's schema;
- a body field the endpoint does not define, which the server would strip without telling anyone.

It looks for the spec at `VISIN_OPENAPI`, then at `../visin` next to this checkout, and skips when
there is neither. `make contract` sets `VISIN_REQUIRE_CONTRACT=1`, which turns that skip into a
failure. CI fetches the spec from Visin's main branch, on every change and weekly.

## Adding a test

- Put it in the file that mirrors the module you changed.
- Drive the public API where you can (`visin.init`, `Run`, `Api`, `visin.sync`, the CLI). Reach for
  `_internal` only for behaviour the public API cannot reach.
- Use the `server` fixture when the code under test builds its own client (`init`, `sync`, `Api`,
  the CLI), and the `client` fixture when you construct a `Run` yourself.
- Queue answers with `session.route(method, url_fragment, *responses)`. A response can also be an
  exception to raise, such as `TransportError("down")` for an unreachable server.
