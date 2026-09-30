# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code Style

- AVOID inline comments in new code. Reasoning a reader needs goes in the docstring of the function or module, or in `docs/architecture.md`. Existing comments stay until the code around them is touched.
- Don't edit `CHANGELOG.md` release sections or `_version.py` by hand. The Release workflow builds the changelog and picks the version from Conventional Commit messages. Only add a line under `## [Unreleased]` for something a commit subject cannot say.
- Support Python 3.9. Use `from __future__ import annotations`; never `isinstance(x, int | str)` at runtime.
- `requests` is the only runtime dependency. Anything else goes behind an extra in `pyproject.toml`.
- Ruff runs a wide rule set (see `pyproject.toml`): every public module, class, function and method needs a docstring, `print` is allowed only in the CLI, and a lazy import needs a per-file entry. Fix a finding; widen an ignore only with a reason.
- Public names are the ones exported from `src/visin/__init__.py`. Anything under `visin._internal` may change freely.

## Commands

### Development Setup
```bash
make install          # .venv with every dev tool, plus the pre-commit hooks
```

### Running Tests
```bash
make test                                   # everything
pytest tests/test_run.py -k resume          # one file, or one test
make coverage                               # fails under the floor in pyproject.toml
make contract                               # every request checked against Visin's OpenAPI spec
```

### Formatting, Linting and Types
```bash
make format           # ruff format + ruff check --fix
make check            # lint, mypy --strict, coverage, build, docs: what CI runs
```

Run `make check` before a commit. Never lower the coverage floor to make a change pass.

### Running the Application
```bash
visin check --write   # confirm this machine can report (creates and deletes a test run)
visin runs            # list runs
visin sync            # send reports kept on disk
visin datasets        # list datasets
```

## Architecture Overview

`visin` is a small client for the Visin platform: it reports training runs from a script and reads them back. The layers are kept apart.

### Core Flow
1. **User API** (`run.py`, `__init__.py`) - `init()`, `Run.attach()`, and the `log_*` methods. `Run` turns each call into an *op*.
2. **Delivery** (`_internal/reports.py`) - one `deliver` function sends any op. Live sends, the spool and `visin sync` all use it.
3. **Sending** (`_internal/sender.py`) - a daemon thread drains a queue, so `log_epoch` costs the training loop nothing.
4. **Persistence** (`_internal/spool.py`, `offline.py`) - ops kept on disk as JSONL when Visin is unreachable or `VISIN_MODE=offline`.
5. **Transport** (`_internal/transport.py`) - `HttpClient`: retries, signed uploads, resumable downloads.
6. **Reading** (`api.py`, `datasets.py`, `models.py`) - `Api` and `Datasets` return typed models.
7. **Frameworks** (`integrations/`) - Lightning and Keras callbacks.

### Key Design Decisions

**Nothing raises into the training loop.** A metrics backend is not worth a training job. Failures are logged and counted, and `finish` summarises them. `strict=True` is for tests. The one exception is starting a run: a refusal (bad token, wrong project) raises, because every later report would fail the same way.

**Everything written is an op.** An op is a JSON dict. Sent now, written to disk, or sent from disk a week later, it goes through the same code path.

**Retries are safe by construction.** Runs, epochs and test results carry caller-generated UUIDs, and a 409 counts as delivered. `epoch_uuid_for(run_uuid, epoch)` is deterministic, so any process can name an epoch. Benchmarks and configs carry no id, so they are the only ops a lost answer can duplicate.

**Create and attach are separate.** `init` always registers a run, resuming it when its UUID is known. `Run.attach` reports into one that exists, and `mark_status=False` leaves its status alone.

**No default server address.** Unset means disabled, never somebody else's Visin.

**The server's wire format stays in `models.py`.** Scripts read `run.project_id`; `_id` and `projectId` do not leak out of it.

### Testing Strategy

The tests mirror the package (see `tests/README.md`). No test touches the network, the environment or `~/.visin`: `conftest.py` clears every `VISIN_` variable and isolates the spool. Requests go to the fakes in `tests/fakes.py`. Test names are sentences that state what the package promises. `tests/contract/` checks every request the package sends against the server's OpenAPI spec.

### Important Files for Common Tasks

- **Adding a `log_*` method**: `run.py` (the method and its op), `_internal/reports.py` (how the op is delivered), then the contract test.
- **A new field the server returns**: `models.py`, then `Api` in `api.py` if it needs a new call.
- **A new endpoint**: check the spec first (below). Update `tests/contract/test_openapi.py` so it is exercised.
- **Config and environment variables**: `_internal/config.py`, then `docs/guides/configuration.md`.
- **CLI**: `cli.py`, and the entry point in `pyproject.toml`.
- **A new framework integration**: `integrations/`, sharing `_metrics.py`. Keep the framework import at module top, and test with fakes.

## Source of Truth Elsewhere

- **The server**: `/mnt/ml/projects/visin`. The spec is `apps/backend/vision-service/docs/openapi.yml`, and the dataset API is in `apps/backend/dataset-service`. A request field that is not in the spec is stripped by the server, so check it before adding one.
- **A real consumer**: `/mnt/ml/projects/visin-fusion` (`visin_fusion/integrations/visin.py`). Look there before changing a public signature.

## Making a Change

1. Read the code you are changing, and the test that covers it.
2. Write the test first when fixing a bug, at the boundary that failed. Prefer the public API over internals.
3. Keep the patch scoped to the change. Don't fold in unrelated cleanup.
4. A user-facing change updates the page under `docs/`. Docstrings feed the API reference.
5. `make check`, then commit with a Conventional Commit message: `feat:` (minor), `fix:` (patch), `feat!:` (breaking), or `docs:`, `test:`, `ci:`, `chore:`, `refactor:`, `build:`, `style:` (not listed in the changelog).
6. Commit or push only when asked.
