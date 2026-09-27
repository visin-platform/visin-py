# Changelog

All notable changes to this package are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until 1.0, a minor version may change
the public API; a patch version never does.

The release workflow writes each version's entry from the conventional commits since the previous
one, together with anything written by hand under Unreleased.

## [Unreleased]

## [0.2.1] - 2026-09-27

### Fixed

- drop legacy project token wording

[Unreleased]: https://github.com/visin-platform/visin-py/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/visin-platform/visin-py/compare/v0.2.0...v0.2.1
## [0.2.0] - 2026-09-26

### Added

- `visin.Datasets` and the `visin datasets` / `visin download` commands: list the datasets on Visin's
  dataset service (`VISIN_DATASET_URL`) and download one. A download resumes where an interrupted one
  stopped, is checked against the size Visin reports, is unpacked safely into `VISIN_DATA_DIR`
  (default `~/.cache/visin/datasets`), and is not repeated once complete.
- add datasets download to library

[0.2.0]: https://github.com/visin-platform/visin-py/compare/v0.1.0...v0.2.0
## [0.1.0] - 2026-09-26

### Added

- `visin.init()`, the one call most scripts need: registers a run, or attaches to the one
  `VISIN_TRAINING_UUID` names, and reports nothing when no server is configured.
- `Run.log_epoch`, `log_test_results`, `log_benchmark`, `log_config`, `upload_visualization` and
  `update`, sent from a background thread in the order they were made.
- `train=` and `val=` on `log_epoch`, for the two curves Visin's charts read.
- Safe retries: epochs and test results carry their own UUIDs, and a POST without one is repeated
  only when it provably never reached the server.
- Store and forward: reports wait on disk while Visin is unreachable and are sent when it answers.
  `VISIN_MODE=offline` keeps every report on disk for a machine with no route to Visin.
- `visin sync` and `visin.sync()`, which send kept reports, resume where they stopped, and never
  send a benchmark or config twice.
- `visin check`, which tests the connection and the token, and with `--write` creates and deletes
  a test run.
- `visin runs` and `visin.Api`, for reading projects, runs, epochs, test results and benchmarks,
  with `Api.epochs_frame` for a pandas DataFrame.
- `system_info()` for benchmarks and `system_metrics()` for the run's System tab, including GPUs
  through PyTorch or `nvidia-smi`.
- Keras and PyTorch Lightning callbacks.
- Conversion of NumPy and PyTorch scalars; NaN and infinity are sent as gaps.
- Only rank zero of a distributed job reports.
- A crashed or terminated run still sends its epochs and is marked failed.
- `Run.attach(..., mark_status=False)`, for a test or benchmark script that reports into a
  training's run without changing its status.
- `epoch_uuid=` on `log_test_results`, `log_benchmark` and `upload_visualization`, to name an epoch
  directly, such as the one a checkpoint's file name records.
- `Run.resumed`, and a resumed run is marked running again while it trains.
- A warning when an epoch logged again is kept at its first recorded values.
- `visin.enable_console_logging()`.

### Fixed

- github release flow

[0.1.0]: https://github.com/visin-platform/visin-py/releases/tag/v0.1.0
