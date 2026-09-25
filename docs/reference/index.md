# visin

Everything a script needs is importable from `visin` itself:

| Name | What it is |
| --- | --- |
| [`init`](run.md#visin.run.init) | Start reporting this script as a run. The one call most scripts need. |
| [`Run`](run.md#visin.run.Run) | A run: `log_epoch`, `log_test_results`, `log_benchmark`, `log_config`, `upload_visualization`, `update`, `finish`. |
| [`Api`](api.md#visin.api.Api) | Read projects, runs, epochs, test results and benchmarks. |
| [`sync`](offline.md#visin.offline.sync), [`pending`](offline.md#visin.offline.pending) | Send reports kept on disk, or list them. |
| [`system_info`](system.md#visin.system.system_info), [`system_metrics`](system.md#visin.system.system_metrics) | Describe the machine, and what a run is using. |
| [`epoch_uuid_for`](run.md#visin.run.epoch_uuid_for) | The UUID an epoch of a run always has. |
| [`flatten`](api.md#visin.api.flatten) | Nested results as `val.loss`-style keys. |
| [`enable_console_logging`](system.md#visin.enable_console_logging) | Print visin's log lines: the run, what failed to send, the summary. |
| [`VisinError`](errors.md) and subclasses | What `strict=True` raises. |

Modules whose names start with an underscore (`visin._internal`) are not part of the public API and
may change in any release.
