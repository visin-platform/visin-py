# Configuration

Every setting can come from the environment, which keeps tokens out of code, or be passed to
`visin.init` directly.

| Variable | `init` argument | Meaning |
| --- | --- | --- |
| `VISIN_URL` | `url=` | Your deployment's API address, such as `https://vision-api.example.com`. `…/api` works too. |
| `VISIN_TOKEN` | `token=` | A project token, or a user API key (`vsn_live_…`). |
| `VISIN_PROJECT` | `project=` | The project's id or slug. Needed with a user API key; a project token's runs always go to its own project. |
| `VISIN_TRAINING_UUID` | `training_uuid=` | Report into this run instead of creating one. |
| `VISIN_MODE` | `mode=` | `online` (the default), `offline` or `disabled`. |
| `VISIN_DIR` | `directory=` | Where reports wait when they cannot be sent. Default `~/.visin`. |
| `VISIN_VERIFY_SSL` | | `0` turns off TLS verification, for a self-signed development server only. |
| `VISIN_DATASET_URL` | `Datasets(url=)` | The dataset service's address, such as `https://dataset-api.example.com`. Datasets are served apart from runs. |
| `VISIN_DATA_DIR` | `Datasets(directory=)` | Where downloaded datasets go. Default `~/.cache/visin/datasets` (`$XDG_CACHE_HOME/visin/datasets`). |

`VISIN_API_URL`, `VISIN_API_TOKEN` and `VISIN_PROJECT_ID`, the names earlier versions used, are still
read.

There is no default server address. A script with none configured reports nothing, so it can never
send runs to somebody else's Visin by accident.

## Modes

| Mode | What happens |
| --- | --- |
| `online` | Reports are sent as they are made. When Visin stops answering, they wait on disk. |
| `offline` | Every report is written to disk, for `visin sync` to send later. See [offline](offline.md). |
| `disabled` | Nothing is sent. This is what `online` becomes when no URL or token is set. |

## Which run

`visin.init` decides like this:

1. If you give no `name` and `VISIN_TRAINING_UUID` is set, it reports into that run. This is for
   jobs that an orchestrator registered before launching them.
2. Otherwise it registers a run called `name`, or named after the script and the time when you give
   none. If `VISIN_TRAINING_UUID` or `training_uuid=` names a run that already exists, that run is
   reused rather than duplicated. A restarted job with a fixed UUID therefore continues the same run.

## Distributed training

Every process in a data-parallel job runs the same script. Only rank zero reports: the rank is read
from `RANK`, `SLURM_PROCID`, `OMPI_COMM_WORLD_RANK` or `PMI_RANK`, and every other rank gets a
disabled run.

## Logging

The package logs through the standard `logging` module, under the logger named `visin`. Warnings
cover anything that was not delivered. To see the rest on the console too, such as the run it created
and the summary at `finish`:

```python
visin.enable_console_logging()  # or enable_console_logging(stream=sys.stdout)
```

For a line per request as well:

```python
import logging

logging.getLogger("visin").setLevel(logging.DEBUG)
```
