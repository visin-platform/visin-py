# Configuration

Every setting can come from the environment, which keeps tokens out of code, or be passed to
`visin.init` directly.

| Variable | `init` argument | Meaning |
| --- | --- | --- |
| `VISIN_URL` | `url=` | Your deployment's API address, such as `https://vision-api.visin.eu`. `…/api` works too. |
| `VISIN_TOKEN` | `token=` | An API key (`vsn_live_…`): a pipeline key from the project's settings, or one from **Account → API keys**. |
| `VISIN_PROJECT` | `project=` | The project's id or slug. Needed with a key that is not limited to a project; a pipeline key's runs always go to its own project. |
| `VISIN_TRAINING_UUID` | `training_uuid=` | Report into this run instead of creating one. |
| `VISIN_MODE` | `mode=` | `online` (the default), `offline` or `disabled`. |
| `VISIN_PROVENANCE` | `init(provenance=)` | `0` stops a run recording what it was started from: the git commit and branch (and whether the tree was dirty), the command line with credentials redacted, the installed packages and the machine. On by default. |
| `VISIN_DIR` | `directory=` | Where reports wait when they cannot be sent. Default `~/.visin`. |
| `VISIN_VERIFY_SSL` | | `0` turns off TLS verification, for a self-signed development server only. |
| `VISIN_DATASET_URL` | `Datasets(url=)` | The dataset service's address, such as `https://dataset-api.visin.eu`. Datasets are served apart from runs, so downloading needs this as well as `VISIN_URL`. |
| `VISIN_APP_URL` | | The web app's address, so `run.url` and `visin runs --json` can link to a run. Known for the hosted Visin (`https://app.visin.eu`); set it for your own deployment. |
| `VISIN_DATA_DIR` | `Datasets(directory=)` | Where downloaded datasets go. Default `~/.cache/visin/datasets` (`$XDG_CACHE_HOME/visin/datasets`). |

## Where settings come from

A setting is looked up in this order, and the first one found wins:

1. an argument (`init(url=...)`, `--url`);
2. an environment variable;
3. the file named by `VISIN_ENV_FILE`, which must exist;
4. the config file: `$VISIN_CONFIG`, else `~/.visin/config`. `visin login` writes it. It stays in
   `~/.visin` even when `VISIN_DIR` moves the reports elsewhere.

Both files hold `NAME=value` lines using the variable names above, with `#` comments and optional
`export` and quotes. Only `VISIN_` names are read from them. The package does not read a `.env` in
the working directory: whichever file sits beside a script would otherwise decide where its runs go.
The config file holds a token, so keep it private; a warning is logged when other users can read it.

```sh
# job.env, for one job:   VISIN_ENV_FILE=job.env python train.py
VISIN_URL=https://vision-api.visin.eu
VISIN_TOKEN=vsn_live_…
```

There is no default server address. A script with none configured reports nothing, so it can never
send runs to somebody else's Visin by accident. If it has a token but no address, or the reverse, it
logs a warning saying which is missing.

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

Active online runs send a heartbeat every 30 seconds. Projects mark silent jobs `stalled` after their
configured timeout (30 minutes by default); a new heartbeat or epoch resumes them.
Heartbeats keep checking the connection after an outage and retry queued reports even during a long epoch. Offline and
post-training reporting (`mark_status=False`) send no heartbeat. Configs, benchmarks and
visualizations carry client IDs so retries and spool replay keep one record.

Dataset reads retry anonymously only when an older server specifically refuses a project-limited
key. Current servers allow pipeline keys to read public datasets and their project's owner's datasets.
