# Getting started

You need an account on [Visin](https://app.visin.eu) (or your own deployment).

## 1. Install

```sh
pip install visin
```

Optional extras:

| Extra | Adds |
| --- | --- |
| `visin[system]` | `psutil`, for more accurate CPU and memory figures in benchmarks and the System tab |
| `visin[pandas]` | `pandas`, for `Api.epochs_frame` |
| `visin[hf]` | `huggingface_hub`, for Hub datasets in `Datasets.download` and `Run.log_model` |

## 2. Get a token

A run belongs to a project. In Visin, open the project, go to **Settings → Pipeline keys** and
choose **New pipeline key**. Copy the two lines it shows.

See [Tokens](guides/tokens.md) for which key to use where.

A pipeline key can write only to its own project. An API key that is not limited to a project
(`vsn_live_…`, under **Account → API keys**) also works, and is the one to use for reading runs
across projects.

## 3. Tell the script where Visin is

The simplest way is to save it once on this machine:

```sh
visin login        # asks for the address and the token, checks them, saves them
```

`visin login` keeps them in `~/.visin/config` (readable only by you), and every script and
command on the machine picks them up. `visin logout` deletes the file. Use
[`VISIN_PROJECT`](guides/configuration.md) or `--project` to set the project too.

Or, for a job or a container, use environment variables, which win over the saved file:

```sh
export VISIN_URL=https://vision-api.visin.eu   # the hosted Visin; use your own address if self-hosted
export VISIN_TOKEN=vsn_live_…
export VISIN_PROJECT=road-seg                      # id or slug; optional with a pipeline key
```

Check the setup before your first real run:

```sh
visin check --write
```

`check` reaches the server, checks that the token is accepted, and with `--write` creates a test run,
sends it an epoch, then deletes it. Each step prints `ok` or explains what is wrong.

## 4. Send a run

```python
import math
import random

import visin

with visin.init("quickstart") as run:
    for epoch in range(1, 21):
        train_loss = 1.2 * math.exp(-epoch / 6) + 0.1 + random.uniform(0, 0.02)
        val_miou = 0.3 + 0.35 * (1 - math.exp(-epoch / 5))
        run.log_epoch(
            epoch,
            train={"loss": train_loss},
            val={"loss": train_loss + 0.05, "mean_iou": val_miou},
            learning_rate=1e-3,
        )
```

Open the project's **Trainings** tab and choose **quickstart**. You will see a loss chart with a
training and a validation curve, and a mean IoU chart.

Leaving the `with` block finishes the run. If the block raises, the run is marked failed, and every
epoch sent before that point is kept.

## 5. Drop it into your own loop

Replace the made-up numbers with what your training step returns. Two habits keep the charts right:

- **Put training metrics in `train` and validation metrics in `val`.** These are the two curves
  each chart draws.
- **Leave out what an epoch did not measure.** If you validate every fifth epoch, send `val` on
  those epochs only. A missing metric shows as a gap, where a `0` would show as a real drop.

Without `VISIN_URL` and `VISIN_TOKEN` the same script runs as before and reports nothing. That is
what lets a colleague run it without an account.

## 6. Download a dataset

Datasets come from their own service, so they need a second address. Public datasets download
without a token; `VISIN_TOKEN` is sent when set, for private ones.

```sh
export VISIN_DATASET_URL=https://dataset-api.visin.eu
visin datasets          # list what is available
visin download zod      # prints the folder it unpacked to
```

Without `VISIN_DATASET_URL`, `visin datasets` fails because the package has no default address.
See [Datasets](guides/datasets.md) for the options.

## If something goes wrong

| You see | Cause |
| --- | --- |
| The script runs but nothing appears in Visin | `VISIN_URL` or `VISIN_TOKEN` is unset, so reporting is disabled. Run `visin check`, or `visin login`. With a token but no URL, the script logs a warning saying so. |
| `visin check` says the key is not accepted | The token is wrong, or belongs to another deployment. Copy it again from **Settings → Pipeline keys**. |
| `visin check --write` says a project is missing | An API key that is not limited to a project needs `VISIN_PROJECT`. A pipeline key does not. |
| `visin datasets` fails at once | `VISIN_DATASET_URL` is unset. Datasets use their own address, shown in step 6. |

Add `-v` (`visin -v check`) to log each request. [Troubleshooting](guides/troubleshooting.md) lists every warning visin logs.

## Next

- [What a run records](guides/reporting.md): test results, benchmarks, frames and configs.
- [Keras, Lightning and Hugging Face](guides/frameworks.md): a callback instead of `log_epoch`.
- [Tokens](guides/tokens.md): which key to use, and how to keep it safe.
- [Configuration](guides/configuration.md): every environment variable.
- [Recipes](guides/recipes.md): SLURM, requeued jobs, Docker, CI and notebooks.
- [Offline and unreliable networks](guides/offline.md): compute nodes with no route to Visin.
