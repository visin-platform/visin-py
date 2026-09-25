# Getting started

You need a Visin you can sign in to and Python 3.9 or newer.

## 1. Install

```sh
pip install visin
```

Optional extras:

| Extra | Adds |
| --- | --- |
| `visin[system]` | `psutil`, for more accurate CPU and memory figures in benchmarks and the System tab |
| `visin[pandas]` | `pandas`, for `Api.epochs_frame` |

## 2. Get a token

A run belongs to a project. In Visin, open the project, go to **Settings → API Tokens** and choose
**Generate New Token**. Copy it now, because it is shown only once.

A project token can write only to its own project. A user API key (`vsn_live_…`, under **Account →
API keys**) also works, and is the one to use for reading runs across projects.

## 3. Tell the script where Visin is

```sh
export VISIN_URL=https://vision-api.example.com   # your deployment's API address
export VISIN_TOKEN=paste-your-token-here
export VISIN_PROJECT=road-seg                      # id or slug; optional with a project token
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
