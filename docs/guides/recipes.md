# Recipes

Short setups for the places training jobs run. Each assumes `pip install visin` in the job's
environment.

## A SLURM cluster, with compute nodes that have no internet

Report offline from the compute node, and send from the login node, which usually shares your home
directory.

```sh
#!/bin/bash
#SBATCH --gres=gpu:1
export VISIN_MODE=offline
export VISIN_DIR=$HOME/.visin            # on the filesystem the login node shares
python train.py
```

Then, from the login node, with `visin login` done once:

```sh
visin sync --list    # what is waiting
visin sync
```

Run `visin sync` while the job is still training and you get live-ish charts: reports written so
far are sent, and later ones wait for the next sync. See [Offline](offline.md).

If compute nodes can reach Visin, drop `VISIN_MODE` and put `visin check` at the top of the script.
It exits with status 1 when the job could not report, so a bad token fails in a second, not after
a day of training:

```sh
visin check || exit 1
python train.py
```

## A job that SLURM requeues

A requeued job keeps its `SLURM_JOB_ID`. Derive the run's UUID from it, and the restarted job
continues the same run rather than starting a second:

```sh
export VISIN_TRAINING_UUID=$(python -c "import uuid, os; print(uuid.uuid5(uuid.NAMESPACE_DNS, 'road-seg-' + os.environ['SLURM_JOB_ID']))")
python train.py     # visin.init("unet baseline") resumes it if it exists
```

`run.resumed` says which happened. Epochs the restarted job logs again keep their first values, so
resume from the epoch of your checkpoint and number the epochs the same way. See
[Resuming](reporting.md#resuming).

## Training and testing as separate scripts

The training script creates the run. Later scripts, perhaps on other machines, add to it without
touching its status:

```python
# test.py
import visin

with visin.Run.attach(training_uuid, mark_status=False) as run:
    run.log_test_results(epoch, scores)
```

An orchestrator that registered the run first can export `VISIN_TRAINING_UUID` instead of passing
the UUID. A script can also look a run up by name, with
[`Api().find(name)`](reading.md#finding-a-run).

## Docker and Compose

Pass the settings as environment variables, and keep the token out of the image:

```yaml
services:
  train:
    image: my-training-image
    environment:
      VISIN_URL: https://vision-api.visin.eu
      VISIN_TOKEN: ${VISIN_TOKEN}
      VISIN_DIR: /spool
    volumes:
      - visin-spool:/spool
volumes:
  visin-spool:
```

Without the volume, reports that could not be sent die with the container. Alternatively mount a
file and set `VISIN_ENV_FILE=/run/secrets/visin.env`, with the same `NAME=value` lines.

## Continuous integration and tests

Leave `VISIN_URL` and `VISIN_TOKEN` unset: every call does nothing, so the same code runs in CI
unchanged, and nobody needs a secret to run the tests. To check your own code's calls without a
server, use `visin.Run.disabled()`. To make a failure to report an error in a test, pass
`strict=True`:

```python
with visin.init("test", strict=True) as run:
    run.log_epoch(1)  # raises: it needs results, where a normal run logs a warning
```

## A notebook

Use a short, explicit setup, and the console log, since a notebook has no terminal to show
warnings:

```python
import visin

visin.enable_console_logging()
run = visin.init("exploration", project="road-seg", tags=["notebook"])
print(run.url)
...
run.finish()
```

A notebook that is closed without `finish` is finished when the kernel exits, so the run is not
left "running" for ever.

## Several GPUs

Every process runs the same script, and only rank zero reports (read from `RANK`, `SLURM_PROCID`,
`OMPI_COMM_WORLD_RANK` or `PMI_RANK`). Log the metrics you have already reduced across ranks, since
rank zero's own numbers are what Visin shows.
