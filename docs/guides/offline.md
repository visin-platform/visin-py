# Offline and unreliable networks

Compute nodes often have no route to the outside at all, and a node that does can lose it for an
hour. Either way the run should keep training and its reports should arrive eventually.

## When the connection drops

This needs no setup. When an online run cannot reach Visin even after retrying, it:

1. writes each report to `~/.visin/runs/<run uuid>.jsonl`, in order;
2. tries the server again every minute, and once it answers, sends everything that was kept, in
   order, then carries on live;
3. tries once more at `finish`.

None of this holds up training. Registering the run at `init` gives up after two attempts, and each
later try is a single attempt, so even a server that drops packets rather than refusing costs
seconds, not minutes. Reports that `finish` has no time to send are written to disk too.

Anything still undelivered when the process exits is left on disk, and `finish` says so. Send it
with `visin sync`.

## Running fully offline

On a machine with no route to Visin at all:

```sh
export VISIN_MODE=offline
python train.py
```

The script is unchanged. Every report, including the run's creation and the files of its
visualizations, is written under `~/.visin`. Afterwards, from a machine that can reach Visin, such as
a cluster's login node, which usually shares your home directory:

```sh
export VISIN_URL=https://vision-api.example.com VISIN_TOKEN=…
visin sync --list      # what is waiting
visin sync             # send it
```

`VISIN_DIR` moves the directory, for instance onto a scratch filesystem shared between nodes.

## What sync guarantees

- **Order is kept.** A run's reports arrive in the order they were made.
- **It is safe to repeat.** An interrupted sync resumes where it stopped. Runs, epochs and test
  results carry their own ids, so a report that did arrive is recognised if it is sent again.
  Benchmarks and configs have no such id, so sync records each one as it is delivered and never
  sends it twice.
- **A running job can keep writing.** Sync first renames the file it sends, so reports written
  during a sync wait for the next one.
- **Refusals are kept.** A report the server refuses, such as one with a token outside its project,
  stays on disk. It is listed as refused, and can be sent again once the cause is fixed. `--drop-rejected`
  forgets such reports instead.

!!! warning "One sync at a time"
    Two `visin sync` commands over the same directory at the same time can deliver a benchmark or
    config twice. A sync running alongside a training job is fine.

## From Python

```python
import visin

visin.pending()  # {run uuid: reports waiting}
for result in visin.sync():
    print(result.training_uuid, result.sent, result.rejected, result.remaining)
```
