# Datasets

Visin keeps datasets as ZIP files, served by a dataset service at an address of its own. The hosted
one is `https://dataset-api.visin.eu`. Point `VISIN_DATASET_URL` at it, list what is there, and
download one. Public datasets need no account or token.

```bash
export VISIN_DATASET_URL=https://dataset-api.visin.eu
visin datasets                 # id, size and name of each
visin download zod             # prints the folder it unpacked to
```

or from Python:

```python
from visin import Datasets

with Datasets() as datasets:
    for dataset in datasets.list():
        print(dataset.name, dataset.size)
    root = datasets.download("zod")  # a pathlib.Path
```

A dataset is named by its name (any case) or its id. For private datasets, also set `VISIN_TOKEN`.
Without `VISIN_DATASET_URL` there is no server to ask, so the command and `Datasets()` fail: the
package has no default address.

## Where it goes

The data directory is `VISIN_DATA_DIR`, else `~/.cache/visin/datasets`. On a cluster, point it at
scratch space: home quotas rarely fit a dataset.

The folder returned is the ZIP's single top-level folder when it has one (`zod_dataset/`), else the
unpacked folder itself. The next `download` of the same dataset finds the complete folder and
returns at once.

## What a download does

1. Asks the dataset service for a signed link to the ZIP.
2. Downloads it to `<data dir>/<id>-<file>.zip.part`, resuming a part left by an interrupted download
   (with a fresh link, as links expire) and retrying a connection that drops.
3. Checks the size against what Visin reports. A short download stays a `.part` to resume from; it is
   never unpacked.
4. Unpacks it into `<data dir>/<name>-<id>/`, refusing any entry that would land outside that folder,
   marks the folder complete and removes the ZIP.

## Seeing and freeing the space

```bash
visin cache            # what is on disk, and how large
visin cache rm zod     # delete a dataset, its ZIP and any partial download
```

From Python, `visin.cached_datasets()` and `visin.remove_cached("zod")` do the same, with no server.

## Progress

In a terminal, a download shows one line with percentage, size, speed and time remaining. A resumed
download starts at the bytes already on disk. Redirected output gets periodic log lines instead.
Pass `quiet=True` to `Datasets.download()` to hide progress.

## Keeping or skipping the ZIP

```bash
visin download zod                  # unpack, then delete the ZIP (the default)
visin download zod --keep-archive   # unpack, and keep the ZIP
visin download zod --no-unzip       # download only; print the ZIP's path
```

The Python options are keyword-only:

```python
with Datasets() as datasets:
    folder = datasets.download("zod", keep_archive=True)
    archive = datasets.download("waymo", unzip=False)
```

- With `unzip=False` the returned path is the ZIP, which is always kept, so `keep_archive` has no
  effect.
- A complete ZIP whose size matches is reused, including when retrying a failed unpack.
- `keep_archive=True` on an already unpacked dataset fetches the missing ZIP without unpacking again.
- The ZIP is deleted only after a successful unpack. If deleting it fails, a warning is logged and
  the dataset folder is still returned.
