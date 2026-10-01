# Datasets

Visin keeps datasets as zips, served by its dataset service at an address of its own. Point
`VISIN_DATASET_URL` at it, list what is there, and download one:

```bash
export VISIN_DATASET_URL=https://dataset-api.example.com
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

A dataset is named by its name (any case) or its id.

## What a download does

1. Asks the dataset service for a signed link to the zip. Public datasets need no token; with
   `VISIN_TOKEN` set, the private ones it may read are listed and downloadable too.
2. Downloads it to `<data dir>/<id>-<file>.zip.part`, resuming a part left by an interrupted download
   (with a fresh link, as links expire) and retrying a connection that drops.
3. Checks the size against what Visin reports. A short download stays a `.part` to resume from; it is
   never unpacked.
4. Unpacks it into `<data dir>/<name>-<id>/`, refusing any entry that would land outside that folder,
   marks the folder complete and removes the zip.

The next `download` of the same dataset finds the complete folder and returns at once. The folder
returned is the zip's single top-level folder when it has one (`zod_dataset/`), else the unpacked
folder itself.

The data directory is `VISIN_DATA_DIR`, else `~/.cache/visin/datasets`. On a cluster, point it at
scratch space: home quotas rarely fit a dataset.

In a terminal, downloads show a single-line progress bar with percentage,
downloaded/total size, transfer speed and estimated time remaining. Resumed
downloads start at the bytes already on disk. Updates are throttled to five per
second; redirected output retains periodic log lines. Extraction is announced
after the download finishes. Pass `quiet=True` to `Datasets.download()` to hide
download progress.
