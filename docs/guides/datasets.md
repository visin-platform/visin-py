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

## Datasets on Hugging Face

A dataset can live on the Hugging Face Hub instead of, or as well as, a ZIP on Visin. Visin then
stores only a pointer: the repo and one commit, so a run from last year still names the same data.
`download` fetches it from the Hub, at that commit, into the same folder layout and with the same
completeness marker, so a config that says `"dataset_root": "visin:zod"` keeps working.

```bash
pip install 'visin[hf]'
export HF_TOKEN=hf_...          # only for a private or gated repo
visin download zod
```

- The Hub is preferred when the dataset has a ZIP too. If the Hub cannot be reached, or the extra is
  not installed, the ZIP is used instead and the folder records that version.
- A dataset on the Hub alone has no ZIP: `--no-unzip` and `--keep-archive` refuse it.
- Pointing the dataset at a new commit on its Visin page is a new version: the next `download` fetches it.
- Your token is read by `huggingface_hub` itself. This package never sends it to Visin.

### Publishing a dataset to the Hub

```bash
visin push zod --repo acme/zod-png            # a new repo is private; add --public to open it
```

```python
commit = datasets.push("zod", "acme/zod-png")
```

Downloads the dataset if it is not here yet, uploads its folder to the Hub dataset repo (created when it
does not exist) with your own Hub token, then tells Visin the repo and the commit that upload made. From
then on `download` fetches it from the Hub, and the ZIP on Visin stays as the fallback. You need to be
able to manage the dataset on Visin, and `pip install 'visin[hf]'`.

Publish only data whose licence allows redistribution, and never images of people without consent: a
public Hub repo is public. If the upload works but Visin refuses the link, the message names the commit
so you can set it on the dataset's page.

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

The dataset marker records the revision and size returned with the signed download URL. If the
archive is replaced after the metadata lookup, the cache and marker use the archive actually
downloaded. Servers that omit these fields continue to use the dataset metadata.

Invalid cache markers trigger a fresh download and extraction. Archive filenames are reduced to
a basename before saving, so an uploaded filename cannot place the ZIP outside the cache.
