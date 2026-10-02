# The visin command

Installing the package puts a `visin` command on your path, and `python -m visin` does the same.
Every command reads `VISIN_URL` and `VISIN_TOKEN` (from the environment or the file `visin login`
saved), and takes `--url` and `--token` to override them. `visin check` shows which of these each
value came from.

## visin login

```text
$ visin login
Visin API address [https://vision-api.visin.eu]:
Token (a pipeline key or API key):
  ok    reached https://vision-api.visin.eu
  ok    the API key is accepted
saved to /home/me/.visin/config
```

Saves the address and token so no `export` lines are needed. It checks them first and saves nothing
if the check fails. The file is created readable only by you, and lines it does not set are kept.
For the hosted Visin it also saves the dataset address, so `visin download` works at once.

| Option | |
| --- | --- |
| `--url URL`, `--token TOKEN` | Give them instead of being asked. Outside a terminal, `--token` is required. |
| `--project P` | Also save `VISIN_PROJECT`. |
| `--no-check` | Save without contacting Visin, for a machine with no route to it. |

## visin logout

Deletes the saved file. Environment variables are not touched.

## visin check

Checks that this machine can report, before a long job finds out that it can't:

```text
$ visin check --write
visin <version>
  url      https://vision-api.visin.eu
  token    vsn_…c21e (API key)
  project  road-seg
  mode     online
  kept in  /home/me/.visin

  ok    reached https://vision-api.visin.eu/api
  ok    the API key is accepted
  ok    project 'Road segmentation' is visible
  ok    created a test run (0b6f…)
  ok    sent an epoch to it
  ok    deleted the test run
```

Without `--write` it only reads. It exits with status 1 if any step fails, so you can put it at the
top of a job script.

`--write` needs a project: pass `--project <id-or-slug>` or set `VISIN_PROJECT`, unless the token is
a pipeline key, which brings its own. The CLI cannot tell the two kinds of key apart from the token
text, so with no project it reports the missing one when the server refuses the test run.

## visin sync

Sends reports that were kept on disk, by an offline run or by a run that lost the server. See
[offline](offline.md).

| Option | |
| --- | --- |
| `--list` | Show what is waiting, and send nothing. With `--json`, print `{run uuid: reports}`. |
| `--run UUID` | Only this run. |
| `--dir DIR` | Where the reports are kept, instead of `VISIN_DIR` or `~/.visin`. |
| `--drop-rejected` | Forget reports the server refuses, instead of keeping them to retry. |

It exits with status 1 if anything is still waiting afterwards.

## visin runs

```text
$ visin runs --project road-seg --status completed --limit 5
```

Lists recent runs: their UUID, status, last update and name. `--json` prints them as JSON instead,
each with a `url` to its page in the web app when the app's address is known (see `VISIN_APP_URL`).

## visin datasets

```text
$ visin datasets --search zod
```

Lists the datasets on Visin: id, size and name. `--search` keeps those whose name matches, and
`--json` prints them as JSON. It reads
`VISIN_DATASET_URL` (or `--url`), which is a different address from `VISIN_URL`; see
[Datasets](datasets.md).

## visin download

```text
$ visin download zod
/home/me/.cache/visin/datasets/zod-65f1…/zod_dataset
```

Downloads a dataset by name or id, unpacks it, and prints the folder, so a job script can use
`root=$(visin download zod)`. Progress is logged to stderr.

| Option | |
| --- | --- |
| `--dir DIR` | Where datasets go, instead of `VISIN_DATA_DIR` or `~/.cache/visin/datasets`. |
| `--keep-archive` | Keep the ZIP after unpacking it. |
| `--no-unzip` | Download only, and print the ZIP's path. |
| `--url`, `--token` | Override `VISIN_DATASET_URL` and `VISIN_TOKEN`. A token is only needed for private datasets. |

## visin cache

```text
$ visin cache
dataset     12.4 GB  ZOD  /home/me/.cache/visin/datasets/zod-6aad…
partial      3.1 GB  9f2c-waymo.zip.part  /home/me/.cache/visin/datasets/9f2c-waymo.zip.part
15.5 GB in 2 items

$ visin cache rm zod
```

Shows what downloads left on disk, largest first, without contacting Visin. A `partial` is an
interrupted download or unpack, which the next `visin download` resumes or redoes. `rm` deletes a
dataset by name or id along with its ZIP and any partial download, and says how much it freed.
`--json` prints the list as JSON, and `--dir` names another data directory.

## visin version

Prints the installed version.

Add `-v` before any command (`visin -v sync`) to log each request.
