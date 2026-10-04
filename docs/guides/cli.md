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
The server tells it where its dataset service and web app are (`GET /api/.well-known/visin`), so `visin download` and run links work at once with no further exports. An address you already set is kept. It also says which kind of key you gave it, such as `pipeline key for project 'Road' (vision:read, vision:write)`.

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

## visin suites

Publishes, lists and shows scoring protocols. See [evaluating models](evaluation.md).

```text
$ visin suites push suites/road-test.json --project road-seg --public
$ visin suites list --all
$ visin suites show road-test@1
$ visin suites digest suites/road-test.json
```

`digest FILE` prints the digest Visin gives a suite or protocol file without publishing it, which is what an evaluation
sends as the protocol it ran. `manifest day=FILE night=FILE` prints the digest and sample counts of split files, for a suite's `data`; it needs no
server. `push` reads JSON, or YAML with `pip install 'visin[yaml]'`. The project comes from `--project`, then the file, then
`VISIN_PROJECT`, then the project a pipeline key is limited to. `--public` lets anyone read the protocol; `--private`
(the default for a new suite) keeps it to the project's readers.

## visin evaluate

Records a checkpoint's results on a suite, from a JSON file of results.

```text
$ visin evaluate results.json --suite road-test@1 --checkpoint best.pth --sample-count day=1200 --sample-count night=800
```

| Option | |
| --- | --- |
| `--suite SLUG@VERSION` | The suite version. Required. |
| `--checkpoint FILE`, `--label NAME` | The weights file that was loaded. Only its SHA-256 and the label are sent. |
| `--hub-repo`, `--hub-commit`, `--hub-path` | A Hub checkpoint instead: repo, the full commit, and a file inside it. |
| `--sample-count NAME=N`, `--sample-counts FILE` | Samples each condition scored. |
| `--project`, `--run`, `--epoch` | The project, and the run and epoch the checkpoint came from. |
| `--evaluator-package`, `--evaluator-version`, `--evaluator-commit` | What produced the numbers. Package and version are also sent as evidence (optional). |
| `--data KIND=VALUE` | The data you read: `external=<manifest sha256>`, `visin=<archive sha256>` or `hf=org/name@<commit>`. |
| `--protocol FILE`, `--protocol-digest SHA` | The suite or protocol file you ran (its digest is asked of Visin, at `visin sync` if offline), or the digest itself. |
| `--classes-scored A,B`, `--classes-ignored C` | The classes you scored and left out. |
| `--uuid`, `--supersedes ID` | Your id, to make a repeat harmless; the evaluation this one corrects. |
| `--dry-run` | Judge it and store nothing. |
| `--require-ranked` | Exit 3 when it is not ranked, for CI. |
| `--json` | Print the outcome as JSON. |

It exits 0 when recorded or checked, 1 when refused or unusable, and 3 with `--require-ranked` for a result that is
not ranked.

## visin leaderboard

```text
$ visin leaderboard road-test@1
$ visin leaderboard road-test@1 --public
$ visin leaderboard road-test@1 --page 2 --limit 50
$ visin leaderboard road-test@1 --all
```

The ranking of a suite version, with where each model is weakest and what could not be ranked. `--public` reads the
anonymous ranking of published results. `--json` prints it as JSON.

The server sends at most 100 checkpoints per page. Without a flag you get the first page, and a line under the table
says which page that is and how many checkpoints there are in all. Ranks are global, so page 2 starts at the rank the
whole pool gives it, never at 1.

| Option | Meaning |
| --- | --- |
| `--observed` | Rank only results whose evaluator sent complete evidence. |
| `--page N`, `--limit N` | One page of ranked checkpoints (`--limit` at most 100). |
| `--unranked-page N` | One page of the unranked list, which pages apart from the ranking. Not for `--public`. |
| `--all` | Read every page of both lists. Cannot be combined with `--page` or `--unranked-page`. |

## visin diff

Fails a CI job when one evaluation scores worse than another on the same suite version.

```text
$ visin diff BASELINE CANDIDATE --max-drop 0.01
road-test@1: 6ab5932a0e9a6b7570e30e30 against 6ab5932a0e9a6b7570e30e2e (allowed drop 0.01)
  improved   overall               mIoU_foreground  0.7 -> 0.72  +0.02
  ok         day                   mIoU_foreground  0.8 -> 0.795  -0.005
  regressed  night                 mIoU_foreground  0.6 -> 0.55  -0.05
  FAILED
```

Both are evaluation ids (from `visin evaluations`), or uuids with `--project`. The headline metric is compared for the
overall figure and for every condition; `--all-metrics` compares every metric the suite names. Each score is read in
the suite's direction, so for a latency a rise is the regression. A score may fall by `--max-drop` (in its own units,
default 0) before it counts. Only ranked results are compared, and only on one suite version: a different version or
protocol is refused, since it is a different measurement.

| Exit status | Meaning |
| --- | --- |
| 0 | Nothing fell by more than `--max-drop`. |
| 1 | Refused or unusable: an unknown evaluation, two suite versions, a bad argument. |
| 4 | A score regressed past `--max-drop`. |
| 5 | A score could not be compared: a condition one side lacks, or an evaluation that is not ranked. Takes precedence over 4, since the comparison is then incomplete. |

`--json` prints every compared score for scripts.

## visin evaluations

```text
$ visin evaluations --project road-seg --suite road-test@1 --state incomplete --limit 10
```

Lists evaluations with their verdicts, and marks those on the public leaderboard.

## visin publish, withdraw and promote

```text
$ visin publish 6ab5932a0e9a6b7570e30e2e
$ visin withdraw 6ab5932a0e9a6b7570e30e2e
$ visin promote 6ab5932a0e9a6b7570e30e2e --suite road-test@1 --checkpoint best.pth --sample-count day=1200
```

`publish` puts a ranked evaluation on its suite's public leaderboard and `withdraw` takes it off. `promote` copies a
result recorded without a suite (a test a run reported) onto a suite so it can be ranked; it takes the same checkpoint and sample-count options as
`evaluate`. Both need manage access to the project.

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

## visin push

```text
$ visin push zod --repo acme/zod-png
acme/zod-png@3f2a1c9d8e7b6a5f4e3d2c1b0a99887766554433
```

Publishes a dataset kept on Visin to the Hugging Face Hub with your own Hub token, then points Visin at
the repo and the commit. See [Datasets](datasets.md#publishing-a-dataset-to-the-hub). Needs `visin[hf]`
and the right to manage the dataset.

| Option | |
| --- | --- |
| `--repo ORG/NAME` | The Hub dataset repo. Required. Created when it does not exist. |
| `--public` | Make a newly created repo public. The default is private. |
| `--dir DIR` | Where datasets go, as for `download`. |
| `--url`, `--token` | Override `VISIN_DATASET_URL` and `VISIN_TOKEN`. |

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
