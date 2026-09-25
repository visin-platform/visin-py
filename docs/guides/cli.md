# The visin command

Installing the package puts a `visin` command on your path, and `python -m visin` does the same.
Every command reads `VISIN_URL` and `VISIN_TOKEN`, and takes `--url` and `--token` to override them.

## visin check

Checks that this machine can report, before a long job finds out that it can't:

```text
$ visin check --write
visin 0.1.0
  url      https://vision-api.example.com
  token    3f9a…c21e (project token)
  project  road-seg
  mode     online
  kept in  /home/me/.visin

  ok    reached https://vision-api.example.com/api
  ok    the project token is accepted
  ok    project 'Road segmentation' is visible
  ok    created a test run (0b6f…)
  ok    sent an epoch to it
  ok    deleted the test run
```

Without `--write` it only reads. It exits with status 1 if any step fails, so you can put it at the
top of a job script.

## visin sync

Sends reports that were kept on disk, by an offline run or by a run that lost the server. See
[offline](offline.md).

| Option | |
| --- | --- |
| `--list` | Show what is waiting, and send nothing. |
| `--run UUID` | Only this run. |
| `--dir DIR` | Where the reports are kept, instead of `VISIN_DIR` or `~/.visin`. |
| `--drop-rejected` | Forget reports the server refuses, instead of keeping them to retry. |

It exits with status 1 if anything is still waiting afterwards.

## visin runs

```text
$ visin runs --project road-seg --status completed --limit 5
```

Lists recent runs: their UUID, status, last update and name.

## visin version

Prints the installed version.

Add `-v` before any command (`visin -v sync`) to log each request.
