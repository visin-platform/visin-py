# Troubleshooting

Start with `visin check --write`. It tests the address, the token, the project and a real write,
and says which step fails. Then turn the log on, because most of what visin has to say is a
warning:

```python
import visin

visin.enable_console_logging()
```

Warnings are always shown by Python's logging. The lines marked *info* below only appear with the
console log on. Every message starts with `visin:`.

## Nothing is reported

| Message | What it means and what to do |
| --- | --- |
| *info* `VISIN_URL/VISIN_TOKEN unset, so reporting is disabled` | Neither is set. This is the normal laptop case. To report, run `visin login`, or export both. |
| `a token is set but VISIN_URL is not` | Set `VISIN_URL` (the hosted one is `https://vision-api.visin.eu`), or run `visin login`. |
| `VISIN_URL is set but VISIN_TOKEN is not` | Create a pipeline key in the project's **Settings → Pipeline keys**. |
| *info* `VISIN_MODE=disabled, so reporting is off` | Someone set it. Unset it, or use `offline` to keep reports for `visin sync`. |
| *debug* `rank N of a distributed job; only rank 0 reports` | Expected. Only rank zero of a distributed job reports. |

## Starting a run fails

`visin.init` raises when Visin refuses the run, because every later report would be refused the same
way.

| Error | Cause |
| --- | --- |
| `401` or "Invalid token" | The token is wrong, expired, or from another deployment. |
| `403` "Access denied to project" | The key cannot write to that project. A pipeline key writes only to its own. |
| `400` "A training needs a project" | An API key that is not limited to a project needs `project=` or `VISIN_PROJECT`. |
| `a run needs a name` | An empty `name`. |
| `VISIN_MODE='…'; expected one of` | `VISIN_MODE` is not `online`, `offline` or `disabled`. |

Being unreachable does not raise. See the next section.

## Visin cannot be reached

| Message | What it means |
| --- | --- |
| `could not reach Visin to create run …; reports are kept in … and will be sent when it answers` | The run could not be registered, and the script carries on. Reports wait on disk. |
| `lost Visin (…); keeping reports in … until it answers` | The connection dropped mid-run. Reports wait on disk and are retried every minute. |
| *info* `Visin answers again; sent N kept reports` | It recovered. |
| `run … finished: N reports sent, … waiting on disk; send them with visin sync` | Some reports are still on disk. Run `visin sync`. |
| `run … finished offline; N reports kept in …` | An offline run, as intended. Send them from a machine that can reach Visin. |

On a machine that never has a route, use `VISIN_MODE=offline`. See
[Offline and unreliable networks](offline.md).

## Reports that were not delivered

| Message | Cause |
| --- | --- |
| `report failed: …` | Visin refused one report, such as a body it does not accept. The training goes on. |
| `Visin refused a kept report: …` | A report kept on disk was refused when sent again. `visin sync --list` shows it. |
| `queue full, dropped a report` | More than 10,000 reports were waiting, because the network is much slower than training. The drop is counted in the summary. |
| `flush timed out after Ns; some reports may be unsent` | `finish` ran out of time. What it could not send is kept on disk for `visin sync`. |
| `could not keep a report on disk: …` | The directory (`VISIN_DIR`, default `~/.visin`) is full or read-only. |
| `run … has finished; nothing more is sent` | A `log_*` call after `finish`. Move the call before it. |
| `nothing more will be reported for run …` | The run was refused at the start and was switched off. |

## Data that was changed on the way

| Message | Cause |
| --- | --- |
| `NaN or infinity in … sent as null` | JSON cannot carry them, so they become gaps in the chart. A real NaN usually means the loss diverged. Warned once per run. |
| `run name truncated to 200 characters` | The name is cut, and kept whole as the description when you gave none. |
| `description truncated to 1000 characters` | Visin stores at most that. |
| `epoch N of run … was already recorded` | A restarted job logged an epoch that exists, and Visin keeps the first copy. A job whose epoch numbering starts over should start a new run. |
| `could not log epoch: …` | The results were not numbers or dicts. The message says what. Pass `strict=True` in tests to get the exception. |
| `no such file` / `Visin stores PNG, JPEG, GIF, WebP, MP4, MOV and PDF files` | `upload_visualization` was given a file it cannot store. |
| `cannot make an image of shape …` | An array that is not `HxW`, `HxWx3`, `HxWx4` or channels first. |
| `an image in memory needs disk to be kept on` | In-memory frames are written to `VISIN_DIR` first. Make it writable, or save a file. |

## Configuration

| Message | Cause |
| --- | --- |
| `… holds a token but is readable by others; run: chmod 600 …` | The config file or the `VISIN_ENV_FILE` is readable by other users. |
| `VISIN_ENV_FILE does not point to a file` | The path is wrong. |
| `no Visin to read from`, `no Visin dataset service`, `nowhere to send to` | `Api`, `Datasets` or `sync` have no address. Set `VISIN_URL` or `VISIN_DATASET_URL`, or pass `url=`. |

## Datasets

| Message | Cause |
| --- | --- |
| `dataset extracted, but could not delete archive` | The dataset is usable. Delete the ZIP by hand, or use `--keep-archive`. |
| A download stops and starts over | It resumes from the `.part` file with a fresh link. Run the command again. |
| Disk full while extracting | Point `VISIN_DATA_DIR` at larger storage. |

## Still stuck

Add `-v` to a command (`visin -v check`) to log every request, or set the logger to debug in a
script:

```python
import logging

logging.getLogger("visin").setLevel(logging.DEBUG)
```

Include that output, and `visin check`'s, in a bug report.
