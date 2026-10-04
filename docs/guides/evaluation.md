# Evaluating and ranking models

A training run says how a model did on its validation set. **Evaluation** says how a *checkpoint* did on a
fixed test protocol, in a way that can be compared with other checkpoints, and shows where it is weakest.

Three words matter:

- A **suite** is a written-down way of scoring: which data, which conditions (`day`, `night`, `rain`) and how many
  samples each has, which metrics, which way is better, and how the overall figure is formed. A published suite
  version never changes, so two scores on the same version can be compared. Any change to what a score means is a
  new version.
- An **evaluation** is one checkpoint's results on one suite version.
- A **verdict** is Visin's judgement of whether an evaluation can be **ranked**, and if not, exactly why. A result
  that cannot be ranked is still kept.

The page below follows one result from a file on disk to a public leaderboard. Everything has a command and a
Python call. Visin's own documentation lists every reason a result can be unranked, under *Why is my result
unranked?*.

## 1. Write the suite once

A suite is a small file kept next to your evaluation code. JSON works everywhere; YAML needs
`pip install 'visin[yaml]'`.

```json
{
  "slug": "road-test",
  "version": 1,
  "name": "Road scenes, day and night",
  "protocol": {
    "task": "semantic-segmentation",
    "data": {"kind": "external", "label": "Road test frames", "manifestSha256": "9f2b…"},
    "split": "test",
    "conditions": [{"name": "day", "sampleCount": 1200}, {"name": "night", "sampleCount": 800}],
    "metrics": [
      {"key": "mIoU_foreground", "direction": "max", "unit": "ratio", "range": {"min": 0, "max": 1}, "headline": true}
    ],
    "aggregation": "equal-mean-of-conditions",
    "evaluator": {"package": "visin-fusion"}
  }
}
```

```text
$ visin suites push suites/road-test.json --project road-seg
road-test@1  private  protocol 3a7f19c20be4
```

From Python: `visin.push_suite("suites/road-test.json", project="road-seg")`.

### The data's digest and the sample counts

A suite names the samples it scores by a digest, and says how many each condition has, so that two people scoring
"the test set" cannot be scoring different frames. If each condition has a file listing its samples (one per line),
let `visin` compute both:

```text
$ visin suites manifest day=test_day.txt night=test_night.txt
manifestSha256  3c1e90b4d2f7…
  day                   1200 samples
  night                 800 samples
```

`--json` prints the same as the fragment a suite file takes (`data.manifestSha256` and each condition's
`sampleCount`). From Python, `visin.manifest_digest({"day": samples, ...})` does the same from lists. The digest is
of the sorted samples under each condition name, so the order of a file does not matter, and a sample added,
removed, renamed or moved to another condition changes it.

Pushing is safe to repeat: the same protocol again returns the suite that exists. A **different** protocol under
a version that exists is refused, with the version to use instead. `visin suites list` and
`visin suites show road-test@1` read suites back.

## 2. Say which checkpoint you scored

A ranking is only as good as knowing which model each row is. Name the bytes:

```python
import visin

checkpoint = visin.local_checkpoint("checkpoints/epoch_40.pth", label="clftv2-epoch-40")
# or, for a model on the Hugging Face Hub, pinned to a full commit:
checkpoint = visin.hub_checkpoint("acme/clftv2", "3f2a1c9d8e7b6a5f4e3d2c1b0a99887766554433")
```

`local_checkpoint` sends the SHA-256 of the file and your label, never the path or the file. Hash the file you
actually load. Two evaluations of the same weights are the same model, however the file is named or moved. A
Hub checkpoint needs a project whose storage is Hugging Face, and a commit, not a branch.

## 3. Mark it as observed (optional)

You can skip this step: a result with a checkpoint, sample counts and scores is ranked. Do it when you want your row
labelled **observed** instead of **reported**. You say what you actually ran, and Visin compares each part with the
suite:

```python
evidence = dict(
    data={"kind": "external", "manifestSha256": visin.manifest_digest(test_splits)},
    protocol="suites/road-test.json",  # the file your evaluator loaded
    evaluator={"package": "visin-fusion", "version": "1.4.2"},
)
verdict = visin.evaluate(
    results, suite="road-test@1", checkpoint=checkpoint, sample_counts=counts, **evidence
).verdict
```

- **`data`** is the identity of what you read, in the shape the suite pins it: a manifest digest
  (`{"kind": "external", "manifestSha256": ...}`, see [`manifest_digest`](../reference/evaluation.md)), a Hub dataset
  (`{"kind": "hf", "repo": ..., "commit": ...}`) or a Visin archive (`{"kind": "visin", "archiveSha256": ...}`).
- **`protocol`** is the protocol file or dict you ran. Its digest is asked of Visin, which computes it (defaults
  filled in, keys sorted) so it always equals the suite's `digest`, and it is sent as the protocol you ran. If Visin
  cannot be reached, the file waits with the result and the digest is asked for at `visin sync`. Already have it?
  Pass `protocol_digest=`, or get it with `visin.check_protocol(path)` / `visin suites digest FILE`.
- **`evaluator`** gives the package and version (you pass it anyway for provenance); Visin enforces the suite's
  `minVersion`.
- **`classes={"scored": [...], "ignored": [...]}`** is compared with the suite's classes when you send it.

Give all of `data`, `protocol` and `evaluator` and the result is `observed`. Give less and it is `reported`, with a
warning naming what is missing; it is still ranked. What you do give must match the suite, or the result is
`incompatible` with a reason that names the part (for example `data-mismatch(manifestSha256)`: equal sample counts on
other data do not pass). `verdict.evidence` says which level you got; a promoted result is `attested`. This is your
report compared with the suite, not proof: Visin cannot see which bytes you read.

From the command line:

```text
$ visin evaluate results.json --suite road-test@1 --project road-seg --checkpoint best.pth \
      --sample-count day=1200 --sample-count night=800 \
      --data external=<manifest sha256> --protocol suites/road-test.json \
      --evaluator-package visin-fusion --evaluator-version 1.4.2
```

## 4. Check before you spend the GPU-day

`dry_run=True` stores nothing and returns the verdict at once, so you learn why a result would be unranked before you
produce more of them.

```python
verdict = visin.evaluate(
    results,  # what your evaluator produced
    suite="road-test@1",
    checkpoint=checkpoint,
    sample_counts={"day": 1200, "night": 800},
    project="road-seg",
    dry_run=True,
).verdict
print(verdict)  # eligible    or    incomplete: missing-condition(night)
```

`results` is the condition → class → metric structure your evaluator already writes. The suite names which metrics
count; everything else is kept and shown.

From the command line, with the results in a JSON file:

```text
$ visin evaluate results.json --suite road-test@1 --project road-seg \
      --checkpoint checkpoints/epoch_40.pth --label clftv2-epoch-40 \
      --sample-count day=1200 --sample-count night=800 --dry-run
checked on road-test@1; nothing was stored
  verdict   incomplete
  - missing-condition(night)
```

## 5. Record it

Drop `dry_run` to store it:

```python
evaluation = visin.evaluate(
    results,
    suite="road-test@1",
    checkpoint=checkpoint,
    sample_counts={"day": 1200, "night": 800},
    project="road-seg",
    evaluator={"package": "visin-fusion", "version": "1.4.2"},
)
evaluation.ranked  # True when it can be ranked
evaluation.verdict  # the state, the reasons, and the scores the ranking uses
```

- **`project`** defaults to `VISIN_PROJECT`, and then to the project a pipeline key is limited to.
- **`sample_counts`** is how many samples each condition scored. A suite pins those: a result that does not say, or
  says a different number, is not ranked. A condition that was skipped is not ranked either, so it cannot lift a score.
- **`run`, `epoch`** say where the checkpoint came from. That is provenance; it does not say which bytes were scored,
  which `checkpoint` does.
- **`evaluator`** is what produced the numbers (package, version, commit). `provenance` takes anything else worth
  keeping, and Visin refuses credentials in it. The machine and code are collected as for a run, unless
  `VISIN_PROVENANCE=0`.

Provenance is **reported by you**. Visin checks that results fit the suite; it cannot prove that the checkpoint or the
data were what you say, and it never calls a result verified.

### Retries, corrections and a bad network

Give your evaluation a `uuid`. Sending the same result again returns the stored one (the machine and code collected as
provenance are not part of the comparison, so running it again somewhere else is a retry too); a *different* result
under that uuid raises an error, because recorded results never change. To correct a result, send a new one with
`supersedes=<the old evaluation>`, on the same suite and the same checkpoint: the corrected result then leaves the
ranking, even if the correction cannot be ranked yet, and stays stored as an attempt.

Results are not lost to the network. With `VISIN_MODE=offline`, or when Visin cannot be reached, the evaluation waits
on disk, `evaluate` returns with `queued=True`, and `visin sync` sends it later (see [offline](offline.md)). A
refusal (an unknown suite, no access, a bad body) is not kept: it raises `visin.errors.ApiError` with the server's
own words.

### In CI

`--require-ranked` makes `visin evaluate` exit 3 when the result is not ranked, so a pipeline fails on a result
nobody can compare:

| Exit status | Meaning |
| --- | --- |
| 0 | Recorded or checked. |
| 1 | Refused, unreachable, or the arguments were not usable. |
| 3 | With `--require-ranked`: recorded, but not ranked. |

`--json` prints the outcome, including the verdict's reasons, for scripts.

To stop a change that makes a model worse, compare the new evaluation with the last good one:

```text
$ visin diff $LAST_GOOD $NEW --max-drop 0.01      # exit 4 when a score fell by more than 0.01
```

It compares the headline in every condition and the overall figure, in the suite's direction (see
[`visin diff`](cli.md#visin-diff)), and exits 5 when a condition is missing, so a skipped test set cannot hide a drop.
In Python: `visin.diff_evaluations(baseline, candidate, api.suite("road-test@1"), max_drop=0.01)`.

## 6. Read the ranking

```text
$ visin leaderboard road-test@1
road-test@1  mIoU_foreground (higher is better); ranked among 5 visible to you
    1  clftv2-epoch-40                           0.735  weakest night 0.69  gap 0.045  2 attempts
    2  baseline                                  0.702  weakest night 0.58  gap 0.122  1 attempt
  not ranked  sha256:9c1e…  incomplete: missing-condition(rain)
```

```python
board = visin.Api().leaderboard("road-test@1")
for entry in board.entries:
    print(entry.rank, entry.name, entry.headline, entry.worst_condition, entry.gap)
```

- **One row per checkpoint**, from its **latest ranked** attempt, never its best: a checkpoint cannot be re-run until
  it looks good. `attempts` says how many were recorded, and all of them stay in `visin evaluations`.
- **`worst` and `gap`** say where a model is weak: the condition the headline is worst on, and how far it falls below
  the overall figure. A model can lead overall and still collapse in one condition.
- **Ties share a rank**, and the next rank skips.
- **`candidates`** is the pool that was ranked: the evaluations of this suite in projects *you* can read. A rank is a
  position in that pool, not among everything ever run. It counts evaluations, so it is larger than the number of
  checkpoints: `board.pagination.total` is how many checkpoints were selected from it.
- **Pages.** A response carries at most 100 checkpoints. `Api.leaderboard("road-test@1", page=2, limit=50)` reads one
  page (`board.pagination` says where it sits, and ranks stay global), `unranked_page=` pages the unranked list
  separately, and `all_pages=True` reads everything and sets `board.complete`. `Api.public_leaderboard` takes the same
  `page`, `limit` and `all_pages`; `Api.iter_public_leaderboards()` walks every public leaderboard.
- **Observed only.** `observed=True` (`visin leaderboard --observed`) ranks only the results whose evaluator sent
  complete evidence. The pool and the ranks are then those of that subset; `reported` and promoted results are left out.

`visin evaluations --suite road-test@1 --state incomplete` lists what could not be ranked, and
`Api.evaluation(id)` opens one with its results and provenance.

## Rank results you already have

A result recorded without a suite (with `run.log_test_results`, which Visin keeps as an evaluation with no suite) can be
ranked without running anything again. **Promote** one onto a suite:

```python
visin.promote(result, suite="road-test@1", checkpoint=checkpoint, sample_counts={"day": 1200, "night": 800})
```

```text
$ visin promote 6ab5932a0e9a6b7570e30e2e --suite road-test@1 --checkpoint checkpoints/epoch_40.pth \
      --sample-count day=1200 --sample-count night=800
```

The first result is not changed: its numbers are copied, and the new evaluation records where it came from. A result
recorded without a suite never said which checkpoint it was or how many samples each condition scored, so you say both,
and what you say is yours to vouch for. Promoting needs manage access to the project.

## Publish a result

A ranked result stays private to its project until a manager publishes it:

```text
$ visin publish 6ab5932a0e9a6b7570e30e2e
6ab5932a0e9a6b7570e30e2e  public
$ visin leaderboard road-test@1 --public
```

The evaluation must be ranked, in a public project, on a public suite (`"visibility": "public"` in the suite file).
Only a fixed set of fields becomes public: scores, checkpoint, sample counts and the evaluator, never commands,
hosts or configuration. `visin withdraw <id>` takes it off at once. `Api.public_leaderboard("road-test@1")` reads the
anonymous ranking without a credential.
