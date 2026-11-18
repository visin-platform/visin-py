# VLM judge runs

A sample-scored suite pins the vocabulary, scoring rule, data identity and reference used to
compare judges. Create the suite before starting runs, and use a concrete version such as
`zod-sam-verification@1`. For unpublished work, use a **private project** as well as a private
suite: a private suite inside a public project does not make the project's runs private.

## Report an inference run

Set `VISIN_URL`, `VISIN_TOKEN` and `VISIN_PROJECT`, then report each frame's rows as it finishes:

```python
import visin

checkpoint = visin.ollama_checkpoint("qwen2.5vl:72b")
run = visin.init(
    "judge experiment",
    kind="inference",
    suite="zod-sam-verification@1",
    total=4110,
    unit="frame",
    uuid="judge-experiment-1",
)

try:
    for done, frame in enumerate(frames, start=1):
        rows = judge_frame(frame)  # produce rows conforming to this suite
        run.log_samples(rows)
        if done % 25 == 0:
            run.log_progress(done, degenerate_rate=current_degenerate_rate())
except Exception as error:
    run.fail(error)
    raise
else:
    result = visin.evaluate(suite="zod-sam-verification@1", run=run, checkpoint=checkpoint)
    print(result.verdict if result.verdict else "queued for sync")
```

Each row has a stable `id`, an `attrs` mapping and a `fields` mapping. For example, a suite may
declare the following values:

```json
{"id":"frame_000012/m7","attrs":{"class":"pedestrian","role":"classify","size":"<250","frame":"frame_000012","pixels":212,"bbox":[10,20,50,80]},"fields":{"verdict":"invalid","decision":"reject","outcome":"delete"},"parseFailed":false,"raw":"A pole","latencyMs":11390}
```

Attributes, field values and any required fields must match the suite. Raw replies are limited to
1,024 UTF-8 bytes by the server. `log_samples` accepts an iterable, queues batches of at most 2,000
rows, and warns rather than raising into inference. When the suite can be fetched online, the client
warns about vocabulary errors; the server validates every batch. An offline run cannot fetch the
suite and its rows are validated when they are synchronized.
During synchronization, adjacent sample reports are combined into batches of at most 2,000 rows;
progress and other reports remain in their original order. Online inference reports are also kept
on disk before a background sender combines them, so an interrupted sender can be synchronized later.
If a row cannot be queued locally, the run is marked failed instead of being evaluated as complete.

## Progress metrics

`log_progress(done, **metrics)` moves the run's progress bar and, when metrics are given, adds a
point at `done` frames to the run's charts. The run page reads the unit from the name, so a chart
never mixes units:

- a name ending in `_rate` is a share from 0 to 1, drawn with the other rates in percent;
- a name ending in `_ms` is a duration in milliseconds, drawn on a latency chart;
- any other number gets a chart of its own.

A mapping of numbers is one line per member, in the mapping's unit:

```python
run.log_progress(
    done, degenerate_rate=0.018, reject_rate={"pedestrian": 0.35, "sign": 0.11}, latency_ms=11390
)
```

Leave a metric out of a point that did not measure it; it is drawn as a gap, not a zero. Send a
last point when the run ends, so a finished run shows all its frames.

## Sample IDs

The sample ID must identify the same input in every run. A repeat with identical content is safe;
reusing the ID for different content produces a conflict. A rejected sample batch is retained and
blocks later completion and evaluation until the issue is resolved. See
[offline and unreliable networks](offline.md) for the spool location and sync workflow.

`evaluate` finishes the run before creating the evaluation. The server calculates sample-suite
results from the stored samples. If the run is offline or still has reports waiting, the evaluation
is queued after its samples and completion; its verdict is available once sync succeeds. A failed
run can be resumed using the same UUID and suite. A completed inference run cannot be reopened.

## Checkpoint identities

`visin.ollama_checkpoint(model)` reads the installed model's manifest digest and quantization from
Ollama. It uses `OLLAMA_HOST` or `http://127.0.0.1:11434`; pass `host=` to select another server.
Call it against the model actually used by the job: a tag may move between runs. For a human
reference use `visin.human_checkpoint("annotator_1")`; reference evaluations do not rank as models.

## Offline and command-line use

On a compute node without a route to Visin, set `VISIN_MODE=offline` and put `VISIN_DIR` on storage
accessible from a machine with a connection. After the job:

```sh
visin sync --run judge-experiment-1
```

To add a JSON Lines file of rows to an existing inference run, or inspect a pair of completed
evaluations:

```sh
visin samples push rows.jsonl --run judge-experiment-1
visin compare zod-sam-verification@1 EVALUATION_ID_A EVALUATION_ID_B
visin compare zod-sam-verification@1 EVALUATION_ID_A EVALUATION_ID_B --json
```

The comparison takes evaluation IDs, not run UUIDs. `samples push` uses the same ordered spool as
the run and returns a failure if the server refuses a batch.
