"""Record how a checkpoint scored on a suite, and rank the checkpoints that did.

A **suite** is a written-down way of scoring a model (see :mod:`visin.suites`); an **evaluation** is one
checkpoint's results on one suite version. Visin judges each result against the suite and says, in a
:class:`~visin.models.Verdict`, whether it can be ranked and why not. A result that cannot be ranked is
still kept.

    import visin

    checkpoint = visin.local_checkpoint("checkpoints/epoch_40.pth", label="clftv2-epoch-40")
    evaluation = visin.evaluate(
        results,                                  # {"day": {"overall": {"mIoU_foreground": 0.78}}, ...}
        suite="road-test@1",
        checkpoint=checkpoint,
        sample_counts={"day": 1200, "night": 800},
        project="road-seg",
    )
    print(evaluation.verdict)                     # eligible, or incomplete: missing-condition(night)

With ``dry_run=True`` nothing is stored and the verdict comes back at once, so a pipeline learns why a
result would be unranked before it spends a day producing more of them.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid as uuidlib
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ._internal import provenance as provenance_info
from ._internal.config import Settings, read_settings
from ._internal.connect import connect
from ._internal.providers import HUB
from ._internal.reports import deliver, digest_of, discover_project
from ._internal.serialize import to_jsonable
from ._internal.spool import Spool
from ._internal.transport import worth_retrying_later
from .errors import ConfigurationError, VisinError
from .models import Evaluation, TestResult, Training
from .suites import protocol_of

logger = logging.getLogger("visin")

__all__ = [
    "evaluate",
    "hub_checkpoint",
    "local_checkpoint",
    "promote",
    "publish",
    "sha256_of",
    "withdraw",
]

_BLOCK = 8 * 2**20


def sha256_of(path: str | Path) -> str:
    """The SHA-256 of a file, read in blocks so a large checkpoint never has to fit in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def local_checkpoint(path: str | Path, label: str | None = None) -> dict[str, Any]:
    """A checkpoint held on this machine, named by the digest of the weights that will be loaded.

    Only the digest and the label are sent, never the path or the file. Two evaluations of the same weights
    are the same model whatever the file is called or wherever it moves. Hash the file you actually load.
    """
    file = Path(path)
    return {"kind": "local", "sha256": sha256_of(file), "label": label or file.stem}


def hub_checkpoint(repo: str, commit: str, path: str | None = None) -> dict[str, Any]:
    """A checkpoint on the Hugging Face Hub, pinned to a full commit.

    ``commit`` is the 40-character hash, never a branch or a tag: a branch moves, and a result must keep
    naming the bytes it scored. ``path`` is a file or folder inside the repo when the model is not all of
    it. The project's storage must be Hugging Face, or Visin refuses a Hub checkpoint.
    """
    checkpoint: dict[str, Any] = {"kind": HUB, "repo": repo, "commit": commit.lower()}
    if path:
        checkpoint["path"] = path
    return checkpoint


def _run_uuid(run: Any) -> str | None:
    if run is None:
        return None
    if isinstance(run, Training):
        return run.uuid
    if isinstance(run, str):
        return run
    return getattr(run, "training_uuid", None)


def _timestamp(value: str | datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


_SECRET_WORDS = frozenset(
    {"token", "tokens", "secret", "secrets", "password", "passwords", "passwd", "credential", "credentials"}
    | {"authorization", "apikey", "apikeys"}
)
_SECRET_PAIRS = (("api", "key"), ("private", "key"), ("secret", "key"), ("access", "key"))


def _names_a_credential(key: str) -> bool:
    """Whether Visin would refuse ``key`` in provenance: a credential word that is a word of its own.

    The same rule the server applies, so that a package called ``auth-token-helper`` in the environment is
    left out of the collected packages instead of making every evaluation fail. ``tokenizers`` is fine.
    """
    words = [
        word for word in re.split(r"[^a-z0-9]+", re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key).lower()) if word
    ]
    return any(
        word in _SECRET_WORDS or (word, following) in _SECRET_PAIRS
        for word, following in zip(words, [*words[1:], ""])
    )


def _provenance(
    evaluator: Mapping[str, Any] | None, given: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """What a reader needs to repeat the number: the machine and code, plus the evaluator and anything given.

    Collected like a run's provenance, so the same switch turns it off. The server refuses a credential
    anywhere in it: the collector already strips the secrets from the command line, and package names that
    look like credentials are dropped here.
    """
    merged: dict[str, Any] = provenance_info.collect() if provenance_info.enabled(None) else {}
    packages = merged.get("packages")
    if isinstance(packages, dict):
        merged["packages"] = {name: v for name, v in packages.items() if not _names_a_credential(str(name))}
    if evaluator:
        merged["evaluator"] = dict(evaluator)
    if given:
        merged.update(given)
    return merged or None


def _spool_name(uuid: str) -> str:
    return "evaluation-" + hashlib.sha256(uuid.encode()).hexdigest()[:16]


def _evaluation_id(evaluation: Evaluation | str) -> str:
    ident = evaluation.id if isinstance(evaluation, Evaluation) else evaluation
    if not ident:
        raise ValueError("this evaluation has no id: it was a dry run, or it is still waiting to be sent")
    return str(ident)


def evaluate(
    results: Mapping[str, Any],
    *,
    suite: str | None,
    checkpoint: Mapping[str, Any] | None,
    project: str | None = None,
    sample_counts: Mapping[str, int] | None = None,
    run: Any = None,
    epoch: int | None = None,
    epoch_uuid: str | None = None,
    evaluator: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    evidence: Mapping[str, Any] | None = None,
    data: Mapping[str, Any] | None = None,
    protocol: str | Path | Mapping[str, Any] | None = None,
    protocol_digest: str | None = None,
    classes: Mapping[str, Any] | None = None,
    uuid: str | None = None,
    supersedes: Evaluation | str | None = None,
    executed_at: str | datetime | None = None,
    status: str = "completed",
    dry_run: bool = False,
    url: str | None = None,
    token: str | None = None,
    directory: str | Path | None = None,
) -> Evaluation:
    """Record one checkpoint's results on a suite version, and return Visin's verdict.

    ``results`` is the condition → class → metric blob your evaluator produced; the suite names which
    metrics count, and everything else is kept. ``suite`` is ``"slug@version"`` (a concrete version:
    ``latest`` moves). ``checkpoint`` comes from :func:`local_checkpoint` or :func:`hub_checkpoint`.
    ``sample_counts`` says how many samples each condition scored: a suite pins those, so a result that does
    not say, or says a different number, is not ranked. ``project`` defaults to ``VISIN_PROJECT``, then to
    the project a pipeline key is limited to.

    ``run`` (a run, or its UUID), ``epoch`` and ``epoch_uuid`` say where the checkpoint came from. That is
    provenance only; it does not say which bytes were scored, which ``checkpoint`` does. ``evaluator`` is
    what produced the numbers, such as ``{"package": "visin-fusion", "version": "1.4.2", "commit":
    "9d1c2ab"}``, and ``provenance`` is anything else worth keeping. Visin refuses credentials in it.

    Send your own ``uuid`` to make a repeat harmless: the same result again returns the stored one, and a
    different result under that uuid raises. ``supersedes`` names the evaluation this one corrects.

    You need none of the following to be ranked. A result can also say what it actually ran, and Visin
    compares each part with the suite: with the data, the protocol and the evaluator all given (and matching)
    the row is labelled ``observed``, with less it is ``reported``. ``data`` is the identity of the data your
    evaluator read, in the shape the suite pins it:
    ``{"kind": "external", "manifestSha256": ...}`` (see :func:`~visin.manifest_digest`),
    ``{"kind": "hf", "repo": ..., "commit": ...}`` or ``{"kind": "visin", "archiveSha256": ...}``.
    ``protocol`` is the protocol file (or dict) your evaluator loaded: its digest is asked of Visin, and
    waits until ``visin sync`` when Visin cannot be reached. ``protocol_digest`` is that digest when you
    already have it (see :func:`~visin.check_protocol`). The evaluator's package and version come from
    ``evaluator``. ``classes`` is ``{"scored": [...], "ignored": [...]}``, compared when given. ``evidence``
    is the whole object for anything else; the arguments above are merged over it. What you give must match
    the suite, or the result is ``incompatible`` with a reason that names the part. None of it is verified:
    it is what your evaluator reported.

    Results are not lost to a bad network. With ``VISIN_MODE=offline``, or when Visin cannot be reached, the
    result waits on disk and ``visin sync`` sends it; the returned evaluation then has ``queued=True`` and no
    verdict. A refusal (an unknown suite, a bad body, no access) raises :class:`~visin.errors.ApiError`.
    With nothing configured the call does nothing and returns an evaluation that was not stored.

    ``dry_run=True`` stores nothing and returns the verdict only (``stored=False``, no id). It needs a server.
    """
    settings = read_settings(url=url, token=token, project=project, directory=directory)
    mode = settings.effective_mode
    if mode == "disabled" and not dry_run:
        logger.warning("visin: nothing is configured, so this evaluation was not recorded")
        return Evaluation(suite=suite, stored=False)

    body: dict[str, Any] = {"status": status, "results": to_jsonable(dict(results))}
    if suite:
        body["suite"] = suite
    if checkpoint:
        body["checkpoint"] = dict(checkpoint)
    if sample_counts:
        body["sampleCounts"] = {str(name): int(count) for name, count in sample_counts.items()}
    source: dict[str, Any] = {}
    training = _run_uuid(run)
    if training:
        source["trainingUuid"] = training
    if epoch_uuid:
        source["epochUuid"] = epoch_uuid
    if epoch is not None:
        source["epoch"] = int(epoch)
    if source:
        body["source"] = source
    gathered = _provenance(evaluator, to_jsonable(dict(provenance)) if provenance else None)
    if gathered:
        body["provenance"] = gathered
    reported = _evidence(evidence, data, protocol_digest, classes, evaluator)
    unresolved = (
        protocol_of(protocol) if protocol is not None and not (reported or {}).get("protocolDigest") else None
    )
    if reported:
        body["evidence"] = reported
    if executed_at is not None:
        body["executedAt"] = _timestamp(executed_at)
    if supersedes is not None:
        body["supersedesId"] = _evaluation_id(supersedes)
    body["uuid"] = uuid or str(uuidlib.uuid4())
    chosen = project or settings.project or _run_project(run)

    if mode == "offline":
        if not chosen:
            raise ConfigurationError("an evaluation needs a project: pass project= or set VISIN_PROJECT")
        return _keep(settings, {**body, "projectId": chosen}, "offline", unresolved)

    client = connect(settings, "record an evaluation")
    try:
        chosen = chosen or discover_project(client)
        if not chosen:
            raise ConfigurationError(
                "an evaluation needs a project: pass project= or set VISIN_PROJECT "
                "(a pipeline key is limited to one project and supplies it)"
            )
        body["projectId"] = chosen
        if unresolved:
            try:
                body["evidence"] = {
                    **body.get("evidence", {}),
                    "protocolDigest": digest_of(client, unresolved),
                }
            except VisinError as exc:
                if dry_run or not worth_retrying_later(exc):
                    raise
                logger.warning(
                    "visin: Visin is not answering; the evaluation waits for `visin sync`: %s", exc
                )
                return _keep(settings, body, "unreachable", unresolved)
        if dry_run:
            del body["uuid"]
            data = client.request("POST", "/evaluations/check", json=body)
            return _checked(data or {}, suite)
        try:
            data = deliver(client, {"op": "evaluation", "body": body}, _context())
        except VisinError as exc:
            if not worth_retrying_later(exc):
                raise
            logger.warning("visin: Visin is not answering; the evaluation waits for `visin sync`: %s", exc)
            return _keep(settings, body, "unreachable", None)
    finally:
        client.close()
    evaluation = Evaluation.from_json(data or {})
    _report(evaluation)
    return evaluation


def _evidence(
    evidence: Mapping[str, Any] | None,
    data: Mapping[str, Any] | None,
    protocol_digest: str | None,
    classes: Mapping[str, Any] | None,
    evaluator: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """What the evaluator reports having run: ``evidence``, with the named arguments merged over it."""
    reported: dict[str, Any] = dict(to_jsonable(dict(evidence))) if evidence else {}
    if data:
        reported["data"] = dict(data)
    if protocol_digest:
        reported["protocolDigest"] = protocol_digest
    if classes:
        reported["classes"] = {
            "scored": [str(name) for name in classes.get("scored") or ()],
            "ignored": [str(name) for name in classes.get("ignored") or ()],
        }
    if evaluator and evaluator.get("package") and evaluator.get("version") and "evaluator" not in reported:
        reported["evaluator"] = {"package": str(evaluator["package"]), "version": str(evaluator["version"])}
    return reported or None


def _context() -> Any:
    from ._internal.reports import DeliveryContext

    return DeliveryContext()


def _run_project(run: Any) -> str | None:
    value = getattr(run, "project_id", None)
    return str(value) if value else None


def _checked(data: Mapping[str, Any], suite: str | None) -> Evaluation:
    from .models import Verdict

    found = data.get("suite") or {}
    return Evaluation(
        checkpoint_key=data.get("checkpointKey"),
        suite=f"{found['slug']}@{found['version']}" if found else suite,
        suite_digest=found.get("digest") if found else None,
        verdict=Verdict.from_json(data.get("validation") or {}),
        stored=False,
    )


def _keep(settings: Settings, body: dict[str, Any], why: str, protocol: dict[str, Any] | None) -> Evaluation:
    spool = Spool(settings.directory, _spool_name(str(body["uuid"])))
    spool.append({"op": "evaluation", "body": body, **({"protocol": protocol} if protocol else {})})
    logger.info(
        "visin: evaluation %s kept in %s (%s); send it with `visin sync`", body["uuid"], spool.path, why
    )
    return Evaluation(
        uuid=str(body["uuid"]),
        suite=body.get("suite"),
        checkpoint=dict(body.get("checkpoint") or {}),
        stored=False,
        queued=True,
    )


def _report(evaluation: Evaluation) -> None:
    verdict = evaluation.verdict
    if verdict is None:
        return
    if verdict.ranked:
        logger.info("visin: evaluation recorded and ranked on %s", evaluation.suite)
    else:
        logger.warning(
            "visin: evaluation recorded but not ranked on %s: %s", evaluation.suite or "any suite", verdict
        )


def promote(
    evaluation: Evaluation | TestResult | str,
    *,
    suite: str,
    checkpoint: Mapping[str, Any],
    sample_counts: Mapping[str, int],
    url: str | None = None,
    token: str | None = None,
) -> Evaluation:
    """Rank a result recorded without a suite by copying it onto one, without running anything again.

    ``evaluation`` is the evaluation to copy, or a test result (which is one), or its id. It is not changed. A
    result recorded without a suite never said which checkpoint it was or how many samples each condition
    scored, so you say both here, and what you say is yours to vouch for. Needs manage access to the result's
    project. Promoting the same result onto the same suite twice returns the first.
    """
    ident = evaluation.id if isinstance(evaluation, (Evaluation, TestResult)) else evaluation
    if not ident:
        raise ValueError("this result has no id")
    settings = read_settings(url=url, token=token)
    client = connect(settings, "promote a result")
    try:
        data = client.request(
            "POST",
            "/evaluations/promote",
            json={
                "evaluationId": str(ident),
                "suite": suite,
                "checkpoint": dict(checkpoint),
                "sampleCounts": {str(name): int(count) for name, count in sample_counts.items()},
            },
            idempotent=True,
        )
    finally:
        client.close()
    promoted = Evaluation.from_json(data or {})
    _report(promoted)
    return promoted


def _change_publication(
    evaluation: Evaluation | str, action: str, url: str | None, token: str | None
) -> Evaluation:
    ident = _evaluation_id(evaluation)
    client = connect(read_settings(url=url, token=token), f"{action} an evaluation")
    try:
        data = client.request("POST", f"/evaluations/{quote(ident, safe='')}/{action}", idempotent=True)
    finally:
        client.close()
    return Evaluation.from_json(data or {})


def publish(evaluation: Evaluation | str, *, url: str | None = None, token: str | None = None) -> Evaluation:
    """Put one ranked evaluation on its suite's public leaderboard.

    Needs manage access to the project. The evaluation must be ranked, in a public project, on a public
    suite that takes your project's results, and the checkpoint's latest ranked result on it, or Visin
    refuses and says which: a board shows each model by its latest result, and a newer ranked result
    withdraws the older one. On a suite that approves submissions, another project's result waits for a
    manager of the suite before it is shown. What becomes public is a fixed set of fields: scores,
    checkpoint, sample counts and the evaluator, never commands, hosts or configuration. Safe to repeat.
    """
    return _change_publication(evaluation, "publish", url, token)


def withdraw(evaluation: Evaluation | str, *, url: str | None = None, token: str | None = None) -> Evaluation:
    """Take an evaluation off the public leaderboard at once. Safe to repeat."""
    return _change_publication(evaluation, "withdraw", url, token)
