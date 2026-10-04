"""The ``visin`` command.

visin login            save the address and token, so no export lines are needed
visin logout           forget them
visin check            can this machine reach Visin, and is the token good?
visin check --write    ...and may it create a run? (makes one, then deletes it)
visin sync             send reports kept on disk by offline or cut-off runs
visin sync --list      show what is waiting, send nothing
visin runs             list recent runs
visin datasets         list the datasets on Visin
visin download zod     download a dataset (once) and print its folder
visin push zod --repo org/name   publish a dataset to Hugging Face, and point Visin at it
visin cache            show what downloads left on disk; `visin cache rm zod` deletes one
visin suites push FILE publish a scoring protocol (JSON, or YAML with visin[yaml]); list, show
visin suites digest FILE   the digest Visin gives a protocol file, to send as what an evaluator ran
visin evaluate RESULTS.json --suite road-test@1 --checkpoint best.pth --sample-count day=1200
                       record a checkpoint's results on a suite (--dry-run to only check them)
visin leaderboard road-test@1   the ranking of a suite version (--public for the anonymous one)
visin evaluations      list recorded evaluations, with their verdicts
visin diff A B         fail when evaluation B scores worse than A (--max-drop N), for CI
visin publish ID       put a ranked evaluation on the public leaderboard; `visin withdraw ID` takes it off
visin promote ID       rank an older test result on a suite without running anything again
visin version
"""

from __future__ import annotations

import argparse
import dataclasses
import getpass
import json
import logging
import os
import sys
import uuid as uuidlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ._internal.config import (
    HOSTED_DATASET_URL,
    HOSTED_URL,
    config_path,
    read_settings,
    write_config,
)
from ._internal.providers import HUB
from ._internal.spool import Spool, pending_runs
from ._internal.transport import HttpClient
from ._version import __version__
from .compare import Diff, diff_evaluations
from .errors import ApiError, ConfigurationError, VisinError
from .evaluation import evaluate, hub_checkpoint, local_checkpoint, promote, publish, withdraw
from .models import Evaluation, Leaderboard, Pagination
from .offline import sync
from .run import epoch_uuid_for
from .suites import check_protocol, load_suite, manifest_digest, push_suite, read_split


def _credential_kind(token: str | None) -> str:
    if not token:
        return "none"
    if token.startswith("vsn_live_"):
        return "API key"
    if token.count(".") == 2:
        return "JWT"
    return "unrecognised token"


def _discover(client: HttpClient) -> dict[str, Any]:
    """What the server says about itself and the credential it was sent: ``{}`` when it cannot say."""
    try:
        return dict(client.request("GET", "/.well-known/visin") or {})
    except VisinError:
        return {}


def _describe_credential(credential: Mapping[str, Any]) -> str:
    """``pipeline key for project 'Road' (vision:read, vision:write)``, as the server described it."""
    kind = {"pipeline-key": "pipeline key", "api-key": "API key", "session": "browser session"}.get(
        str(credential.get("kind")), "credential"
    )
    project = credential.get("project") or {}
    scoped = f" for project {project['name']!r}" if project.get("name") else ""
    scopes = credential.get("scopes")
    return f"{kind}{scoped}" + (f" ({', '.join(scopes)})" if scopes else "")


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2))


def _origin(settings: Any, name: str) -> str:
    where = settings.sources.get(name)
    return f"  [{where}]" if where and where != "environment" else ""


def _mask(token: str | None) -> str:
    if not token:
        return "-"
    return token[:4] + "…" + token[-4:] if len(token) > 12 else "…"


class _Printer:
    def __init__(self) -> None:
        self.failed = False

    def ok(self, text: str) -> None:
        print(f"  ok    {text}")

    def fail(self, text: str) -> None:
        self.failed = True
        print(f"  FAIL  {text}")

    def info(self, text: str) -> None:
        print(f"        {text}")


def cmd_check(args: argparse.Namespace) -> int:
    settings = read_settings(url=args.url, token=args.token, project=args.project)
    out = _Printer()
    print(f"visin {__version__}")
    print(f"  url      {settings.url or '(VISIN_URL is not set)'}{_origin(settings, 'url')}")
    print(
        f"  token    {_mask(settings.token)} ({_credential_kind(settings.token)}){_origin(settings, 'token')}"
    )
    print(f"  project  {settings.project or '-'}{_origin(settings, 'project')}")
    print(f"  mode     {settings.effective_mode}")
    print(f"  kept in  {settings.directory}")
    print()
    if not settings.url:
        out.fail(f"VISIN_URL is not set, so nothing will be reported (hosted Visin: {HOSTED_URL})")
        return 1
    client = HttpClient(settings.url, settings.token, verify=settings.verify_ssl, retries=1)
    try:
        return _check(client, settings.token, settings.project, args.write, out)
    finally:
        client.close()


def _check(client: HttpClient, token: str | None, project: str | None, write: bool, out: _Printer) -> int:
    try:
        client.request("GET", "/trainings", params={"limit": 1})
        out.ok(f"reached {client.base_url}")
    except VisinError as exc:
        out.fail(f"could not reach {client.base_url}: {exc}")
        return 1

    if not token:
        out.fail("VISIN_TOKEN is not set, so writes will be refused")
        return 1
    try:
        # Answers only a credential it accepts; the public reads above would
        # let a wrong token through as anonymous.
        client.request("GET", "/trainings/deleted", params={"limit": 1})
    except ApiError as exc:
        out.fail(f"the token was refused: {exc}")
        return 1
    except VisinError as exc:
        out.fail(str(exc))
        return 1
    credential = _discover(client).get("credential") or {}
    described = _describe_credential(credential) if credential else _credential_kind(token)
    out.ok(f"the {described} is accepted")
    if credential and "vision:write" not in (credential.get("scopes") or ["vision:write"]):
        out.info("this key can only read: it cannot report runs")

    project_id = None
    if project:
        try:
            found = client.request("GET", f"/projects/{quote(project, safe='')}") or {}
            project_id = found.get("_id")
            out.ok(f"project {found.get('name', project)!r} is visible")
        except ApiError as exc:
            hint = " (a pipeline key sees only its own project)" if exc.status in (403, 404) else ""
            out.fail(f"project {project!r}: {exc}{hint}")
        except VisinError as exc:
            out.fail(f"project {project!r}: {exc}")
    elif (credential.get("kind") == "api-key") if credential else _credential_kind(token) == "API key":
        out.info("runs go to the key's project if it is a pipeline key; otherwise set VISIN_PROJECT")
    if not credential and _credential_kind(token) == "unrecognised token":
        out.info("this is not an API key: create a pipeline key in the project's Settings")

    if not write:
        if not out.failed:
            out.info("run with --write to also create (and then delete) a test run")
        return 1 if out.failed else 0
    return _check_write(client, project, project_id, out)


def _check_write(client: HttpClient, project: str | None, project_id: str | None, out: _Printer) -> int:
    run_uuid = str(uuidlib.uuid4())
    body: dict[str, Any] = {
        "uuid": run_uuid,
        "name": "visin check",
        "status": "running",
        "tags": ["visin-check"],
    }
    if project:
        body["projectId"] = project
    try:
        training = client.request("POST", "/trainings", json=body, idempotent=True) or {}
        out.ok(f"created a test run ({run_uuid})")
        landed = training.get("projectId")
        if project_id and landed and str(landed) != str(project_id):
            # A pipeline key writes to its own project whatever the request names.
            out.fail(f"the run went to project {landed}, not {project!r}: the key belongs to another project")
    except VisinError as exc:
        if not project and isinstance(exc, ApiError) and exc.status == 400 and "project" in str(exc).lower():
            out.fail("This key needs a project: pass --project <id-or-slug> or set VISIN_PROJECT")
        else:
            out.fail(f"could not create a run: {exc}")
        return 1
    try:
        client.request(
            "POST",
            "/epochs/upload",
            json={
                "training_uuid": run_uuid,
                "epoch_uuid": epoch_uuid_for(run_uuid, 1),
                "epoch": 1,
                "results": {"train": {"loss": 1.0}},
            },
            idempotent=True,
        )
        out.ok("sent an epoch to it")
    except VisinError as exc:
        out.fail(f"could not send an epoch: {exc}")
    ident = training.get("_id")
    try:
        if ident:
            client.request("DELETE", f"/trainings/{ident}")
            out.ok("deleted the test run")
    except VisinError as exc:
        out.fail(f"could not delete the test run {ident}; delete it in the app: {exc}")
    return 1 if out.failed else 0


def _ask(question: str, default: str | None = None) -> str:
    shown = f" [{default}]" if default else ""
    answer = input(f"{question}{shown}: ").strip()
    return answer or default or ""


def cmd_login(args: argparse.Namespace) -> int:
    try:
        current = read_settings()
    except VisinError as exc:
        print(f"visin login: {exc}", file=sys.stderr)
        return 1
    interactive = sys.stdin.isatty()
    url = args.url or (_ask("Visin API address", current.url or HOSTED_URL) if interactive else None)
    url = (url or current.url or HOSTED_URL).strip().rstrip("/")
    if "://" not in url:
        url = "https://" + url
    token = args.token or (
        getpass.getpass("Token (a pipeline key or API key): ").strip() if interactive else ""
    )
    if not token:
        print("visin login: pass --token, or run it in a terminal to be asked", file=sys.stderr)
        return 1
    found: dict[str, Any] = {}
    if not args.no_check:
        out = _Printer()
        client = HttpClient(url, token, verify=current.verify_ssl, retries=1)
        try:
            if _check(client, token, args.project, False, out) != 0:
                print("visin login: not saved, because the check failed (use --no-check to save anyway)")
                return 1
            found = _discover(client)
        finally:
            client.close()
    values = {"VISIN_URL": url, "VISIN_TOKEN": token}
    if args.project:
        values["VISIN_PROJECT"] = args.project
    if not current.dataset_url:
        if found.get("datasetApiUrl"):
            values["VISIN_DATASET_URL"] = str(found["datasetApiUrl"])
        elif url.removesuffix("/api") == HOSTED_URL:
            values["VISIN_DATASET_URL"] = HOSTED_DATASET_URL
    if found.get("appUrl") and not current.app_url:
        values["VISIN_APP_URL"] = str(found["appUrl"])
    path = write_config(values)
    print(f"saved to {path}")
    shadowing = [name for name in values if (os.environ.get(name) or "").strip()]
    if shadowing:
        print(f"note: {', '.join(shadowing)} set in this environment win over the saved values")
    return 0


def cmd_logout(_args: argparse.Namespace) -> int:
    path = config_path()
    if not path.exists():
        print(f"nothing saved at {path}")
        return 0
    os.remove(path)
    print(f"removed {path}")
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    settings = read_settings(url=args.url, token=args.token, directory=args.dir)
    uuids = [args.run] if args.run else pending_runs(settings.directory)
    if not uuids and args.list and args.json:
        _print_json({})
        return 0
    if not uuids:
        print(f"nothing waiting in {settings.directory}")
        return 0
    if args.list and args.json:
        _print_json({uuid: Spool(settings.directory, uuid).count() for uuid in uuids})
        return 0
    if args.list:
        for uuid in uuids:
            print(f"{uuid}  {Spool(settings.directory, uuid).count()} reports")
        return 0
    try:
        results = sync(
            settings.directory,
            url=args.url,
            token=args.token,
            training_uuid=args.run,
            keep_rejected=not args.drop_rejected,
        )
    except VisinError as exc:
        print(f"visin sync: {exc}", file=sys.stderr)
        return 1
    status = 0
    for result in results:
        line = f"{result.training_uuid}  sent {result.sent}"
        if result.rejected:
            line += f", refused {len(result.rejected)}"
        if result.remaining:
            line += f", {result.remaining} still waiting"
        print(line)
        for message in result.rejected:
            print(f"    refused: {message}")
        if not result.complete:
            status = 1
    unsent = len(uuids) - len(results)
    if unsent > 0:
        print(f"stopped: Visin is not answering; {unsent} more runs are waiting")
        status = 1
    return status


def cmd_runs(args: argparse.Namespace) -> int:
    from .api import Api

    try:
        api = Api(args.url, args.token)
    except VisinError as exc:
        print(f"visin runs: {exc}", file=sys.stderr)
        return 1
    with api:
        try:
            runs = list(api.trainings(project=args.project, status=args.status, limit=args.limit))
        except VisinError as exc:
            print(f"visin runs: {exc}", file=sys.stderr)
            return 1
    if args.json:
        settings = read_settings(url=args.url)
        _print_json(
            [
                {
                    **{
                        k: v
                        for k, v in dataclasses.asdict(run).items()
                        if k not in ("raw", "metadata", "metrics")
                    },
                    "url": settings.run_link(run.id),
                }
                for run in runs
            ]
        )
        return 0
    for run in runs:
        updated = (run.updated_at or "")[:16].replace("T", " ")
        print(f"{run.uuid:36}  {run.status or '':9}  {updated:16}  {run.name}")
    return 0


def cmd_datasets(args: argparse.Namespace) -> int:
    from .datasets import Datasets

    try:
        with Datasets(args.url, args.token) as datasets:
            found = datasets.list(search=args.search)
    except VisinError as exc:
        print(f"visin datasets: {exc}", file=sys.stderr)
        return 1
    if args.json:
        _print_json(
            [
                {
                    "id": d.id,
                    "name": d.name,
                    "size": d.size,
                    "filename": d.filename,
                    "downloadable": d.downloadable,
                }
                for d in found
            ]
        )
        return 0
    for dataset in found:
        shown = f"{dataset.size / 2**30:6.1f} GB" if dataset.size else "  no zip "
        print(f"{dataset.id:24}  {shown}  {dataset.name}")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    from .datasets import Datasets

    logging.getLogger("visin").setLevel(logging.INFO)  # show the progress
    try:
        with Datasets(args.url, args.token, directory=args.dir) as datasets:
            root = datasets.download(args.dataset, unzip=not args.no_unzip, keep_archive=args.keep_archive)
    except VisinError as exc:
        print(f"visin download: {exc}", file=sys.stderr)
        return 1
    print(root)
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    from .datasets import Datasets

    logging.getLogger("visin").setLevel(logging.INFO)
    try:
        with Datasets(args.url, args.token, directory=args.dir) as datasets:
            revision = datasets.push(args.dataset, args.repo, private=not args.public)
    except VisinError as exc:
        print(f"visin push: {exc}", file=sys.stderr)
        return 1
    print(f"{args.repo}@{revision}")
    return 0


def _size(value: int) -> str:
    for unit, scale in (("GB", 2**30), ("MB", 2**20), ("KB", 2**10)):
        if value >= scale:
            return f"{value / scale:.1f} {unit}"
    return f"{value} B"


def cmd_cache(args: argparse.Namespace) -> int:
    from .datasets import cached_datasets, remove_cached

    try:
        if args.action == "rm":
            removed = remove_cached(args.dataset, args.dir)
            for item in removed:
                print(f"removed {item.path}  ({_size(item.size)})")
            print(f"freed {_size(sum(item.size for item in removed))}")
            return 0
        items = cached_datasets(args.dir)
    except VisinError as exc:
        print(f"visin cache: {exc}", file=sys.stderr)
        return 1
    if args.json:
        _print_json([{**dataclasses.asdict(item), "path": str(item.path)} for item in items])
        return 0
    for item in items:
        print(f"{item.kind:9}  {_size(item.size):>9}  {item.name}  {item.path}")
    if not items:
        print("nothing downloaded")
    else:
        print(f"{_size(sum(item.size for item in items))} in {len(items)} items")
    return 0


def cmd_version(_args: argparse.Namespace) -> int:
    print(f"visin {__version__}")
    return 0


EXIT_NOT_RANKED = 3
EXIT_REGRESSION = 4
EXIT_NOT_COMPARABLE = 5


def _fail(command: str, exc: BaseException) -> int:
    print(f"visin {command}: {exc}", file=sys.stderr)
    return 1


def _checkpoint(args: argparse.Namespace) -> dict[str, Any] | None:
    """The checkpoint the flags describe, or ``None`` when none was given."""
    if args.checkpoint and args.hub_repo:
        raise ConfigurationError("give either --checkpoint (a file on this machine) or --hub-repo, not both")
    if args.checkpoint:
        return local_checkpoint(args.checkpoint, args.label)
    if args.hub_repo:
        if not args.hub_commit:
            raise ConfigurationError(
                "--hub-repo needs --hub-commit: the full 40-character commit, never a branch"
            )
        return hub_checkpoint(args.hub_repo, args.hub_commit, args.hub_path)
    return None


def _sample_counts(args: argparse.Namespace) -> dict[str, int]:
    """Samples per condition, from ``--sample-counts FILE`` and each ``--sample-count NAME=N``."""
    counts: dict[str, int] = {}
    if args.sample_counts:
        try:
            loaded = json.loads(Path(args.sample_counts).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigurationError(f"cannot read --sample-counts {args.sample_counts}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ConfigurationError('--sample-counts should hold an object like {"day": 1200}')
        counts.update({str(name): int(value) for name, value in loaded.items()})
    for item in args.sample_count or ():
        name, separator, value = item.rpartition("=")
        if not (separator and name and value.isdigit()):
            raise ConfigurationError(f"--sample-count {item!r}: expected NAME=NUMBER, such as day=1200")
        counts[name] = int(value)
    return counts


def _hub_data(value: str) -> dict[str, Any] | None:
    repo, at, commit = value.partition("@")
    return {"kind": HUB, "repo": repo, "commit": commit} if at and repo and commit else None


_DATA_KINDS: dict[str, Callable[[str], dict[str, Any] | None]] = {
    "external": lambda value: {"kind": "external", "manifestSha256": value},
    "visin": lambda value: {"kind": "visin", "archiveSha256": value},
    HUB: _hub_data,
}


def _data_evidence(text: str | None) -> dict[str, Any] | None:
    """The data identity ``--data KIND=VALUE`` names.

    A digest for ``external`` or ``visin``, and ``repo@commit`` for ``hf``. Each kind parses its own value
    in ``_DATA_KINDS``, so another kind of data is one entry there.
    """
    if not text:
        return None
    kind, separator, value = text.partition("=")
    if not (separator and value):
        raise ConfigurationError(
            f"--data {text!r}: expected KIND=VALUE, such as external=<manifest sha256>, "
            "visin=<archive sha256> or hf=org/name@<40-character commit>"
        )
    parse = _DATA_KINDS.get(kind)
    if parse is None:
        raise ConfigurationError(f"--data {text!r}: the kind is {', '.join(_DATA_KINDS)}")
    parsed = parse(value)
    if parsed is None:
        raise ConfigurationError(f"--data {text!r}: a Hub dataset is hf=org/name@<commit>")
    return parsed


def _class_names(text: str | None) -> list[str]:
    return [name.strip() for name in (text or "").split(",") if name.strip()]


def _classes_evidence(args: argparse.Namespace) -> dict[str, list[str]] | None:
    if args.classes_ignored is not None and args.classes_scored is None:
        raise ConfigurationError(
            "--classes-ignored needs --classes-scored: say which classes were scored too"
        )
    if args.classes_scored is None:
        return None
    return {"scored": _class_names(args.classes_scored), "ignored": _class_names(args.classes_ignored)}


def _print_evaluation(evaluation: Evaluation) -> None:
    if evaluation.queued:
        print(f"kept on disk, to send with `visin sync` (uuid {evaluation.uuid})")
        return
    if not evaluation.stored and evaluation.verdict is None:
        print("not recorded: nothing is configured to send to")
        return
    verdict = evaluation.verdict
    where = f" on {evaluation.suite}" if evaluation.suite else ""
    if evaluation.stored:
        print(f"recorded evaluation {evaluation.id}{where}")
    else:
        print(f"checked{where}; nothing was stored")
    if verdict is None:
        return
    print(f"  verdict   {verdict.state}")
    if verdict.evidence:
        print(f"  evidence  {verdict.evidence}")
    for reason in verdict.reasons:
        print(f"  - {reason}")
    for warning in verdict.warnings:
        print(f"  ! {warning}")
    if not verdict.ranked and verdict.reasons:
        print(
            "  every reason is explained, with how to fix it, in the docs under 'Why is my result unranked?'"
        )


def cmd_evaluate(args: argparse.Namespace) -> int:
    try:
        results = json.loads(Path(args.results).read_text(encoding="utf-8"))
        if not isinstance(results, dict):
            raise ConfigurationError(
                f"{args.results} should hold the results object: conditions, then metrics"
            )
        evaluator = {
            key: value
            for key, value in (
                ("package", args.evaluator_package),
                ("version", args.evaluator_version),
                ("commit", args.evaluator_commit),
            )
            if value
        }
        evaluation = evaluate(
            results,
            suite=args.suite,
            checkpoint=_checkpoint(args),
            project=args.project,
            sample_counts=_sample_counts(args) or None,
            run=args.run,
            epoch=args.epoch,
            evaluator=evaluator or None,
            data=_data_evidence(args.data),
            protocol=args.protocol,
            protocol_digest=args.protocol_digest,
            classes=_classes_evidence(args),
            uuid=args.uuid,
            supersedes=args.supersedes,
            dry_run=args.dry_run,
            url=args.url,
            token=args.token,
        )
    except (OSError, ValueError, VisinError) as exc:
        return _fail("evaluate", exc)
    if args.json:
        _print_json(_evaluation_json(evaluation))
    else:
        _print_evaluation(evaluation)
    if args.require_ranked and not evaluation.ranked:
        return EXIT_NOT_RANKED
    return 0


def _evaluation_json(evaluation: Evaluation) -> dict[str, Any]:
    verdict = evaluation.verdict
    return {
        "id": evaluation.id,
        "uuid": evaluation.uuid,
        "suite": evaluation.suite,
        "stored": evaluation.stored,
        "queued": evaluation.queued,
        "ranked": evaluation.ranked,
        "verdict": None
        if verdict is None
        else {
            "state": verdict.state,
            "reasons": [{"code": r.code, "detail": r.detail} for r in verdict.reasons],
            "warnings": [{"code": r.code, "detail": r.detail} for r in verdict.warnings],
        },
    }


def cmd_suites(args: argparse.Namespace) -> int:
    from .api import Api

    try:
        if args.action == "push":
            suite = push_suite(
                load_suite(args.file),
                project=args.project,
                visibility="public" if args.public else "private" if args.private else None,
                url=args.url,
                token=args.token,
            )
            print(f"{suite.ref}  {suite.visibility}  protocol {(suite.digest or '')[:12]}")
            return 0
        if args.action == "manifest":
            return _print_manifest(args)
        if args.action == "digest":
            print(check_protocol(args.file, url=args.url, token=args.token))
            return 0
        with Api(args.url, args.token) as api:
            if args.action == "list":
                found = api.suites(project=args.project, include_archived=args.all)
                if args.json:
                    _print_json([dataclasses.asdict(item) | {"raw": None} for item in found])
                    return 0
                for item in found:
                    archived = "  archived" if item.archived_at else ""
                    print(f"{item.ref:30}  {item.visibility or '':8}  {item.name}{archived}")
                return 0
            suite = api.suite(args.suite)
            if args.json:
                _print_json(suite.raw)
                return 0
            headline = suite.headline
            protocol = suite.protocol
            print(f"{suite.ref}  {suite.name}  ({suite.visibility})")
            print(f"  protocol    {suite.digest}")
            print(f"  task        {protocol.get('task')}, split {protocol.get('split')}")
            better = "higher" if headline.get("direction") == "max" else "lower"
            print(f"  headline    {headline.get('key')}, {better} is better")
            print(f"  overall     {protocol.get('aggregation')}")
            for condition in protocol.get("conditions") or ():
                print(f"  condition   {condition.get('name')}  {condition.get('sampleCount')} samples")
            return 0
    except (OSError, ValueError, VisinError) as exc:
        return _fail("suites", exc)


def _print_manifest(args: argparse.Namespace) -> int:
    conditions: dict[str, list[str]] = {}
    for item in args.splits:
        name, separator, path = item.partition("=")
        if not (separator and name and path):
            raise ConfigurationError(f"{item!r}: expected NAME=FILE, such as day_fair=test_day_fair.txt")
        if name in conditions:
            raise ConfigurationError(f"condition {name!r} is given twice")
        conditions[name] = read_split(path)
    digest = manifest_digest(conditions)
    counts = [{"name": name, "sampleCount": len(samples)} for name, samples in conditions.items()]
    if args.json:
        _print_json({"manifestSha256": digest, "conditions": counts})
        return 0
    print(f"manifestSha256  {digest}")
    for item in counts:
        print(f"  {item['name']:20}  {item['sampleCount']} samples")
    return 0


def _page_note(label: str, shown: int, pagination: Pagination | None, complete: bool) -> str | None:
    if pagination is None or (pagination.pages <= 1 and not complete):
        return None
    if complete:
        return f"all {pagination.total} {label}"
    return f"{label}: page {pagination.page} of {pagination.pages} ({shown} of {pagination.total} shown)"


def _print_leaderboard(board: Leaderboard) -> None:
    where = "public" if board.public else "visible to you"
    unit = "higher" if board.direction == "max" else "lower"
    print(f"{board.suite}  {board.headline_key} ({unit} is better); ranked among {board.candidates} {where}")
    for entry in board.entries:
        print(
            f"  {entry.rank:>3}  {entry.name:40}  {entry.headline:.4g}  "
            f"weakest {entry.worst_condition} {entry.worst_value:.4g}  gap {entry.gap:.4g}  "
            f"{entry.attempts} attempt{'s' if entry.attempts != 1 else ''}"
        )
    for row in board.unranked:
        reasons = ", ".join(
            f"{r.get('code')}" + (f"({r['detail']})" if r.get("detail") else "")
            for r in row.get("reasons") or ()
        )
        print(f"  not ranked  {row.get('checkpointKey')}  {row.get('state')}: {reasons}")
    notes = (
        _page_note("checkpoints", len(board.entries), board.pagination, board.complete),
        _page_note("unranked", len(board.unranked), board.unranked_pagination, board.complete),
    )
    for note in notes:
        if note:
            print(f"  ({note})")


def cmd_leaderboard(args: argparse.Namespace) -> int:
    from .api import Api

    try:
        with Api(args.url, args.token) as api:
            if args.public:
                if args.unranked_page is not None:
                    raise ValueError("the public ranking has no unranked list")
                board = api.public_leaderboard(
                    args.suite, page=args.page, limit=args.limit, all_pages=args.all, observed=args.observed
                )
            else:
                board = api.leaderboard(
                    args.suite,
                    page=args.page,
                    limit=args.limit,
                    unranked_page=args.unranked_page,
                    all_pages=args.all,
                    observed=args.observed,
                )
    except (ValueError, VisinError) as exc:
        return _fail("leaderboard", exc)
    if args.json:
        _print_json(board.raw)
    else:
        _print_leaderboard(board)
    return 0


def _print_diff(result: Diff) -> None:
    print(f"{result.suite}: {result.candidate} against {result.baseline} (allowed drop {result.max_drop:g})")
    for change in result.changes:
        if change.improvement is None:
            before = "-" if change.baseline is None else f"{change.baseline:.4g}"
            after = "-" if change.candidate is None else f"{change.candidate:.4g}"
            detail = f"{before} -> {after}  not comparable"
        else:
            detail = f"{change.baseline:.4g} -> {change.candidate:.4g}  {change.improvement:+.4g}"
        print(f"  {change.status:9}  {change.scope:20}  {change.metric}  {detail}")
    for problem in result.problems:
        print(f"  ! {problem}")
    print("  ok" if result.passed else "  FAILED")


def cmd_diff(args: argparse.Namespace) -> int:
    from .api import Api

    try:
        with Api(args.url, args.token) as api:
            baseline = api.evaluation(args.baseline, project=args.project)
            candidate = api.evaluation(args.candidate, project=args.project)
            if not baseline.suite:
                raise ValueError(
                    "the baseline names no suite: only results on one suite version can be compared"
                )
            result = diff_evaluations(
                baseline,
                candidate,
                api.suite(baseline.suite),
                max_drop=args.max_drop,
                all_metrics=args.all_metrics,
            )
    except (ValueError, VisinError) as exc:
        return _fail("diff", exc)
    if args.json:
        _print_json(
            {
                "suite": result.suite,
                "baseline": result.baseline,
                "candidate": result.candidate,
                "maxDrop": result.max_drop,
                "passed": result.passed,
                "problems": list(result.problems),
                "changes": [dataclasses.asdict(change) for change in result.changes],
            }
        )
    else:
        _print_diff(result)
    if result.missing or result.problems:
        return EXIT_NOT_COMPARABLE
    return EXIT_REGRESSION if result.regressions else 0


def cmd_evaluations(args: argparse.Namespace) -> int:
    from .api import Api

    try:
        with Api(args.url, args.token) as api:
            found = api.evaluations(project=args.project, suite=args.suite, state=args.state)[: args.limit]
    except VisinError as exc:
        return _fail("evaluations", exc)
    if args.json:
        _print_json([_evaluation_json(item) | {"checkpoint": item.checkpoint} for item in found])
        return 0
    for item in found:
        verdict = str(item.verdict) if item.verdict else "-"
        name = item.checkpoint.get("label") or item.checkpoint.get("repo") or "-"
        published = "  public" if item.published else ""
        print(f"{item.id}  {item.suite or '-':24}  {name:28}  {verdict}{published}")
    return 0


def cmd_publication(args: argparse.Namespace) -> int:
    action = publish if args.command == "publish" else withdraw
    try:
        evaluation = action(args.evaluation, url=args.url, token=args.token)
    except VisinError as exc:
        return _fail(args.command, exc)
    state = (
        "waiting for approval"
        if evaluation.published and evaluation.pending_approval
        else "public"
        if evaluation.public
        else "hidden by the suite's managers"
        if evaluation.hidden
        else "not public"
    )
    print(f"{evaluation.id}  {state}")
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    try:
        checkpoint = _checkpoint(args)
        if checkpoint is None:
            raise ConfigurationError(
                "say which checkpoint it was: --checkpoint FILE, or --hub-repo and --hub-commit"
            )
        counts = _sample_counts(args)
        if not counts:
            raise ConfigurationError(
                "say how many samples each condition scored: --sample-count day=1200 ..."
            )
        evaluation = promote(
            args.evaluation,
            suite=args.suite,
            checkpoint=checkpoint,
            sample_counts=counts,
            url=args.url,
            token=args.token,
        )
    except (OSError, ValueError, VisinError) as exc:
        return _fail("promote", exc)
    _print_evaluation(evaluation)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="visin", description="Report training runs to Visin.")
    parser.add_argument("-v", "--verbose", action="store_true", help="log what the client does")
    commands = parser.add_subparsers(dest="command", metavar="command")
    commands.required = True

    def server(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--url", help="the Visin API address (default: VISIN_URL)")
        sub.add_argument("--token", help="the credential (default: VISIN_TOKEN)")

    login = commands.add_parser("login", help="save the address and token for this machine")
    login.add_argument("--url", help="the Visin API address (asked for in a terminal)")
    login.add_argument("--token", help="the credential (asked for in a terminal, without echo)")
    login.add_argument("--project", help="the project runs go to; saved as VISIN_PROJECT")
    login.add_argument("--no-check", action="store_true", help="save without contacting Visin")
    login.set_defaults(handler=cmd_login)

    logout = commands.add_parser("logout", help="delete the saved address and token")
    logout.set_defaults(handler=cmd_logout)

    check = commands.add_parser("check", help="check the connection and the token")
    server(check)
    check.add_argument("--project", help="the project runs go to (default: VISIN_PROJECT)")
    check.add_argument(
        "--write", action="store_true", help="also create a test run, send an epoch, delete it"
    )
    check.set_defaults(handler=cmd_check)

    sync_cmd = commands.add_parser("sync", help="send reports kept on disk")
    server(sync_cmd)
    sync_cmd.add_argument("--dir", help="where they are kept (default: VISIN_DIR or ~/.visin)")
    sync_cmd.add_argument("--run", help="only this run's reports (its training UUID)")
    sync_cmd.add_argument("--list", action="store_true", help="show what is waiting and send nothing")
    sync_cmd.add_argument(
        "--drop-rejected",
        action="store_true",
        help="forget reports Visin refuses, instead of keeping them to retry",
    )
    sync_cmd.add_argument(
        "--json", action="store_true", help="with --list: print {run uuid: reports} as JSON"
    )
    sync_cmd.set_defaults(handler=cmd_sync)

    runs = commands.add_parser("runs", help="list recent runs")
    server(runs)
    runs.add_argument("--project", help="only this project's runs (its id or slug)")
    runs.add_argument("--status", choices=["pending", "running", "completed", "failed"])
    runs.add_argument("--limit", type=int, default=20)
    runs.add_argument("--json", action="store_true", help="print the runs as JSON, with a link to each")
    runs.set_defaults(handler=cmd_runs)

    def dataset_server(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--url", help="the Visin dataset service (default: VISIN_DATASET_URL)")
        sub.add_argument("--token", help="the credential, for private datasets (default: VISIN_TOKEN)")

    datasets = commands.add_parser("datasets", help="list the datasets on Visin")
    dataset_server(datasets)
    datasets.add_argument("--search", help="only datasets whose name matches")
    datasets.add_argument("--json", action="store_true", help="print the datasets as JSON")
    datasets.set_defaults(handler=cmd_datasets)

    download = commands.add_parser("download", help="download a dataset (once) and print its folder")
    dataset_server(download)
    download.add_argument("dataset", help="its name (e.g. zod) or id")
    download.add_argument("--dir", help="where datasets go (default: VISIN_DATA_DIR)")
    download.add_argument(
        "--no-unzip", action="store_true", help="keep the ZIP without extracting; print its path"
    )
    download.add_argument("--keep-archive", action="store_true", help="retain the ZIP after extraction")
    download.set_defaults(handler=cmd_download)

    push = commands.add_parser("push", help="publish a dataset to Hugging Face and point Visin at it")
    dataset_server(push)
    push.add_argument("dataset", help="its name (e.g. zod) or id")
    push.add_argument("--repo", required=True, help="the Hub dataset repo, org/name (created if needed)")
    push.add_argument("--public", action="store_true", help="make a new repo public (default: private)")
    push.add_argument("--dir", help="where datasets go (default: VISIN_DATA_DIR)")
    push.set_defaults(handler=cmd_push)

    cache = commands.add_parser("cache", help="show or delete downloaded datasets")
    cache.add_argument("--dir", help="the data directory (default: VISIN_DATA_DIR)")
    cache.add_argument("--json", action="store_true", help="print the list as JSON")
    cache_actions = cache.add_subparsers(dest="action", metavar="action")
    remove = cache_actions.add_parser("rm", help="delete a downloaded dataset, by name or id")
    remove.add_argument("dataset", help="its name or id")
    remove.add_argument(
        "--dir", default=argparse.SUPPRESS, help="the data directory (default: VISIN_DATA_DIR)"
    )
    cache.set_defaults(handler=cmd_cache, action=None)

    def checkpoint_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--checkpoint", help="the weights file that was loaded; only its SHA-256 is sent")
        sub.add_argument("--label", help="a name for a --checkpoint (default: the file name)")
        sub.add_argument("--hub-repo", help="the checkpoint's Hugging Face repo, org/name")
        sub.add_argument("--hub-commit", help="the full 40-character commit of --hub-repo")
        sub.add_argument("--hub-path", help="a file or folder inside --hub-repo")
        sub.add_argument(
            "--sample-count",
            action="append",
            metavar="NAME=N",
            help="samples scored in a condition; repeat for each condition",
        )
        sub.add_argument("--sample-counts", metavar="FILE", help='the counts as a JSON object, {"day": 1200}')

    suites = commands.add_parser("suites", help="publish, list and show scoring protocols")
    suite_actions = suites.add_subparsers(dest="action", metavar="action")
    suite_actions.required = True
    suite_push = suite_actions.add_parser("push", help="publish a suite version from a JSON or YAML file")
    server(suite_push)
    suite_push.add_argument("file", help="the suite file")
    suite_push.add_argument(
        "--project", help="the project it belongs to (default: the file's, VISIN_PROJECT)"
    )
    visible = suite_push.add_mutually_exclusive_group()
    visible.add_argument("--public", action="store_true", help="let anyone read the protocol")
    visible.add_argument("--private", action="store_true", help="only the project's readers (the default)")
    suite_manifest = suite_actions.add_parser(
        "manifest", help="the manifest digest and sample counts of split files, for a suite's data"
    )
    suite_manifest.add_argument(
        "splits", nargs="+", metavar="NAME=FILE", help="a condition and the file listing its samples"
    )
    suite_manifest.add_argument("--json", action="store_true", help="print it as the JSON a suite file takes")
    suite_digest = suite_actions.add_parser(
        "digest", help="the digest Visin gives a suite or protocol file, without publishing it"
    )
    server(suite_digest)
    suite_digest.add_argument("file", help="a suite file, or a file holding just the protocol")
    suite_list = suite_actions.add_parser("list", help="list the suites you can read")
    server(suite_list)
    suite_list.add_argument("--project", help="only this project's suites")
    suite_list.add_argument("--all", action="store_true", help="include archived suites")
    suite_list.add_argument("--json", action="store_true", help="print the suites as JSON")
    suite_show = suite_actions.add_parser(
        "show", help="show one suite: slug@version, or a slug for the latest"
    )
    server(suite_show)
    suite_show.add_argument("suite")
    suite_show.add_argument("--json", action="store_true", help="print the suite as JSON")
    suites.set_defaults(handler=cmd_suites)

    evaluate_cmd = commands.add_parser("evaluate", help="record a checkpoint's results on a suite")
    server(evaluate_cmd)
    evaluate_cmd.add_argument("results", help="a JSON file of results: conditions, then metrics")
    evaluate_cmd.add_argument("--suite", required=True, help="slug@version, such as road-test@1")
    evaluate_cmd.add_argument("--project", help="the project (default: VISIN_PROJECT, or the key's project)")
    checkpoint_args(evaluate_cmd)
    evaluate_cmd.add_argument("--run", help="the training UUID the checkpoint came from")
    evaluate_cmd.add_argument("--epoch", type=int, help="the epoch it was saved at")
    evaluate_cmd.add_argument("--evaluator-package", help="what produced the numbers, e.g. visin-fusion")
    evaluate_cmd.add_argument("--evaluator-version")
    evaluate_cmd.add_argument("--evaluator-commit")
    evaluate_cmd.add_argument(
        "--data",
        metavar="KIND=VALUE",
        help="the data you read: external=<manifest sha256>, visin=<archive sha256> or hf=org/name@<commit>",
    )
    evaluate_cmd.add_argument(
        "--protocol",
        metavar="FILE",
        help="the suite or protocol file you ran; its digest is asked of Visin (at `visin sync` if offline)",
    )
    evaluate_cmd.add_argument("--protocol-digest", help="the digest of the protocol you ran, if you have it")
    evaluate_cmd.add_argument("--classes-scored", help="comma-separated class ids you scored")
    evaluate_cmd.add_argument("--classes-ignored", help="comma-separated class ids you left out")
    evaluate_cmd.add_argument("--uuid", help="your id for it: sending the same result again is harmless")
    evaluate_cmd.add_argument("--supersedes", help="the id of the evaluation this one corrects")
    evaluate_cmd.add_argument("--dry-run", action="store_true", help="judge it and store nothing")
    evaluate_cmd.add_argument(
        "--require-ranked",
        action="store_true",
        help=f"exit {EXIT_NOT_RANKED} when the result is not ranked, for CI",
    )
    evaluate_cmd.add_argument("--json", action="store_true", help="print the outcome as JSON")
    evaluate_cmd.set_defaults(handler=cmd_evaluate)

    board = commands.add_parser("leaderboard", help="show the ranking of a suite version")
    server(board)
    board.add_argument("suite", help="slug@version")
    board.add_argument("--public", action="store_true", help="the anonymous ranking of published results")
    board.add_argument(
        "--page", type=int, help="the page of ranked checkpoints (default 1; ranks stay global)"
    )
    board.add_argument("--limit", type=int, help="checkpoints per page (default and maximum 100)")
    board.add_argument("--unranked-page", type=int, help="the page of unranked checkpoints (default 1)")
    board.add_argument("--all", action="store_true", help="read every page of both lists")
    board.add_argument(
        "--observed", action="store_true", help="rank only results whose evaluator sent complete evidence"
    )
    board.add_argument("--json", action="store_true", help="print it as JSON")
    board.set_defaults(handler=cmd_leaderboard)

    diff = commands.add_parser(
        "diff", help="fail when one evaluation scores worse than another on the same suite version"
    )
    server(diff)
    diff.add_argument(
        "baseline", help="the evaluation to compare against: its id, or its uuid with --project"
    )
    diff.add_argument("candidate", help="the evaluation that must not be worse")
    diff.add_argument("--project", help="the project, when the evaluations are named by uuid")
    diff.add_argument(
        "--max-drop", type=float, default=0.0, help="how far a score may fall, in its own units (default 0)"
    )
    diff.add_argument(
        "--all-metrics", action="store_true", help="every metric the suite names, not only the headline"
    )
    diff.add_argument("--json", action="store_true", help="print the diff as JSON")
    diff.set_defaults(handler=cmd_diff)

    listing = commands.add_parser("evaluations", help="list recorded evaluations")
    server(listing)
    listing.add_argument("--project", help="only this project's evaluations")
    listing.add_argument("--suite", help="only this suite version, slug@version")
    listing.add_argument(
        "--state", choices=["eligible", "incomplete", "incompatible", "exploratory", "legacy-unverified"]
    )
    listing.add_argument("--limit", type=int, default=20)
    listing.add_argument("--json", action="store_true", help="print them as JSON")
    listing.set_defaults(handler=cmd_evaluations)

    for name, text in (
        ("publish", "put a ranked evaluation on its suite's public leaderboard"),
        ("withdraw", "take an evaluation off the public leaderboard"),
    ):
        change = commands.add_parser(name, help=text)
        server(change)
        change.add_argument("evaluation", help="the evaluation's id")
        change.set_defaults(handler=cmd_publication)

    promote_cmd = commands.add_parser("promote", help="rank a result recorded without a suite on a suite")
    server(promote_cmd)
    promote_cmd.add_argument("evaluation", help="the id of the evaluation (or test result) to copy")
    promote_cmd.add_argument("--suite", required=True, help="slug@version")
    checkpoint_args(promote_cmd)
    promote_cmd.set_defaults(handler=cmd_promote)

    version = commands.add_parser("version", help="print the version")
    version.set_defaults(handler=cmd_version)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(message)s")
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
