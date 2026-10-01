"""The ``visin`` command.

visin check            can this machine reach Visin, and is the token good?
visin check --write    ...and may it create a run? (makes one, then deletes it)
visin sync             send reports kept on disk by offline or cut-off runs
visin sync --list      show what is waiting, send nothing
visin runs             list recent runs
visin datasets         list the datasets on Visin
visin download zod     download a dataset (once) and print its folder
visin version
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid as uuidlib
from collections.abc import Sequence
from typing import Any
from urllib.parse import quote

from ._internal.config import read_settings
from ._internal.spool import Spool, pending_runs
from ._internal.transport import HttpClient
from ._version import __version__
from .errors import ApiError, VisinError
from .offline import sync
from .run import epoch_uuid_for


def _credential_kind(token: str | None) -> str:
    if not token:
        return "none"
    if token.startswith("vsn_live_"):
        return "API key"
    if token.count(".") == 2:
        return "JWT"
    return "unrecognised token"


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
    print(f"  url      {settings.url or '(VISIN_URL is not set)'}")
    print(f"  token    {_mask(settings.token)} ({_credential_kind(settings.token)})")
    print(f"  project  {settings.project or '-'}")
    print(f"  mode     {settings.effective_mode}")
    print(f"  kept in  {settings.directory}")
    print()
    if not settings.url:
        out.fail("VISIN_URL is not set, so nothing will be reported")
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
        out.ok(f"the {_credential_kind(token)} is accepted")
    except ApiError as exc:
        out.fail(f"the token was refused: {exc}")
        return 1
    except VisinError as exc:
        out.fail(str(exc))
        return 1

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
    elif _credential_kind(token) == "API key":
        out.info("runs go to the key's project if it is a pipeline key; otherwise set VISIN_PROJECT")
    if _credential_kind(token) == "unrecognised token":
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


def cmd_sync(args: argparse.Namespace) -> int:
    settings = read_settings(url=args.url, token=args.token, directory=args.dir)
    uuids = [args.run] if args.run else pending_runs(settings.directory)
    if not uuids:
        print(f"nothing waiting in {settings.directory}")
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


def cmd_version(_args: argparse.Namespace) -> int:
    print(f"visin {__version__}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="visin", description="Report training runs to Visin.")
    parser.add_argument("-v", "--verbose", action="store_true", help="log what the client does")
    commands = parser.add_subparsers(dest="command", metavar="command")
    commands.required = True

    def server(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--url", help="the Visin API address (default: VISIN_URL)")
        sub.add_argument("--token", help="the credential (default: VISIN_TOKEN)")

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
    sync_cmd.set_defaults(handler=cmd_sync)

    runs = commands.add_parser("runs", help="list recent runs")
    server(runs)
    runs.add_argument("--project", help="only this project's runs (its id or slug)")
    runs.add_argument("--status", choices=["pending", "running", "completed", "failed"])
    runs.add_argument("--limit", type=int, default=20)
    runs.set_defaults(handler=cmd_runs)

    def dataset_server(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--url", help="the Visin dataset service (default: VISIN_DATASET_URL)")
        sub.add_argument("--token", help="the credential, for private datasets (default: VISIN_TOKEN)")

    datasets = commands.add_parser("datasets", help="list the datasets on Visin")
    dataset_server(datasets)
    datasets.add_argument("--search", help="only datasets whose name matches")
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
