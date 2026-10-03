"""Dataset reads use the published dataset-service contract."""

import json
import os
from pathlib import Path

import pytest
from fakes import ok

from visin import Datasets

from .test_openapi import find_operation, resolve

jsonschema = pytest.importorskip("jsonschema")


DEFAULT_DATASET_SPEC = (
    Path(__file__).resolve().parents[3] / "visin" / "apps/frontend/landing-front/public/openapi/dataset.json"
)


def test_dataset_reads_follow_the_service_contract(client, session):
    path = DEFAULT_DATASET_SPEC
    path = Path(os.environ.get("VISIN_DATASET_OPENAPI") or path)
    if not path.exists():
        if os.environ.get("VISIN_REQUIRE_CONTRACT"):
            pytest.fail(f"no dataset API spec at {path}")
        pytest.skip(f"no dataset API spec at {path}; set VISIN_DATASET_OPENAPI")
    spec = json.loads(path.read_text())
    ident = "0123456789abcdef01234567"
    item = {
        "_id": ident,
        "name": "Dataset",
        "owner": {"kind": "user", "id": ident},
        "visibility": "public",
        "archive": {"size": 1},
    }
    listed = {"datasets": [item], "pagination": {"page": 1, "limit": 1000, "total": 1, "pages": 1}}
    session.route("GET", "/datasets", ok(listed))
    session.route("GET", f"/datasets/{ident}", ok(item))
    datasets = Datasets(client=client)
    assert datasets.list()[0].id == ident
    assert datasets.get(ident).name == "Dataset"
    download = {
        "downloadUrl": "https://files.example.test/data.zip",
        "filename": "data.zip",
        "size": 1,
        "revision": "2026-10-02T10:00:00.000Z",
        "expiresAt": "2026-10-02T12:00:00.000Z",
    }
    hub = {"provider": "hf", "repo": "acme/zod-png", "revision": "3f2a1c9d8e7b6a5f4e3d2c1b0a99887766554433"}
    on_hub = {"source": hub, "revision": hub["revision"]}
    both = {
        **download,
        "source": hub,
        "revision": hub["revision"],
        "archiveRevision": "2026-10-02T10:00:00.000Z",
    }
    hosted = {**item, "source": hub}
    session.route("GET", f"/datasets/{ident}/download", ok(download), ok(on_hub), ok(both))
    session.route("GET", f"/datasets/{ident}", ok(hosted))
    for _ in range(3):
        datasets._read(f"/datasets/{ident}/download")
    datasets._read(f"/datasets/{ident}")
    responses = [listed, item, download, on_hub, both, hosted]
    for call, response in zip(session.calls, responses, strict=True):
        template, operation = find_operation(spec, call["method"], call["url"].split("/api", 1)[1])
        assert template is not None
        for parameter in operation.get("parameters", []):
            if parameter["in"] == "query" and parameter["name"] in (call["params"] or {}):
                jsonschema.Draft202012Validator(resolve(parameter["schema"], spec)).validate(
                    call["params"][parameter["name"]]
                )

        schema = resolve(operation["responses"]["200"]["content"]["application/json"]["schema"], spec)
        jsonschema.Draft202012Validator(schema).validate({"success": True, "data": response})


def test_publishing_a_dataset_sends_what_the_service_accepts(client, session, hf, monkeypatch, tmp_path):
    path = Path(os.environ.get("VISIN_DATASET_OPENAPI") or DEFAULT_DATASET_SPEC)
    if not path.exists():
        if os.environ.get("VISIN_REQUIRE_CONTRACT"):
            pytest.fail(f"no dataset API spec at {path}")
        pytest.skip(f"no dataset API spec at {path}; set VISIN_DATASET_OPENAPI")
    spec = json.loads(path.read_text())
    ident = "0123456789abcdef01234567"
    archive = {"filename": "d.zip", "size": 1}
    session.route("GET", "/datasets", ok({"datasets": [{"_id": ident, "name": "D", "archive": archive}]}))
    session.route(
        "GET",
        f"/datasets/{ident}/download",
        ok({"downloadUrl": "https://f.example.test/d.zip", "revision": "r"}),
    )
    monkeypatch.setattr(Datasets, "download", lambda self, *_a, **_k: tmp_path)
    monkeypatch.setenv("VISIN_DATASET_URL", "https://visin.example.test")
    Datasets(client=client).push("d", "acme/d")
    (call,) = [c for c in session.calls if c["method"] == "PATCH"]
    template, operation = find_operation(spec, "PATCH", call["url"].split("/api", 1)[1])
    assert template == "/datasets/{id}"
    schema = resolve(operation["requestBody"]["content"]["application/json"]["schema"], spec)
    jsonschema.Draft202012Validator(schema).validate(call["json"])
    known = next(
        branch["properties"] for branch in schema["properties"]["source"]["anyOf"] if "properties" in branch
    )
    assert set(call["json"]["source"]) <= set(known)
