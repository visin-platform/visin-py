"""Every request this package sends, checked against Visin's published API spec.

The spec is ``landing-front/public/openapi/vision.json`` in the Visin repository:
the bundled copy Visin publishes at ``/docs/api``. Point ``VISIN_OPENAPI`` at a
copy (CI downloads the one on Visin's main branch); without one, next to this
checkout at ``../visin``, the test is skipped. ``VISIN_REQUIRE_CONTRACT=1``
makes a missing spec a failure instead.

Two kinds of drift are caught:

* a request whose method and path the API no longer has, or whose body no
  longer satisfies the endpoint's schema;
* a body field the endpoint does not define. The server strips those without a
  word, which is how the scripts this package replaced lost every run's model
  name while reporting success.
"""

import json
import os
import re
from pathlib import Path

import pytest
from fakes import ok

import visin
from visin.cli import main

jsonschema = pytest.importorskip("jsonschema")

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = (
    REPO.parent / "visin" / "apps" / "frontend" / "landing-front" / "public" / "openapi" / "vision.json"
)


def load_spec():
    path = Path(os.environ.get("VISIN_OPENAPI") or DEFAULT_SPEC)
    if not path.exists():
        if os.environ.get("VISIN_REQUIRE_CONTRACT"):
            pytest.fail(f"no API spec at {path}")
        pytest.skip(f"no API spec at {path}; set VISIN_OPENAPI")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def spec():
    return load_spec()


def resolve(node, spec, seen=()):
    """Inline local ``$ref``s, so each schema stands alone for the validator."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/"):
            if ref in seen:
                return {}
            target = spec
            for part in ref[2:].split("/"):
                target = target[part]
            return resolve(target, spec, (*seen, ref))
        return {key: resolve(value, spec, seen) for key, value in node.items()}
    if isinstance(node, list):
        return [resolve(item, spec, seen) for item in node]
    return node


def find_operation(spec, method, path):
    for template, operations in spec["paths"].items():
        pattern = "^" + re.sub(r"\{[^/]+\}", "[^/]+", template) + "$"
        if re.match(pattern, path) and method.lower() in operations:
            return template, operations[method.lower()]
    return None, None


def body_schema(operation, spec):
    content = (operation.get("requestBody") or {}).get("content", {}).get("application/json")
    return resolve(content["schema"], spec) if content else None


@pytest.fixture
def recorded(server, session, uploads, tmp_path, monkeypatch):
    """Drive every public entry point once and return what was sent."""
    session.route(
        "POST",
        "/trainings",
        ok({"_id": "0123456789abcdef01234567", "uuid": "u", "projectId": "0123456789abcdef01234569"}, 201),
    )
    session.route("POST", "/configs/upload", ok({"_id": "0123456789abcdef01234568"}, 201))
    session.route(
        "POST",
        "/visualizations/upload-url",
        ok({"uploadUrl": "https://files.example.test/put", "visualization_uuid": "v1", "fileId": "f1"}),
    )
    frame = tmp_path / "overlay.png"
    frame.write_bytes(b"\x89PNG")

    with visin.init(
        "contract run",
        project="road-seg",
        dataset="zod",
        model="clft",
        config={"lr": 1e-4},
        description="every field",
        tags=["contract"],
        metadata={"seed": 1},
    ) as run:
        run.log_epoch(
            1, train={"loss": 1.0}, val={"loss": 1.1}, learning_rate=0.1, epoch_time=3.0, metadata={"a": 1}
        )
        run.log_test_results(1, {"day": {"overall": {"iou": 0.5}}})
        run.log_benchmark({"device": "cpu", "fps": 1.0}, epoch=1)
        run.upload_visualization(1, frame, "overlay", metadata={"sample": 1})
        run.update(tags=["done"], name="renamed", description="d", metadata={"best": 1})

    monkeypatch.setenv("VISIN_MODE", "offline")
    visin.init("offline run", project="road-seg").finish()
    monkeypatch.delenv("VISIN_MODE")
    visin.sync()

    api = visin.Api()
    list(
        api.trainings(project="road-seg", status="completed", tags=["x"], search="s", dataset="zod", limit=1)
    )
    api.training("0123456789abcdef01234567")
    api.training("some-uuid")
    api.epochs({"_id": "0123456789abcdef01234567", "uuid": "u"})
    api.test_results({"_id": "0123456789abcdef01234567", "uuid": "u"})
    api.benchmarks({"_id": "0123456789abcdef01234567", "uuid": "u"}, project="p")
    api.projects()
    api.project("road-seg")

    monkeypatch.setenv("VISIN_TOKEN", "vsn_live_contract")
    main(["check", "--project", "road-seg", "--write"])
    main(["runs"])
    return session.calls


def test_every_request_is_one_the_api_defines(spec, recorded):
    missing = set()
    for call in recorded:
        path = call["url"].split("/api", 1)[1].split("?")[0]
        template, _ = find_operation(spec, call["method"], path)
        if template is None:
            missing.add(f"{call['method']} {path}")
    assert not missing, f"not in the API spec: {sorted(missing)}"


def test_every_body_satisfies_its_schema_and_sends_nothing_the_server_would_strip(spec, recorded):
    problems = []
    checked = set()
    for call in recorded:
        if call["json"] is None:
            continue
        path = call["url"].split("/api", 1)[1]
        template, operation = find_operation(spec, call["method"], path)
        schema = body_schema(operation, spec) if operation else None
        if schema is None:
            continue
        checked.add(f"{call['method']} {template}")
        for error in jsonschema.Draft202012Validator(schema).iter_errors(call["json"]):
            problems.append(f"{call['method']} {template}: {error.message}")
        known = set(schema.get("properties", {}))
        extra = set(call["json"]) - known
        if known and extra:
            problems.append(f"{call['method']} {template}: fields the server strips: {sorted(extra)}")
    assert not problems, "\n".join(problems)
    # The writes the package exists for, all checked.
    for operation in (
        "POST /trainings",
        "PUT /trainings/{id}",
        "POST /epochs/upload",
        "POST /test-results/upload",
        "POST /benchmarks/upload",
        "POST /configs/upload",
        "POST /visualizations/upload-url",
        "POST /visualizations",
    ):
        assert operation in checked, f"{operation} was never exercised"


def test_config_upload_inherits_the_training_project(recorded):
    configs = [call["json"] for call in recorded if call["url"].endswith("/configs/upload")]
    assert configs and all(body["projectId"] == "0123456789abcdef01234569" for body in configs)
