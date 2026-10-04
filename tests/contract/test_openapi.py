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
def recorded(server, session, uploads, tmp_path, monkeypatch, hf):
    """Drive every public entry point once and return what was sent."""
    session.route(
        "POST",
        "/trainings",
        ok({"_id": "0123456789abcdef01234567", "uuid": "u", "projectId": "0123456789abcdef01234569"}, 201),
    )
    session.route("POST", "/configs/upload", ok({"_id": "0123456789abcdef01234568"}, 201))
    session.route("POST", "/suites/check", *[ok({"digest": "d" * 64, "protocol": {}}) for _ in range(8)])
    session.route(
        "POST",
        "/visualizations/upload-url",
        ok({"uploadUrl": "https://files.example.test/put", "visualization_uuid": "v1", "fileId": "f1"}),
    )
    frame = tmp_path / "overlay.png"
    frame.write_bytes(b"\x89PNG")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"weights")

    monkeypatch.setenv("VISIN_PROVENANCE", "1")
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
        run.log_model(checkpoint, "acme/clft", epoch=1)
        run.update(tags=["done"], name="renamed", description="d", metadata={"best": 1})

    suite_file = {
        "slug": "road-test",
        "version": 1,
        "name": "Road test",
        "description": "every field",
        "visibility": "public",
        "project": "road-seg",
        "protocol": {
            "task": "semantic-segmentation",
            "data": {"kind": "external", "label": "Road frames", "manifestSha256": "a" * 64},
            "split": "test",
            "annotationVersion": "2026-09",
            "conditions": [{"name": "day", "sampleCount": 10}],
            "classes": [{"id": "vehicle", "name": "Vehicle"}],
            "ignoredClasses": ["void"],
            "metrics": [
                {
                    "key": "mIoU",
                    "direction": "max",
                    "unit": "ratio",
                    "range": {"min": 0, "max": 1},
                    "headline": True,
                }
            ],
            "aggregation": "equal-mean-of-conditions",
            "input": {"sensors": ["camera"], "resolution": "1280x720"},
            "evaluator": {"package": "visin-fusion", "minVersion": "1.0.0"},
        },
    }
    visin.push_suite(suite_file)
    visin.check_protocol(suite_file)
    scored = visin.local_checkpoint(checkpoint, "contract-checkpoint")
    for kwargs in (
        {"dry_run": True},
        {
            "uuid": "contract-evaluation",
            "run": "run-uuid",
            "epoch": 1,
            "epoch_uuid": "epoch-uuid",
            "executed_at": "2026-09-30T08:00:00Z",
            "data": {"kind": "external", "manifestSha256": "a" * 64},
            "protocol_digest": "d" * 64,
            "classes": {"scored": ["car"], "ignored": ["void"]},
            "supersedes": "0123456789abcdef01234567",
            "evaluator": {"package": "visin-fusion", "version": "1.4.2", "commit": "9d1c2ab"},
            "provenance": {"seeds": [1]},
        },
        {"protocol": suite_file, "data": {"kind": "hf", "repo": "acme/frames", "commit": "c" * 40}},
    ):
        visin.evaluate(
            {"day": {"overall": {"mIoU": 0.7}}},
            suite="road-test@1",
            checkpoint=scored,
            project="road-seg",
            sample_counts={"day": 10},
            **kwargs,
        )
    visin.evaluate(
        {},
        suite=None,
        checkpoint=visin.hub_checkpoint("acme/clft", "a" * 40, "best.safetensors"),
        project="road-seg",
        status="failed",
    )
    visin.promote(
        "0123456789abcdef01234567", suite="road-test@1", checkpoint=scored, sample_counts={"day": 10}
    )
    visin.publish("0123456789abcdef01234567")
    visin.withdraw("0123456789abcdef01234567")

    monkeypatch.setenv("VISIN_MODE", "offline")
    visin.init("offline run", project="road-seg").finish()
    monkeypatch.delenv("VISIN_MODE")
    visin.sync()

    api = visin.Api()
    api.check_protocol(suite_file)
    list(
        api.trainings(project="road-seg", status="completed", tags=["x"], search="s", dataset="zod", limit=1)
    )
    run = visin.Training(id="0123456789abcdef01234567", uuid="u", name="n")
    api.training("some-uuid")
    api.epochs(run)
    api.test_results(run)
    api.benchmarks(run, project="p")
    api.projects()
    api.project("road-seg")
    api.find("n", project="road-seg")
    api.tags()
    api.config(run)
    api.summary(run)
    list(api.comparisons(project="road-seg", type="trainings", limit=1))
    list(api.findings(project="road-seg", limit=1))
    api.visualizations(run, kind="overlay")
    api.suites(slug="road-test", project="road-seg", include_archived=True)
    api.suite("road-test@1")
    api.suite("road-test")
    api.evaluations(
        project="road-seg", suite="road-test@1", state="eligible", status="completed", checkpoint_key="k"
    )
    api.evaluation("0123456789abcdef01234567")
    api.evaluation("contract-evaluation", project="road-seg")
    api.leaderboard("road-test@1")
    api.public_leaderboards()
    api.public_leaderboard("road-test@1")
    api.leaderboard("road-test@1", page=2, limit=100, unranked_page=2)
    api.leaderboard("road-test@1", observed=True)
    api.public_leaderboard("road-test@1", observed=True)
    api.leaderboard("road-test@1", all_pages=True)
    api.public_leaderboards(page=2, limit=100)
    list(api.iter_public_leaderboards())
    api.public_leaderboard("road-test@1", page=2, limit=100)
    api.public_leaderboard("road-test@1", all_pages=True)

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
        validator = jsonschema.Draft202012Validator(schema)
        problems.extend(
            f"{call['method']} {template}: {error.message}" for error in validator.iter_errors(call["json"])
        )
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
        "POST /benchmarks/upload",
        "POST /configs/upload",
        "POST /visualizations/upload-url",
        "POST /visualizations",
        "POST /trainings/{id}/models",
        "POST /suites",
        "POST /evaluations",
        "POST /evaluations/check",
        "POST /evaluations/promote",
        "POST /suites/check",
    ):
        assert operation in checked, f"{operation} was never exercised"


def test_leaderboard_paging_uses_the_query_parameters_the_api_declares(spec, recorded):
    checked = set()
    problems = []
    for call in recorded:
        path = call["url"].split("/api", 1)[1]
        if "leaderboard" not in path or not call["params"]:
            continue
        template, operation = find_operation(spec, call["method"], path)
        declared = {
            item["name"]: resolve(item.get("schema") or {}, spec)
            for item in operation.get("parameters", [])
            if item.get("in") == "query"
        }
        checked.add(template)
        for name, value in call["params"].items():
            if name not in declared:
                problems.append(f"{template}: undeclared query parameter {name}")
            else:
                problems.extend(
                    f"{template}?{name}={value}: {error.message}"
                    for error in jsonschema.Draft202012Validator(declared[name]).iter_errors(value)
                )
    assert not problems, "\n".join(problems)
    assert checked == {
        "/suites/{slug}/{version}/leaderboard",
        "/public/leaderboards",
        "/public/leaderboards/{slug}/{version}",
    }


def test_config_upload_inherits_the_training_project(recorded):
    configs = [call["json"] for call in recorded if call["url"].endswith("/configs/upload")]
    assert configs and all(body["projectId"] == "0123456789abcdef01234569" for body in configs)


def test_collected_provenance_stays_within_the_platform_request_limits(spec, monkeypatch, tmp_path):
    import sys
    import types

    from visin._internal import provenance

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["train.py", "--token", "secret-value", "x" * 5000])
    git = {
        ("rev-parse", "HEAD"): "a" * 40,
        ("rev-parse", "--abbrev-ref", "HEAD"): "b" * 300,
        ("status", "--porcelain"): "",
        ("config", "--get", "remote.origin.url"): "https://token@host/" + "r" * 600,
    }
    monkeypatch.setattr(provenance, "_git", lambda directory, *args: git.get(args))
    monkeypatch.setattr(provenance.socket, "gethostname", lambda: "h" * 300)
    monkeypatch.setattr(provenance.platform, "platform", lambda: "p" * 300)
    monkeypatch.setattr(provenance.platform, "python_version", lambda: "v" * 110)
    distributions = [
        types.SimpleNamespace(metadata={"Name": name}, version=version)
        for name, version in [("requests", "2.0"), ("n" * 101, "1"), ("custom", "v" * 101)]
    ]
    monkeypatch.setattr(provenance.metadata, "distributions", lambda: distributions)
    collected = provenance.collect()
    assert "secret-value" not in collected["command"]
    assert "token@" not in collected["git"]["remote"]
    assert collected["packages"] == {"requests": "2.0"}
    _, operation = find_operation(spec, "POST", "/trainings")
    jsonschema.Draft202012Validator(body_schema(operation, spec)).validate(
        {"name": "Run", "provenance": collected}
    )
