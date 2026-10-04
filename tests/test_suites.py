"""Suites: reading a file, and publishing it so that the same protocol is harmless to push twice."""

import json
import sys

import pytest
from fakes import ok, refused

import visin
from visin.errors import ApiError, ConfigurationError
from visin.suites import parse_suite_ref

PROTOCOL = {
    "task": "segmentation",
    "data": {"kind": "external", "label": "Road frames", "manifestSha256": "a" * 64},
    "split": "test",
    "conditions": [{"name": "day", "sampleCount": 10}],
    "metrics": [{"key": "mIoU", "direction": "max", "headline": True}],
    "aggregation": "equal-mean-of-conditions",
    "evaluator": {"package": "visin-fusion"},
}
SUITE = {"slug": "road-test", "version": 1, "name": "Road test", "protocol": PROTOCOL}


def published(**overrides):
    return {
        "_id": "s1",
        "slug": "road-test",
        "version": 1,
        "name": "Road test",
        "projectId": "p1",
        "visibility": "private",
        "digest": "d" * 64,
        "protocol": PROTOCOL,
        **overrides,
    }


class TestParsing:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("road-test@1", ("road-test", "1")),
            (" road-test@12 ", ("road-test", "12")),
            ("road-test", ("road-test", "latest")),
            ("road-test@latest", ("road-test", "latest")),
            ("a", ("a", "latest")),
            ("a" * 64, ("a" * 64, "latest")),
        ],
    )
    def test_a_suite_is_named_slug_at_version_and_a_bare_slug_means_the_latest(self, text, expected):
        assert parse_suite_ref(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "Road Test@1",
            "road-test@0",
            "@1",
            "road-test@one",
            "a@1@2",
            "road-@1",
            "a" * 65,
            "road-test@1234567",
        ],
    )
    def test_anything_else_is_refused_with_the_form_to_use(self, text):
        with pytest.raises(ValueError, match="slug@version"):
            parse_suite_ref(text)


class TestManifest:
    def test_the_digest_does_not_depend_on_the_order_of_samples_or_conditions(self):
        one = visin.manifest_digest({"day": ["b.png", "a.png"], "night": ["c.png"]})
        assert one == visin.manifest_digest({"night": ["c.png"], "day": ["a.png", "b.png"]})
        assert len(one) == 64 and set(one) <= set("0123456789abcdef")

    @pytest.mark.parametrize(
        "changed",
        [
            {"day": ["a.png", "b.png", "x.png"], "night": ["c.png"]},
            {"day": ["a.png"], "night": ["c.png"]},
            {"day": ["a.png", "b.png"], "night": ["d.png"]},
            {"day": ["a.png", "b.png"], "dusk": ["c.png"]},
            {"day": ["a.png", "b.png", "b.png"], "night": ["c.png"]},
        ],
    )
    def test_any_sample_added_removed_renamed_or_moved_to_another_condition_changes_it(self, changed):
        assert visin.manifest_digest(changed) != visin.manifest_digest(
            {"day": ["a.png", "b.png"], "night": ["c.png"]}
        )

    def test_a_sample_in_another_condition_is_not_the_same_data(self):
        assert visin.manifest_digest({"day": ["a.png"], "night": ["b.png"]}) != visin.manifest_digest(
            {"day": ["b.png"], "night": ["a.png"]}
        )

    def test_a_split_file_is_its_lines_without_blanks_or_padding(self, tmp_path):
        path = tmp_path / "split.txt"
        path.write_text("camera/a.png\n\n  camera/b.png  \n\n")
        assert visin.read_split(path) == ["camera/a.png", "camera/b.png"]

    def test_a_split_file_that_cannot_be_read_is_named(self, tmp_path):
        with pytest.raises(ConfigurationError, match="cannot read the split file"):
            visin.read_split(tmp_path / "missing.txt")


class TestLoading:
    def test_reads_json(self, tmp_path):
        path = tmp_path / "suite.json"
        path.write_text(json.dumps(SUITE))
        assert visin.load_suite(path) == SUITE

    def test_reads_yaml_when_pyyaml_is_there(self, tmp_path):
        pytest.importorskip("yaml")
        path = tmp_path / "suite.yml"
        path.write_text("slug: road-test\nversion: 1\nname: Road test\nprotocol:\n  task: segmentation\n")
        assert visin.load_suite(path)["protocol"] == {"task": "segmentation"}

    def test_says_how_to_get_yaml_support_when_it_is_missing(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "yaml", None)
        path = tmp_path / "suite.yaml"
        path.write_text("slug: road-test")
        with pytest.raises(ConfigurationError, match=r"pip install 'visin\[yaml\]'.*JSON"):
            visin.load_suite(path)

    def test_names_the_file_when_it_cannot_be_read_or_is_not_json_or_not_a_suite(self, tmp_path):
        with pytest.raises(ConfigurationError, match="cannot read the suite file"):
            visin.load_suite(tmp_path / "missing.json")
        broken = tmp_path / "broken.json"
        broken.write_text("{ nope")
        with pytest.raises(ConfigurationError, match=r"broken\.json is not valid JSON"):
            visin.load_suite(broken)
        listing = tmp_path / "list.json"
        listing.write_text("[1]")
        with pytest.raises(ConfigurationError, match="should hold one suite"):
            visin.load_suite(listing)


class TestPushing:
    def test_publishes_the_file_into_the_project_and_returns_the_suite(self, server, session, tmp_path):
        session.route("POST", "/suites", ok(published(), 201))
        path = tmp_path / "suite.json"
        path.write_text(json.dumps({**SUITE, "project": "road-seg"}))
        suite = visin.push_suite(path)
        body = session.bodies("/suites")[0]
        assert body["projectId"] == "road-seg" and "project" not in body and body["protocol"] == PROTOCOL
        assert (suite.ref, suite.digest, suite.visibility, suite.project_id) == (
            "road-test@1",
            "d" * 64,
            "private",
            "p1",
        )
        assert suite.headline == {"key": "mIoU", "direction": "max", "headline": True}

    def test_takes_a_dict_and_the_argument_beats_the_file_beats_the_environment(
        self, server, session, monkeypatch
    ):
        session.route("POST", "/suites", ok(published(), 201))
        monkeypatch.setenv("VISIN_PROJECT", "from-env")
        visin.push_suite({**SUITE, "project": "from-file"}, project="from-arg")
        visin.push_suite({**SUITE, "project": "from-file"})
        visin.push_suite(SUITE)
        assert [b["projectId"] for b in session.bodies("/suites")] == ["from-arg", "from-file", "from-env"]

    def test_a_projectid_in_the_file_is_honoured_too(self, server, session):
        session.route("POST", "/suites", ok(published(), 201))
        visin.push_suite({**SUITE, "projectId": "by-id"})
        assert session.bodies("/suites")[0]["projectId"] == "by-id"

    def test_a_pipeline_key_supplies_its_project_and_visibility_can_be_set(self, server, session):
        session.route("GET", "/.well-known/visin", ok({"credential": {"project": {"id": "p7"}}}))
        session.route("POST", "/suites", ok(published(visibility="public"), 201))
        suite = visin.push_suite(SUITE, visibility="public")
        body = session.bodies("/suites")[0]
        assert body["projectId"] == "p7" and body["visibility"] == "public" and suite.visibility == "public"

    def test_with_no_project_anywhere_says_how_to_give_one(self, server, session):
        session.route("GET", "/.well-known/visin", refused(404))
        with pytest.raises(ConfigurationError, match="needs a project"):
            visin.push_suite(SUITE)
        assert session.bodies("/suites") == []

    def test_the_same_protocol_again_is_the_stored_suite(self, server, session):
        session.route("POST", "/suites", ok(published(), 201), ok(published(), 200))
        first = visin.push_suite(SUITE, project="road-seg")
        again = visin.push_suite(SUITE, project="road-seg")
        assert first == again

    def test_a_different_protocol_under_a_taken_version_is_refused_with_what_to_do(self, server, session):
        session.route(
            "POST",
            "/suites",
            refused(
                409, "road-test version 1 already exists with a different protocol: publish it as version 2."
            ),
        )
        with pytest.raises(ApiError, match="publish it as version 2"):
            visin.push_suite(SUITE, project="road-seg")

    def test_needs_a_server(self):
        with pytest.raises(ConfigurationError, match="publish a suite"):
            visin.push_suite(SUITE, project="road-seg")
