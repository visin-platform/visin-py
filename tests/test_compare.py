"""The regression gate: two evaluations on one suite version, score by score, in each metric's direction."""

import json

import pytest
from fakes import ok, refused

import visin
from visin.cli import EXIT_NOT_COMPARABLE, EXIT_REGRESSION, main

PROTOCOL = {
    "conditions": [{"name": "day"}, {"name": "night"}],
    "metrics": [
        {"key": "mIoU", "direction": "max", "headline": True},
        {"key": "latency", "direction": "min"},
    ],
}
SUITE = visin.Suite(id="s1", slug="road-test", version=1, name="Road", digest="d" * 64, protocol=PROTOCOL)


def scores(day=0.8, night=0.6, overall=0.7, latency=(10.0, 12.0, 11.0)):
    return {
        "conditions": {
            "day": {"mIoU": day, "latency": latency[0]},
            "night": {"mIoU": night, "latency": latency[1]},
        },
        "overall": {"mIoU": overall, "latency": latency[2]},
    }


def evaluation(ident="e1", state="eligible", result=None, suite="road-test@1", digest="d" * 64):
    return visin.Evaluation(
        id=ident,
        suite=suite,
        suite_digest=digest,
        verdict=visin.models.Verdict(state=state, scores=result if state == "eligible" else None),
    )


class TestDiff:
    def test_a_candidate_that_is_no_worse_passes_and_reads_each_score_in_its_direction(self):
        better = scores(day=0.82, night=0.6, overall=0.71)
        diff = visin.diff_evaluations(evaluation(result=scores()), evaluation("e2", result=better), SUITE)
        assert diff.passed and diff.regressions == () and diff.missing == ()
        statuses = {change.scope: change.status for change in diff.changes}
        assert statuses == {"overall": "improved", "day": "improved", "night": "ok"}
        assert all(change.metric == "mIoU" for change in diff.changes)

    def test_a_drop_past_the_allowed_amount_is_a_regression_and_a_smaller_one_is_not(self):
        worse = scores(day=0.78, night=0.6, overall=0.69)
        assert [
            c.scope
            for c in visin.diff_evaluations(
                evaluation(result=scores()), evaluation("e2", result=worse), SUITE
            ).regressions
        ] == ["overall", "day"]
        allowed = visin.diff_evaluations(
            evaluation(result=scores()), evaluation("e2", result=worse), SUITE, max_drop=0.02
        )
        assert allowed.passed
        edge = visin.diff_evaluations(
            evaluation(result=scores()),
            evaluation("e2", result=scores(day=0.79, overall=0.7)),
            SUITE,
            max_drop=0.01,
        )
        assert edge.passed

    def test_a_metric_where_lower_is_better_regresses_when_it_rises(self):
        slower = scores(latency=(10.0, 12.0, 11.5))
        diff = visin.diff_evaluations(
            evaluation(result=scores()), evaluation("e2", result=slower), SUITE, all_metrics=True
        )
        assert [(c.scope, c.metric) for c in diff.regressions] == [("overall", "latency")]
        assert diff.regressions[0].improvement == pytest.approx(-0.5)
        faster = scores(latency=(9.0, 11.0, 10.0))
        assert visin.diff_evaluations(
            evaluation(result=scores()), evaluation("e2", result=faster), SUITE, all_metrics=True
        ).passed

    def test_only_the_headline_is_compared_unless_every_metric_is_asked_for(self):
        slower = scores(latency=(99.0, 99.0, 99.0))
        diff = visin.diff_evaluations(evaluation(result=scores()), evaluation("e2", result=slower), SUITE)
        assert diff.passed and {c.metric for c in diff.changes} == {"mIoU"}
        assert {
            c.metric
            for c in visin.diff_evaluations(
                evaluation(result=scores()), evaluation("e2", result=slower), SUITE, all_metrics=True
            ).changes
        } == {"mIoU", "latency"}

    def test_a_score_the_candidate_lacks_is_missing_coverage_never_a_zero(self):
        thin = scores()
        del thin["conditions"]["night"]
        diff = visin.diff_evaluations(evaluation(result=scores()), evaluation("e2", result=thin), SUITE)
        assert not diff.passed and [(c.scope, c.status) for c in diff.missing] == [("night", "missing")]
        assert diff.regressions == ()

    def test_a_valid_zero_is_a_score_and_not_a_gap(self):
        diff = visin.diff_evaluations(
            evaluation(result=scores(night=0.0)), evaluation("e2", result=scores(night=0.0)), SUITE
        )
        assert diff.passed and next(c for c in diff.changes if c.scope == "night").status == "ok"

    def test_an_evaluation_that_is_not_ranked_makes_everything_missing_and_says_so(self):
        diff = visin.diff_evaluations(
            evaluation(result=scores()), evaluation("e2", state="incomplete"), SUITE
        )
        assert (
            not diff.passed and len(diff.missing) == 3 and "candidate (e2) is not ranked" in diff.problems[0]
        )

    def test_a_different_suite_version_or_protocol_cannot_be_compared(self):
        with pytest.raises(ValueError, match="only one suite version"):
            visin.diff_evaluations(
                evaluation(result=scores()), evaluation("e2", result=scores(), suite="road-test@2"), SUITE
            )
        with pytest.raises(ValueError, match="no suite"):
            visin.diff_evaluations(
                evaluation(result=scores()), evaluation("e2", result=scores(), suite=None), SUITE
            )
        with pytest.raises(ValueError, match="different protocol"):
            visin.diff_evaluations(
                evaluation(result=scores()), evaluation("e2", result=scores(), digest="e" * 64), SUITE
            )

    def test_a_negative_allowance_is_refused(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            visin.diff_evaluations(
                evaluation(result=scores()), evaluation("e2", result=scores()), SUITE, max_drop=-1
            )


def stored(ident, result=None, state="eligible"):
    return {
        "_id": ident,
        "uuid": f"u-{ident}",
        "suite": {"slug": "road-test", "version": 1, "digest": "d" * 64},
        "status": "completed",
        "validation": {
            "state": state,
            "reasons": [],
            "warnings": [],
            **({"scores": result} if result else {}),
        },
    }


def suite_json():
    return {
        "_id": "s1",
        "slug": "road-test",
        "version": 1,
        "name": "Road",
        "digest": "d" * 64,
        "protocol": PROTOCOL,
    }


class TestDiffCommand:
    def route(self, session, baseline, candidate):
        session.route("GET", "/evaluations/b1", ok(baseline))
        session.route("GET", "/evaluations/c1", ok(candidate))
        session.route("GET", "/suites/road-test/1", ok(suite_json()))

    def test_exits_zero_when_nothing_got_worse(self, server, session, capsys):
        self.route(session, stored("b1", scores()), stored("c1", scores(day=0.9, overall=0.75)))
        code = main(["diff", "b1", "c1", "--max-drop", "0.01"])
        out = capsys.readouterr().out
        assert code == 0 and "road-test@1" in out and "improved" in out and out.rstrip().endswith("ok")

    def test_exits_four_for_a_regression_and_names_the_score(self, server, session, capsys):
        self.route(session, stored("b1", scores()), stored("c1", scores(day=0.7, overall=0.65)))
        assert main(["diff", "b1", "c1", "--max-drop", "0.01"]) == EXIT_REGRESSION == 4
        out = capsys.readouterr().out
        assert "regressed" in out and "day" in out and "FAILED" in out

    def test_exits_five_when_a_score_cannot_be_compared_even_with_a_regression_beside_it(
        self, server, session, capsys
    ):
        thin = scores(day=0.1)
        del thin["conditions"]["night"]
        self.route(session, stored("b1", scores()), stored("c1", thin))
        assert main(["diff", "b1", "c1"]) == EXIT_NOT_COMPARABLE == 5
        assert "not comparable" in capsys.readouterr().out

    def test_exits_five_for_an_evaluation_that_is_not_ranked(self, server, session, capsys):
        self.route(session, stored("b1", scores()), stored("c1", state="incomplete"))
        assert main(["diff", "b1", "c1"]) == 5
        assert "is not ranked" in capsys.readouterr().out

    def test_prints_json_for_scripts(self, server, session, capsys):
        self.route(session, stored("b1", scores()), stored("c1", scores(day=0.7)))
        main(["diff", "b1", "c1", "--json", "--all-metrics"])
        data = json.loads(capsys.readouterr().out)
        assert data["passed"] is False and data["suite"] == "road-test@1" and data["maxDrop"] == 0
        assert {change["metric"] for change in data["changes"]} == {"mIoU", "latency"}

    def test_names_evaluations_by_uuid_with_a_project(self, server, session):
        session.route("GET", "/evaluations/uuid/mine-1", ok(stored("b1", scores())))
        session.route("GET", "/evaluations/uuid/mine-2", ok(stored("c1", scores())))
        session.route("GET", "/suites/road-test/1", ok(suite_json()))
        assert main(["diff", "mine-1", "mine-2", "--project", "road-seg"]) == 0
        assert session.calls[0]["params"] == {"projectId": "road-seg"}

    def test_explains_a_refusal_and_a_comparison_that_makes_no_sense(self, server, session, capsys):
        session.route("GET", "/evaluations/b1", refused(404, "Evaluation not found"))
        assert main(["diff", "b1", "c1"]) == 1
        assert "Evaluation not found" in capsys.readouterr().err
        other = stored("c1", scores())
        other["suite"]["version"] = 2
        self.route(session, stored("b1", scores()), other)
        assert main(["diff", "b1", "c1"]) == 1
        assert "only one suite version" in capsys.readouterr().err
        unscored = stored("b1", scores())
        del unscored["suite"]
        self.route(session, unscored, stored("c1", scores()))
        assert main(["diff", "b1", "c1"]) == 1
        assert "names no suite" in capsys.readouterr().err
