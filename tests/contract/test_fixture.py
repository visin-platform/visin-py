"""The comparison contract Visin's server, app and this SDK all read: the same states, ranks and deltas.

``leaderboard.json`` is a mirror of the fixture in the Visin repository
(``vision-service/src/__tests__/fixtures/leaderboard.json``), whose expected values were worked out by hand.
The mirror must equal the original (checked whenever a checkout sits at ``../visin``), and what this package
reads of it must come out as the contract says: a ranking, a verdict, a valid zero, a tie, and a metric where
lower is better. ``VISIN_REQUIRE_CONTRACT=1`` makes a missing checkout a failure instead of a skip.
"""

import json
import os
from pathlib import Path

import pytest

import visin
from visin.models import Verdict

MIRROR = Path(__file__).with_name("leaderboard.json")
ORIGINAL = (
    Path(__file__).resolve().parents[3]
    / "visin"
    / "apps"
    / "backend"
    / "vision-service"
    / "src"
    / "__tests__"
    / "fixtures"
    / "leaderboard.json"
)
FIXTURE = json.loads(MIRROR.read_text())
SUITES = FIXTURE["suites"]
EVALUATIONS = {item["id"]: item for item in FIXTURE["evaluations"]}
EXPECTED = FIXTURE["expected"]
DIGEST = "d" * 64


def test_the_mirror_is_the_servers_fixture():
    if not ORIGINAL.exists():
        if os.environ.get("VISIN_REQUIRE_CONTRACT"):
            pytest.fail(f"no fixture at {ORIGINAL}")
        pytest.skip(f"no Visin checkout at {ORIGINAL}")
    assert json.loads(ORIGINAL.read_text()) == FIXTURE, (
        "copy the server's fixture over tests/contract/leaderboard.json"
    )


def server_board(name: str) -> dict:
    """The ranking the server answers for a suite, with the fields this package reads."""
    metric = next(item for item in SUITES[name]["metrics"] if item.get("headline"))
    want = EXPECTED["leaderboards"][name]
    return {
        "suite": {
            "slug": name,
            "version": 1,
            "headline": {"key": metric["key"], "direction": metric["direction"]},
        },
        "scope": {"candidates": len(want["entries"]) + len(want["unranked"])},
        "entries": [
            {
                "evaluationId": row["id"],
                "rank": row["rank"],
                "attempts": row["attempts"],
                "checkpoint": {"kind": "local", "label": row["id"]},
                "summary": {"headline": {"value": row["headline"]}, "worst": row["worst"], "gap": row["gap"]},
            }
            for row in want["entries"]
        ],
        "unranked": [
            {"evaluationId": row["id"], "checkpointKey": row["checkpoint"], "state": row["state"]}
            for row in want["unranked"]
        ],
    }


@pytest.mark.parametrize("name", sorted(EXPECTED["leaderboards"]))
def test_a_ranking_reads_as_the_contract_says(name):
    board = visin.Leaderboard.from_json(server_board(name))
    want = EXPECTED["leaderboards"][name]
    assert [(e.evaluation_id, e.rank, e.attempts) for e in board.entries] == [
        (row["id"], row["rank"], row["attempts"]) for row in want["entries"]
    ]
    for entry, row in zip(board.entries, want["entries"], strict=True):
        assert entry.headline == pytest.approx(row["headline"], abs=1e-9)
        assert (entry.worst_condition, entry.worst_value) == (
            row["worst"]["condition"],
            row["worst"]["value"],
        )
        assert entry.gap == pytest.approx(row["gap"], abs=1e-9)
    assert [(row["evaluationId"], row["state"]) for row in board.unranked] == [
        (row["id"], row["state"]) for row in want["unranked"]
    ]
    assert board.direction == next(m["direction"] for m in SUITES[name]["metrics"] if m.get("headline"))


def test_ties_share_a_rank_and_the_next_rank_skips():
    for name in EXPECTED["leaderboards"]:
        ranks = [e.rank for e in visin.Leaderboard.from_json(server_board(name)).entries]
        assert ranks == sorted(ranks)
        for index, rank in enumerate(ranks):
            first = ranks.index(rank)
            assert rank == first + 1, f"{name}: rank {rank} at position {index} should be {first + 1}"


@pytest.mark.parametrize("ident", sorted(EXPECTED["states"]))
def test_a_verdict_reads_as_the_contract_says(ident):
    want = EXPECTED["states"][ident]
    verdict = Verdict.from_json(
        {
            "state": want["state"],
            "reasons": want.get("reasons", []),
            "warnings": want.get("warnings", []),
            **({"evidence": want["evidence"]} if "evidence" in want else {}),
        }
    )
    assert verdict.state == want["state"] and verdict.ranked == (want["state"] == "eligible")
    assert [(r.code, r.detail) for r in verdict.reasons] == [
        (r["code"], r.get("detail")) for r in want.get("reasons", [])
    ]
    assert [r.code for r in verdict.warnings] == [r["code"] for r in want.get("warnings", [])]
    assert verdict.evidence == want.get("evidence")


def evaluation_of(ident: str) -> visin.Evaluation:
    """A fixture evaluation as the server stores it: ranked, with the scores its suite's aggregation gives."""
    item = EVALUATIONS[ident]
    protocol = SUITES[item["suite"]]
    key = next(m["key"] for m in protocol["metrics"] if m.get("headline"))
    conditions = {
        name: {key: item["results"][name]["overall"][key]}
        for name in (c["name"] for c in protocol["conditions"])
    }
    weights = {
        c["name"]: c["sampleCount"] if protocol["aggregation"] == "sample-weighted-mean" else 1
        for c in protocol["conditions"]
    }
    overall = sum(conditions[name][key] * weights[name] for name in conditions) / sum(weights.values())
    scores = {"conditions": conditions, "overall": {key: overall}}
    return visin.Evaluation(
        id=ident,
        suite=f"{item['suite']}@1",
        suite_digest=DIGEST,
        verdict=Verdict(state="eligible", scores=scores),
    )


def suite_of(name: str) -> visin.Suite:
    return visin.Suite(id=name, slug=name, version=1, name=name, digest=DIGEST, protocol=SUITES[name])


def test_the_scores_used_here_are_the_ones_the_contract_ranks():
    for name, ids in (("errors", ["x", "y", "z", "m-attested"]), ("road", ["c-zero"])):
        by_id = {row["id"]: row for row in EXPECTED["leaderboards"][name]["entries"]}
        for ident in ids:
            key = next(m["key"] for m in SUITES[name]["metrics"] if m.get("headline"))
            assert evaluation_of(ident).verdict.scores["overall"][key] == pytest.approx(
                by_id[ident]["headline"], abs=1e-9
            )


def statuses(baseline: str, candidate: str, name: str = "errors") -> dict:
    diff = visin.diff_evaluations(evaluation_of(baseline), evaluation_of(candidate), suite_of(name))
    return {change.scope: change.status for change in diff.changes}


def test_a_lower_is_better_suite_reads_a_fall_as_better_and_a_rise_as_worse():
    assert statuses("x", "y") == {
        "overall": "improved",
        "day": "ok",
        "night": "improved",
    }
    assert statuses("y", "x") == {"overall": "regressed", "day": "ok", "night": "regressed"}
    assert statuses("y", "z") == {"overall": "regressed", "day": "regressed", "night": "improved"}


def test_a_valid_zero_is_a_score_and_compares_as_one():
    assert statuses("c-zero", "c-zero", "road") == {"overall": "ok", "day": "ok", "night": "ok", "rain": "ok"}
    diff = visin.diff_evaluations(evaluation_of("c-zero"), evaluation_of("c-zero"), suite_of("road"))
    night = next(change for change in diff.changes if change.scope == "night")
    assert (night.baseline, night.candidate) == (0.0, 0.0) and diff.passed


def test_an_evaluation_the_contract_does_not_rank_is_never_compared():
    skipped = visin.Evaluation(
        id="d-skipped",
        suite="road@1",
        suite_digest=DIGEST,
        verdict=Verdict.from_json(
            {
                "state": EXPECTED["states"]["d-skipped"]["state"],
                "reasons": EXPECTED["states"]["d-skipped"]["reasons"],
            }
        ),
    )
    diff = visin.diff_evaluations(evaluation_of("c-zero"), skipped, suite_of("road"))
    assert not diff.passed and all(change.status == "missing" for change in diff.changes)
