"""Diff two evaluations on one suite version: a regression gate for CI.

    baseline = api.evaluation("6ab5932a0e9a6b7570e30e2e")
    candidate = api.evaluation("6ab5932a0e9a6b7570e30e30")
    result = visin.diff_evaluations(baseline, candidate, api.suite("road-test@1"), max_drop=0.01)
    result.passed  # False when a score fell by more than 0.01, or one could not be compared

Only scores the server judged eligible are compared, and only on the same suite version, since a different
protocol is a different measurement. Each score is read in the suite's direction (a higher ``mIoU`` is
better, a lower latency is better), so a positive ``improvement`` always means the candidate got better.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .models import Evaluation, Suite

__all__ = ["Change", "Diff", "diff_evaluations"]

OVERALL = "overall"
TOLERANCE = 1e-9


@dataclass(frozen=True)
class Change:
    """One score in both evaluations: ``scope`` is a condition name or ``overall``.

    ``improvement`` is positive when the candidate is better in the metric's direction, negative when it is
    worse, and ``None`` when either evaluation has no value (``status`` is then ``missing``).
    """

    scope: str
    metric: str
    direction: str
    baseline: float | None
    candidate: float | None
    improvement: float | None
    status: str


@dataclass(frozen=True)
class Diff:
    """Every compared score, and what the diff decides.

    ``regressions`` are scores that fell by more than ``max_drop``; ``missing`` are scores one side lacks,
    including every score of an evaluation that could not be ranked. ``passed`` needs neither.
    """

    suite: str
    baseline: str | None
    candidate: str | None
    max_drop: float
    changes: tuple[Change, ...]
    problems: tuple[str, ...] = ()

    @property
    def regressions(self) -> tuple[Change, ...]:
        """The scores that fell by more than the allowed drop."""
        return tuple(change for change in self.changes if change.status == "regressed")

    @property
    def missing(self) -> tuple[Change, ...]:
        """The scores that could not be compared because a side has none."""
        return tuple(change for change in self.changes if change.status == "missing")

    @property
    def passed(self) -> bool:
        """Whether nothing regressed and nothing was missing."""
        return not self.regressions and not self.missing and not self.problems


def _value(scores: Mapping[str, Any] | None, scope: str, metric: str) -> float | None:
    if not scores:
        return None
    block = scores.get(OVERALL) if scope == OVERALL else (scores.get("conditions") or {}).get(scope)
    value = block.get(metric) if isinstance(block, Mapping) else None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _scores(evaluation: Evaluation) -> Mapping[str, Any] | None:
    return evaluation.verdict.scores if evaluation.verdict and evaluation.verdict.ranked else None


def diff_evaluations(
    baseline: Evaluation,
    candidate: Evaluation,
    suite: Suite,
    *,
    max_drop: float = 0.0,
    all_metrics: bool = False,
) -> Diff:
    """Diff the candidate against the baseline on ``suite``, score by score.

    The headline metric is compared for the overall figure and every condition; ``all_metrics=True`` compares
    every metric the suite names. A score that fell by more than ``max_drop`` (in the metric's own units) is
    a regression. Raises ``ValueError`` when the two evaluations are not on this suite version, or not on
    the same protocol, which cannot be compared.
    """
    if max_drop < 0:
        raise ValueError("max_drop is how far a score may fall, so it cannot be negative")
    for label, evaluation in (("baseline", baseline), ("candidate", candidate)):
        if evaluation.suite != suite.ref:
            where = evaluation.suite or "no suite"
            raise ValueError(
                f"the {label} is on {where}, not {suite.ref}: only one suite version is compared"
            )
        if evaluation.suite_digest and suite.digest and evaluation.suite_digest != suite.digest:
            raise ValueError(f"the {label} was judged on a different protocol than {suite.ref} has now")
    metrics = [
        metric
        for metric in suite.protocol.get("metrics") or ()
        if isinstance(metric, Mapping) and (all_metrics or metric.get("headline"))
    ]
    scopes = [
        OVERALL,
        *(str(c.get("name")) for c in suite.protocol.get("conditions") or () if isinstance(c, Mapping)),
    ]
    base_scores, cand_scores = _scores(baseline), _scores(candidate)
    problems = tuple(
        f"the {label} ({evaluation.id or evaluation.uuid}) is not ranked on {suite.ref}"
        for label, evaluation, scores in (
            ("baseline", baseline, base_scores),
            ("candidate", candidate, cand_scores),
        )
        if scores is None
    )
    changes = []
    for metric in metrics:
        key, direction = str(metric.get("key")), str(metric.get("direction") or "max")
        for scope in scopes:
            before, after = _value(base_scores, scope, key), _value(cand_scores, scope, key)
            if before is None or after is None:
                changes.append(Change(scope, key, direction, before, after, None, "missing"))
                continue
            improvement = after - before if direction == "max" else before - after
            status = (
                "regressed"
                if -improvement > max_drop + TOLERANCE
                else "improved"
                if improvement > TOLERANCE
                else "ok"
            )
            changes.append(Change(scope, key, direction, before, after, improvement, status))
    return Diff(suite.ref, baseline.id, candidate.id, max_drop, tuple(changes), problems)
