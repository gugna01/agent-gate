"""The compuerta — the only place that decides what gets written.

Every candidate leaves here as exactly one of three things: ACCEPTED,
REVIEW or REJECTED. There is no fourth path and no way around it. That
constraint is the whole design; everything else in this package exists to
make the constraint enforceable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .errors import ProvenanceMissing, SchemaViolation
from .schema import Candidate, validate


class Verdict(str, Enum):
    ACCEPTED = "accepted"
    REVIEW = "review"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class Decision:
    verdict: Verdict
    candidate: Candidate
    reason: str

    def __str__(self) -> str:
        return f"[{self.verdict.value:>8}] {self.candidate.name or '<unnamed>'} — {self.reason}"


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Where the lines sit.

    ``accept`` is deliberately high and ``review_floor`` deliberately low.
    The band between them is not indecision — it is the part of the problem
    that a person is better at, and pretending otherwise is what fills a
    database with plausible garbage.
    """

    accept: float = 0.85
    review_floor: float = 0.50

    def __post_init__(self) -> None:
        if not 0.0 <= self.review_floor <= self.accept <= 1.0:
            raise ValueError(
                f"thresholds must satisfy 0 <= review_floor ({self.review_floor}) "
                f"<= accept ({self.accept}) <= 1"
            )


class Gate:
    """Structural validation first, confidence second. Never the reverse.

    A high-confidence record with a malformed email is still garbage, and a
    gate that checks the score first will happily wave it through.
    """

    def __init__(self, thresholds: Thresholds | None = None) -> None:
        self.thresholds = thresholds or Thresholds()

    def evaluate(self, candidate: Candidate) -> Decision:
        if candidate.provenance is None:  # type: ignore[comparison-overlap]
            raise ProvenanceMissing("candidate reached the gate without provenance")

        try:
            validate(candidate)
        except SchemaViolation as exc:
            return Decision(Verdict.REJECTED, candidate, f"schema: {exc.field} — {exc.reason}")

        score = candidate.confidence
        if score >= self.thresholds.accept:
            return Decision(Verdict.ACCEPTED, candidate, f"confidence {score:.2f} >= {self.thresholds.accept}")
        if score >= self.thresholds.review_floor:
            return Decision(
                Verdict.REVIEW,
                candidate,
                f"confidence {score:.2f} in review band "
                f"[{self.thresholds.review_floor}, {self.thresholds.accept})",
            )
        return Decision(Verdict.REJECTED, candidate, f"confidence {score:.2f} < {self.thresholds.review_floor}")
