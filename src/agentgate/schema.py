"""The record we are trying to produce, and what makes one valid.

Validation here is structural and boring on purpose: formats, ranges,
required fields. It runs *before* anything reaches a store, and it does not
consult the model that produced the candidate. A model cannot be the judge
of its own output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterator

from .errors import SchemaViolation
from .provenance import Provenance

# Deliberately permissive: this is a shape check, not an existence check.
_EMAIL = re.compile(r"^[^@\s]+@[^@\s.]+\.[^@\s]{2,}$")
_URL = re.compile(r"^https?://[^\s/]+\.[^\s]+$")

VALID_SEGMENTS = frozenset({"enterprise", "mid_market", "small_business"})

# A confidence outside this range means the extractor is miscalibrated,
# not that the record is bad. Those are different problems and deserve
# different alarms.
_CONFIDENCE_RANGE = (0.0, 1.0)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One extracted record, not yet trusted."""

    name: str
    website: str
    contact_email: str
    segment: str
    headcount: int | None
    confidence: float
    provenance: Provenance

    def with_fields(self, **changes: object) -> "Candidate":
        """Return a copy with some fields replaced. Candidates are immutable."""
        return replace(self, **changes)  # type: ignore[arg-type]


def _check(condition: bool, field: str, reason: str) -> None:
    if not condition:
        raise SchemaViolation(field, reason)


def validate(candidate: Candidate) -> None:
    """Raise SchemaViolation on the first structural problem found.

    Fails fast and names the field. A validator that returns False and
    nothing else is a validator you will curse at 2am.
    """
    name = candidate.name.strip()
    _check(bool(name), "name", "empty")
    _check(len(name) <= 200, "name", f"length {len(name)} exceeds 200")

    _check(bool(_URL.match(candidate.website)), "website", "not an http(s) URL")
    _check(bool(_EMAIL.match(candidate.contact_email)), "contact_email", "malformed")

    _check(
        candidate.segment in VALID_SEGMENTS,
        "segment",
        f"{candidate.segment!r} not in {sorted(VALID_SEGMENTS)}",
    )

    # An absent optional field is not a violation. It is missing evidence,
    # and missing evidence belongs in the confidence score, not in the
    # rejection pile. Conflating the two sends good records to the bin and
    # is one of the easiest ways to make a gate look stricter than it is.
    if candidate.headcount is not None:
        _check(isinstance(candidate.headcount, int), "headcount", "not an integer")
        _check(candidate.headcount > 0, "headcount", "must be positive")
        _check(candidate.headcount <= 5_000_000, "headcount", "implausibly large")

    low, high = _CONFIDENCE_RANGE
    _check(
        low <= candidate.confidence <= high,
        "confidence",
        f"{candidate.confidence} outside [{low}, {high}] — extractor is miscalibrated",
    )


def validation_errors(candidates: list[Candidate]) -> Iterator[tuple[Candidate, SchemaViolation]]:
    """Yield every candidate that fails validation, with its reason."""
    for candidate in candidates:
        try:
            validate(candidate)
        except SchemaViolation as exc:
            yield candidate, exc
