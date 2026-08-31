import pytest

from agentgate import Candidate, Gate, Provenance, Thresholds, Verdict
from agentgate.errors import SchemaViolation
from agentgate.schema import validate


def make(**overrides) -> Candidate:
    base = dict(
        name="Northwind Logistics",
        website="https://northwind.example",
        contact_email="ops@northwind.example",
        segment="mid_market",
        headcount=320,
        confidence=0.9,
        provenance=Provenance(source_id="doc-1", extractor="test", raw_excerpt="raw"),
    )
    base.update(overrides)
    return Candidate(**base)


class TestThresholds:
    def test_rejects_inverted_bounds(self):
        with pytest.raises(ValueError):
            Thresholds(accept=0.4, review_floor=0.9)

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            Thresholds(accept=1.5, review_floor=0.5)


class TestRouting:
    def test_high_confidence_is_accepted(self):
        assert Gate().evaluate(make(confidence=0.95)).verdict is Verdict.ACCEPTED

    def test_middle_confidence_goes_to_review(self):
        assert Gate().evaluate(make(confidence=0.7)).verdict is Verdict.REVIEW

    def test_low_confidence_is_rejected(self):
        assert Gate().evaluate(make(confidence=0.2)).verdict is Verdict.REJECTED

    def test_accept_boundary_is_inclusive(self):
        gate = Gate(Thresholds(accept=0.85, review_floor=0.5))
        assert gate.evaluate(make(confidence=0.85)).verdict is Verdict.ACCEPTED

    def test_review_floor_is_inclusive(self):
        gate = Gate(Thresholds(accept=0.85, review_floor=0.5))
        assert gate.evaluate(make(confidence=0.5)).verdict is Verdict.REVIEW


class TestStructureBeatsConfidence:
    """The property the whole design rests on."""

    def test_malformed_record_is_rejected_even_at_full_confidence(self):
        decision = Gate().evaluate(make(contact_email="nope", confidence=1.0))
        assert decision.verdict is Verdict.REJECTED
        assert "contact_email" in decision.reason

    def test_unknown_segment_is_rejected_even_at_full_confidence(self):
        decision = Gate().evaluate(make(segment="mid-size", confidence=1.0))
        assert decision.verdict is Verdict.REJECTED


class TestSchema:
    def test_absent_optional_field_is_not_a_violation(self):
        validate(make(headcount=None))  # must not raise

    def test_zero_headcount_is_a_violation(self):
        with pytest.raises(SchemaViolation):
            validate(make(headcount=0))

    def test_confidence_outside_range_names_miscalibration(self):
        with pytest.raises(SchemaViolation) as exc:
            validate(make(confidence=1.4))
        assert "miscalibrated" in str(exc.value)
