import json

import pytest

from agentgate import HeuristicExtractor, ModelExtractor, Pipeline, Thresholds
from agentgate.errors import ExtractionFailed
from agentgate.normalize import canonical_name, canonical_segment, canonical_website

COMPLETE = (
    "doc-1",
    "Company: Northwind Logistics Inc.\n"
    "Site: https://www.northwind.example/\n"
    "Contact: OPS@Northwind.example\n"
    "Segment: mid_market\n"
    "Employees: 320\n",
)


class TestHeuristicExtractor:
    def test_confidence_reflects_field_coverage(self):
        candidate = HeuristicExtractor().extract(COMPLETE)
        assert candidate.confidence == 1.0

    def test_missing_name_fails_loudly(self):
        with pytest.raises(ExtractionFailed) as exc:
            HeuristicExtractor().extract(("doc-2", "Contact: a@b.example"))
        assert "no company name" in exc.value.reason

    def test_empty_document_fails_loudly(self):
        with pytest.raises(ExtractionFailed):
            HeuristicExtractor().extract(("doc-3", "   "))

    def test_provenance_is_attached(self):
        candidate = HeuristicExtractor().extract(COMPLETE)
        assert candidate.provenance.source_id == "doc-1"
        assert candidate.provenance.fingerprint


class TestNormalization:
    def test_strips_legal_suffix(self):
        assert canonical_name("Northwind Logistics Inc.") == "Northwind Logistics"

    def test_website_drops_www_and_trailing_slash(self):
        assert canonical_website("https://www.northwind.example/") == "https://northwind.example"

    def test_website_gets_scheme(self):
        assert canonical_website("northwind.example") == "https://northwind.example"

    def test_segment_aliases_collapse(self):
        assert canonical_segment("Mid-Market") == "mid_market"
        assert canonical_segment("SMB") == "small_business"

    def test_normalization_does_not_touch_confidence(self):
        from agentgate import normalize

        candidate = HeuristicExtractor().extract(COMPLETE)
        assert normalize(candidate).confidence == candidate.confidence


class TestModelExtractor:
    def test_non_json_output_is_a_failure_not_a_guess(self):
        extractor = ModelExtractor(complete=lambda _: "Sure! Here is the company you asked about.")
        with pytest.raises(ExtractionFailed) as exc:
            extractor.extract(("doc-4", "some text"))
        assert "did not return JSON" in exc.value.reason

    def test_code_fenced_json_is_tolerated(self):
        payload = json.dumps(
            {
                "name": "Beacon Analytics",
                "website": "beacon.example",
                "contact_email": "hi@beacon.example",
                "segment": "enterprise",
                "headcount": 90,
                "confidence": 0.9,
            }
        )
        extractor = ModelExtractor(complete=lambda _: f"```json\n{payload}\n```")
        assert extractor.extract(("doc-5", "text")).name == "Beacon Analytics"

    def test_self_reported_confidence_is_clamped(self):
        payload = json.dumps({"name": "X", "confidence": 42})
        extractor = ModelExtractor(complete=lambda _: payload)
        assert extractor.extract(("doc-6", "text")).confidence == 1.0

    def test_declared_error_is_respected(self):
        extractor = ModelExtractor(complete=lambda _: json.dumps({"error": "no name"}))
        with pytest.raises(ExtractionFailed):
            extractor.extract(("doc-7", "text"))


class TestPipeline:
    def test_routes_each_document_to_exactly_one_exit(self):
        docs = [
            COMPLETE,
            ("doc-b", "Company: Beacon Analytics\nSite: beacon.example\n"
                      "Contact: hi@beacon.example\nSegment: enterprise\n"),
            ("doc-c", "Company: Bad Co\nSite: https://bad.example\nContact: nope\n"
                      "Segment: enterprise\nEmployees: 5\n"),
            ("doc-d", "no identity here"),
        ]
        report = Pipeline(HeuristicExtractor(), Thresholds()).run(docs)

        assert len(report.accepted) == 1
        assert len(report.queue) == 1
        assert len(report.rejected) == 1
        assert len(report.failures) == 1
        assert report.processed == len(docs)

    def test_extraction_failure_is_not_counted_as_rejection(self):
        report = Pipeline(HeuristicExtractor()).run([("doc-x", "")])
        assert report.failures and not report.rejected
