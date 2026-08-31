"""Extraction agents.

Two implementations ship here:

* ``HeuristicExtractor`` — deterministic, no network, no API key. It is what
  makes this repository runnable by anyone in one command, and it is what the
  tests run against.
* ``ModelExtractor`` — the same interface, backed by a callable you supply
  (an LLM client, a hosted endpoint, whatever). The pipeline cannot tell the
  difference, which is the point: the safety machinery does not depend on
  which one is plugged in.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Protocol

from .errors import ExtractionFailed
from .provenance import Provenance
from .schema import Candidate

RawDocument = tuple[str, str]  # (source_id, text)


class Extractor(Protocol):
    """Anything that turns raw text into a candidate record."""

    name: str

    def extract(self, document: RawDocument) -> Candidate: ...


_FIELD_PATTERNS = {
    "name": re.compile(r"(?im)^\s*company\s*:\s*(.+?)\s*$"),
    "website": re.compile(r"(?im)^\s*site\s*:\s*(\S+)\s*$"),
    "contact_email": re.compile(r"(?im)^\s*contact\s*:\s*(\S+)\s*$"),
    "segment": re.compile(r"(?im)^\s*segment\s*:\s*(\w+)\s*$"),
    "headcount": re.compile(r"(?im)^\s*employees\s*:\s*([\d,\.]+)\s*$"),
}


class HeuristicExtractor:
    """Rule-based extraction. Confidence is a function of how much it found.

    The confidence score here is honest by construction: it is the share of
    fields the parser actually located. It never reports certainty it does
    not have, which is exactly the property a probabilistic extractor tends
    to lack unless you force it.
    """

    name = "heuristic-v1"

    def extract(self, document: RawDocument) -> Candidate:
        source_id, text = document
        if not text.strip():
            raise ExtractionFailed(source_id, "empty document")

        found: dict[str, str] = {}
        for field, pattern in _FIELD_PATTERNS.items():
            match = pattern.search(text)
            if match:
                found[field] = match.group(1).strip()

        if "name" not in found:
            # Without an identity there is nothing to reconcile against.
            # Guessing one from surrounding prose is how phantom records
            # get born.
            raise ExtractionFailed(source_id, "no company name found")

        confidence = round(len(found) / len(_FIELD_PATTERNS), 3)

        return Candidate(
            name=found.get("name", ""),
            website=found.get("website", ""),
            contact_email=found.get("contact_email", ""),
            segment=found.get("segment", "").lower(),
            headcount=_to_int(found["headcount"]) if "headcount" in found else None,
            confidence=confidence,
            provenance=Provenance(
                source_id=source_id,
                extractor=self.name,
                raw_excerpt=text[:400],
            ),
        )


class ModelExtractor:
    """Adapter for a generative model, kept behind the same interface.

    ``complete`` receives a prompt and must return the model's raw text. Any
    client works. The contract this class enforces on the model's behalf:

    * output must parse as JSON — a model that free-associates is a failure,
      not a partial success;
    * a missing name is a failure, not a field to invent;
    * the model's self-reported confidence is clamped, then still subject to
      the same gate as everything else. Self-assessment is an input, never
      a verdict.
    """

    name = "model-v1"

    _PROMPT = (
        "Extract a company record from the text below.\n"
        "Return ONLY a JSON object with keys: name, website, contact_email, "
        "segment, headcount, confidence.\n"
        "segment must be one of: enterprise, mid_market, small_business.\n"
        'If the company name is not stated, return {"error": "no name"}.\n'
        "Do not infer values that are not present.\n\n"
        "TEXT:\n<<TEXT>>\n"
    )

    def __init__(self, complete: Callable[[str], str], name: str | None = None) -> None:
        self._complete = complete
        if name:
            self.name = name

    def extract(self, document: RawDocument) -> Candidate:
        source_id, text = document
        if not text.strip():
            raise ExtractionFailed(source_id, "empty document")

        # str.replace, not str.format: the prompt contains literal JSON
        # braces, and format() reads those as placeholders. Caught by a test,
        # which is the only reason it is not a production incident.
        raw = self._complete(self._PROMPT.replace("<<TEXT>>", text))

        try:
            payload = json.loads(_strip_code_fence(raw))
        except json.JSONDecodeError as exc:
            raise ExtractionFailed(source_id, f"model did not return JSON: {exc}") from exc

        if not isinstance(payload, dict):
            raise ExtractionFailed(source_id, "model returned a non-object")
        if "error" in payload:
            raise ExtractionFailed(source_id, str(payload["error"]))
        if not payload.get("name"):
            raise ExtractionFailed(source_id, "model returned no name")

        return Candidate(
            name=str(payload.get("name", "")).strip(),
            website=str(payload.get("website", "")).strip(),
            contact_email=str(payload.get("contact_email", "")).strip(),
            segment=str(payload.get("segment", "")).strip().lower(),
            headcount=_to_int(payload["headcount"]) if payload.get("headcount") else None,
            confidence=_clamp(payload.get("confidence", 0.0)),
            provenance=Provenance(
                source_id=source_id,
                extractor=self.name,
                raw_excerpt=text[:400],
            ),
        )


def _strip_code_fence(raw: str) -> str:
    """Models like to wrap JSON in markdown. Tolerate it, do not celebrate it."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _to_int(value: object) -> int | None:
    """Return None rather than 0 on failure. Zero is a claim; None is silence."""
    try:
        parsed = int(str(value).replace(",", "").replace(".", "").strip())
    except (TypeError, ValueError):
        return None
    return parsed


def _clamp(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
