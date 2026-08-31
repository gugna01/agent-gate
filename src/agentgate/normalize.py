"""Canonicalization.

Normalization runs before the gate, never after. Two records that describe
the same company must look identical by the time anything decides whether
to write them, or the deduplication downstream is decoration.
"""

from __future__ import annotations

import re

from .schema import Candidate

_LEGAL_SUFFIXES = (
    " inc", " inc.", " llc", " ltd", " ltda", " s.a.s", " sas", " s.a.", " sa",
    " corp", " corporation", " co.", " gmbh", " bv", " nv", " plc",
)

_SEGMENT_ALIASES = {
    "smb": "small_business",
    "small": "small_business",
    "small business": "small_business",
    "midmarket": "mid_market",
    "mid market": "mid_market",
    "mid-market": "mid_market",
    "enterprise": "enterprise",
    "ent": "enterprise",
}


def canonical_name(raw: str) -> str:
    """Collapse whitespace, drop trailing legal suffixes, title-case."""
    name = re.sub(r"\s+", " ", raw).strip().rstrip(",.")
    lowered = name.lower()
    for suffix in _LEGAL_SUFFIXES:
        if lowered.endswith(suffix):
            name = name[: len(name) - len(suffix)].rstrip(" ,.")
            break
    return name.title()


def canonical_website(raw: str) -> str:
    """Lowercase host, strip ``www.``, drop trailing slash and query string."""
    url = raw.strip().lower()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    url = url.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    return url.replace("://www.", "://", 1)


def canonical_segment(raw: str) -> str:
    key = re.sub(r"[\s_-]+", " ", raw.strip().lower())
    return _SEGMENT_ALIASES.get(key, _SEGMENT_ALIASES.get(key.replace(" ", ""), raw.strip().lower()))


def normalize(candidate: Candidate) -> Candidate:
    """Return a canonicalized copy. Confidence is never touched here.

    Normalization cleans shape, not truth. Nudging a score during cleanup is
    how a pipeline quietly starts trusting itself more than it should.
    """
    return candidate.with_fields(
        name=canonical_name(candidate.name),
        website=canonical_website(candidate.website),
        contact_email=candidate.contact_email.strip().lower(),
        segment=canonical_segment(candidate.segment),
    )
