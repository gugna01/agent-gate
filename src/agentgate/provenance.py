"""Where a record came from.

Every candidate carries its origin from the moment it is created. This is
what makes it possible to audit backwards when a bad record is found later,
which is the only way to learn anything from a bad record.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class Provenance:
    """Immutable origin trail for a single candidate record."""

    source_id: str
    extractor: str
    raw_excerpt: str
    observed_at: datetime = field(default_factory=_now)

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("provenance requires a non-empty source_id")
        if not self.extractor:
            raise ValueError("provenance requires the name of the extractor")

    @property
    def fingerprint(self) -> str:
        """Stable hash of the raw material this record was derived from.

        Lets you detect that two records came from the same source text even
        after normalization has rewritten both of them.
        """
        digest = hashlib.sha256(self.raw_excerpt.strip().encode("utf-8"))
        return digest.hexdigest()[:16]

    def describe(self) -> str:
        stamp = self.observed_at.isoformat(timespec="seconds")
        return f"{self.source_id} via {self.extractor} at {stamp} [{self.fingerprint}]"
