"""The human review queue.

Designed in from the start rather than bolted on after the first bad batch.
That ordering is the single lesson this repository exists to demonstrate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .gate import Decision
from .schema import Candidate


@dataclass(slots=True)
class ReviewItem:
    decision: Decision
    resolved: bool = False
    resolution: str | None = None

    @property
    def candidate(self) -> Candidate:
        return self.decision.candidate


@dataclass(slots=True)
class ReviewQueue:
    """In-memory queue. Swap for a real store; the interface is the contract."""

    items: list[ReviewItem] = field(default_factory=list)

    def enqueue(self, decision: Decision) -> ReviewItem:
        item = ReviewItem(decision=decision)
        self.items.append(item)
        return item

    def pending(self) -> list[ReviewItem]:
        return [item for item in self.items if not item.resolved]

    def resolve(self, item: ReviewItem, resolution: str) -> None:
        item.resolved = True
        item.resolution = resolution

    def __len__(self) -> int:
        return len(self.items)

    def report(self) -> str:
        if not self.items:
            return "review queue: empty"
        lines = [f"review queue: {len(self.pending())} pending of {len(self.items)}"]
        for item in self.pending():
            lines.append(f"  - {item.candidate.name}  ({item.decision.reason})")
            lines.append(f"    origin: {item.candidate.provenance.describe()}")
        return "\n".join(lines)
