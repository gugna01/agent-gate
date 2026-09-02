"""Deciding whether retrieval found anything worth answering from.

This is the gate applied to retrieval. A system that always returns its
best chunks will always produce an answer, including for questions its
corpus cannot address — and an answer built on irrelevant context is the
most convincing kind of wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .index import Hit, VectorIndex


class RetrievalVerdict(str, Enum):
    GROUNDED = "grounded"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    verdict: RetrievalVerdict
    hits: list[Hit]
    reason: str

    @property
    def context(self) -> str:
        return "\n\n".join(f"[{hit.chunk.chunk_id}] {hit.chunk.text}" for hit in self.hits)

    @property
    def sources(self) -> list[str]:
        return [hit.chunk.chunk_id for hit in self.hits]


class Retriever:
    """Search, then decide whether the result is worth using.

    ``min_score`` is the honest-refusal threshold: below it, the correct
    output is "the corpus does not cover this", not a fluent paragraph.
    """

    def __init__(self, index: VectorIndex, top_k: int = 4, min_score: float = 0.12) -> None:
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")
        self.index = index
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, question: str) -> RetrievalResult:
        if not question.strip():
            return RetrievalResult(RetrievalVerdict.INSUFFICIENT, [], "empty question")

        hits = self.index.search(question, top_k=self.top_k)
        if not hits:
            return RetrievalResult(RetrievalVerdict.INSUFFICIENT, [], "index is empty")

        best = hits[0].score
        if best < self.min_score:
            return RetrievalResult(
                RetrievalVerdict.INSUFFICIENT,
                [],
                f"best match scored {best:.3f}, below floor {self.min_score} — corpus does not cover this",
            )

        # Keep only hits close to the best one. Padding the context with
        # weak matches does not add information; it adds plausible-looking
        # material for the model to drift toward.
        kept = [hit for hit in hits if hit.score >= max(self.min_score, best * 0.55)]
        return RetrievalResult(
            RetrievalVerdict.GROUNDED,
            kept,
            f"{len(kept)} chunk(s) above floor, best {best:.3f}",
        )
