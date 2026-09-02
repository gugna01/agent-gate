"""Storing and searching vectors.

An in-memory index with exhaustive search. That is the right choice at this
scale and an honest one: swapping in a real vector database changes the
search implementation and nothing about the surrounding contract. What
matters here is what the index refuses to do, not how fast it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .chunking import Chunk
from .embedding import Embedder, Vector, cosine


@dataclass(frozen=True, slots=True)
class Hit:
    chunk: Chunk
    score: float

    def __str__(self) -> str:
        return f"{self.chunk.chunk_id}  {self.score:.3f}  {self.chunk.preview(70)}"


@dataclass(slots=True)
class VectorIndex:
    """Chunks plus their vectors, searchable by similarity."""

    embedder: Embedder
    _vectors: dict[str, Vector] = field(default_factory=dict, init=False)
    _chunks: dict[str, Chunk] = field(default_factory=dict, init=False)

    def add(self, chunks: list[Chunk]) -> int:
        for chunk in chunks:
            if not chunk.text.strip():
                continue
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = self.embedder.embed(chunk.text)
        return len(self._chunks)

    def search(self, query: str, top_k: int = 4) -> list[Hit]:
        """Return the ``top_k`` closest chunks, best first.

        No score floor is applied here. Deciding what is *good enough* is a
        policy question, and policy belongs in the retriever where it can be
        configured and tested — not buried in the storage layer.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not self._vectors:
            return []

        query_vector = self.embedder.embed(query)
        scored = [
            Hit(chunk=self._chunks[cid], score=cosine(query_vector, vec))
            for cid, vec in self._vectors.items()
        ]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._chunks)
