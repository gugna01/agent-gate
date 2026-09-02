"""Turning text into vectors.

Two implementations, same interface — the same pattern as the extractors.

``HashingEmbedder`` is deterministic, dependency-free and runs offline, which
is what makes this repository testable and runnable by anyone. It is a
LEXICAL stand-in, not a semantic model: it will match "confidence threshold"
to "threshold of confidence" but not to "cutoff for certainty". That
limitation is stated plainly rather than hidden, because a retrieval system
whose weaknesses you cannot name is one you cannot debug.

``ModelEmbedder`` wraps any real embedding API behind the same protocol.
Swapping it in changes retrieval quality and changes nothing else.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from typing import Callable, Protocol, Sequence

Vector = tuple[float, ...]

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str
    dimensions: int

    def embed(self, text: str) -> Vector: ...


def tokenize(text: str) -> list[str]:
    """Lowercase, strip accents, split on word characters.

    Accent folding matters in Spanish: "análisis" and "analisis" must land
    on the same token or half your corpus becomes unreachable.
    """
    folded = unicodedata.normalize("NFD", text.lower())
    folded = "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")
    return _TOKEN.findall(folded)


def char_ngrams(token: str, low: int = 4, high: int = 5) -> list[str]:
    """Sub-word fragments of a token, padded at the boundaries.

    Spanish inflects heavily, and word-level matching breaks on it:
    "devolver", "devolución" and "devoluciones" are three distinct tokens
    that share no overlap at all, so a question about returns retrieves
    nothing about returns. Character n-grams give them a common core
    ("devol", "evolu") and the match survives the morphology.

    This was not a design insight. The first version indexed words only, and
    the demo question "¿Cuántos días tengo para devolver un producto?"
    retrieved the shipping policy — because both mention "días" and neither
    shared "devolver". The fix is the standard one; finding it required
    running the thing and reading the output.
    """
    padded = f"<{token}>"
    grams: list[str] = []
    for size in range(low, high + 1):
        if len(padded) < size:
            continue
        grams.extend(padded[i : i + size] for i in range(len(padded) - size + 1))
    return grams


class HashingEmbedder:
    """Words, word pairs and character n-grams, hashed into a fixed vector.

    Bigrams are included because word order carries meaning that unigrams
    throw away: "no aceptar" and "aceptar" share every unigram and mean
    opposite things.

    Character n-grams carry less weight than whole words. They exist to
    rescue morphological variants, not to dominate the signal — at equal
    weight, two long unrelated words that happen to share a fragment start
    outranking an exact match.
    """

    name = "hashing-v2"

    #: Sub-word matches are real evidence, but weaker than a word matching a word.
    CHAR_GRAM_WEIGHT = 0.35

    def __init__(self, dimensions: int = 512) -> None:
        if dimensions < 64:
            raise ValueError("dimensions below 64 collide too often to retrieve reliably")
        self.dimensions = dimensions

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dimensions

    def embed(self, text: str) -> Vector:
        tokens = tokenize(text)
        if not tokens:
            return tuple([0.0] * self.dimensions)

        weights = [0.0] * self.dimensions

        for gram in tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]:
            weights[self._bucket(gram)] += 1.0

        for token in tokens:
            if len(token) < 5:
                # Short words are already their own best signal; fragmenting
                # them only manufactures collisions.
                continue
            for gram in char_ngrams(token):
                weights[self._bucket(gram)] += self.CHAR_GRAM_WEIGHT

        # L2 normalization makes cosine similarity a plain dot product and
        # stops long chunks from outranking short ones on length alone.
        norm = math.sqrt(sum(w * w for w in weights))
        if norm == 0.0:
            return tuple(weights)
        return tuple(w / norm for w in weights)


class ModelEmbedder:
    """Adapter for a real embedding API.

    ``encode`` receives text and must return a sequence of floats. The
    adapter normalizes and checks dimensional consistency, because a
    provider silently changing vector size is the kind of failure that
    corrupts an index without raising anything.
    """

    name = "model-v1"

    def __init__(self, encode: Callable[[str], Sequence[float]], dimensions: int, name: str | None = None) -> None:
        self._encode = encode
        self.dimensions = dimensions
        if name:
            self.name = name

    def embed(self, text: str) -> Vector:
        raw = list(self._encode(text))
        if len(raw) != self.dimensions:
            raise ValueError(
                f"embedder returned {len(raw)} dimensions, index expects {self.dimensions} — "
                "mixing vector sizes silently corrupts retrieval"
            )
        norm = math.sqrt(sum(v * v for v in raw))
        if norm == 0.0:
            return tuple(float(v) for v in raw)
        return tuple(float(v) / norm for v in raw)


def cosine(a: Vector, b: Vector) -> float:
    """Dot product of two normalized vectors."""
    if len(a) != len(b):
        raise ValueError(f"cannot compare vectors of size {len(a)} and {len(b)}")
    return sum(x * y for x, y in zip(a, b))
