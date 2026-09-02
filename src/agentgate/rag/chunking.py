"""Splitting documents into retrievable pieces.

Where you cut matters more than how big the pieces are. A chunk that ends
mid-sentence, or that splits a table from its header, retrieves badly no
matter how good the embedding model is: the meaning was destroyed before
the model ever saw it.

So the splitter here respects structure first and size second. It will
return a chunk slightly over target rather than cut a sentence in half.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Paragraph breaks are the strongest signal an author gives about where one
# idea ends. Sentence ends are the fallback.
_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡])")


@dataclass(frozen=True, slots=True)
class Chunk:
    """A piece of a document, with enough context to be cited."""

    doc_id: str
    index: int
    text: str

    @property
    def chunk_id(self) -> str:
        return f"{'$'}{'{'}self.doc_id}#{'$'}{'{'}self.index}"

    def preview(self, width: int = 90) -> str:
        flat = re.sub(r"\s+", " ", self.text).strip()
        return flat if len(flat) <= width else flat[: width - 1] + "…"


def split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def chunk_document(
    doc_id: str,
    text: str,
    target_chars: int = 500,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """Split into chunks of roughly \u0060\u0060target_chars\u0060\u0060, never mid-sentence.

    \u0060\u0060overlap_sentences\u0060\u0060 repeats the tail of each chunk at the head of the
    next one. Overlap costs storage and buys recall: a fact that straddles a
    boundary would otherwise be retrievable from neither side. One sentence
    is usually enough — more starts returning near-duplicate chunks, which
    quietly crowds out genuinely different material in the top-k.
    """
    if target_chars < 100:
        raise ValueError("target_chars below 100 produces chunks too small to carry meaning")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences cannot be negative")

    text = text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    buffer: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal buffer, size
        if not buffer:
            return
        chunks.append(Chunk(doc_id=doc_id, index=len(chunks), text=" ".join(buffer).strip()))
        buffer = buffer[-overlap_sentences:] if overlap_sentences else []
        size = sum(len(s) for s in buffer)

    for paragraph in _PARAGRAPH.split(text):
        for sentence in split_sentences(paragraph):
            # Emit before adding, so a chunk only exceeds the target by the
            # length of one sentence — never by an arbitrary amount.
            if size and size + len(sentence) > target_chars:
                flush()
            buffer.append(sentence)
            size += len(sentence) + 1

    flush()
    return chunks
