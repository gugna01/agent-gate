"""Retrieval-augmented answering with a groundedness gate.

Retrieval decides what the model reads. The gate decides whether what it
said can be traced back to it. Both are needed: the first without the
second produces confident answers built on correct context that still
contain invented sentences.
"""

from .answering import Answer, AnswerVerdict, GroundedAnswerer, groundedness, sentence_support
from .chunking import Chunk, chunk_document, split_sentences
from .embedding import Embedder, HashingEmbedder, ModelEmbedder, Vector, cosine, tokenize
from .index import Hit, VectorIndex
from .retrieval import RetrievalResult, RetrievalVerdict, Retriever

__all__ = [
    "Answer",
    "AnswerVerdict",
    "Chunk",
    "Embedder",
    "GroundedAnswerer",
    "HashingEmbedder",
    "Hit",
    "ModelEmbedder",
    "RetrievalResult",
    "RetrievalVerdict",
    "Retriever",
    "Vector",
    "chunk_document",
    "cosine",
    "groundedness",
    "sentence_support",
    "split_sentences",
    "tokenize",
    "VectorIndex",
]
