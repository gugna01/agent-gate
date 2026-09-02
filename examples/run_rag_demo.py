"""RAG with a groundedness gate, end to end. No API key, no network.

    python examples/run_rag_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentgate.rag import (  # noqa: E402
    GroundedAnswerer,
    HashingEmbedder,
    Retriever,
    VectorIndex,
    chunk_document,
)
from rag_corpus import DOCUMENTS, scripted_model  # noqa: E402

QUESTIONS = [
    # Answered correctly and fully supported by the source.
    "¿Cuántos días tengo para devolver un producto?",
    # Two true sentences plus one invented. The one this package exists for.
    "¿Qué cubre la garantía y cómo la activo?",
    # Retrieval works despite the morphology: "envíos" vs "envío", "rurales".
    "¿Cuánto demoran los envíos a zonas rurales?",
    # Nothing in the corpus covers this. The honest output is a refusal.
    "¿Puedo pagar en cuotas sin interés?",
]


def main() -> int:
    index = VectorIndex(embedder=HashingEmbedder(dimensions=512))
    for doc_id, text in DOCUMENTS.items():
        index.add(chunk_document(doc_id, text, target_chars=320))

    answerer = GroundedAnswerer(
        retriever=Retriever(index, top_k=3, min_score=0.12),
        complete=scripted_model,
    )

    print("=" * 74)
    print(f"corpus: {len(DOCUMENTS)} documento(s) -> {len(index)} fragmento(s) indexados")
    print("=" * 74)

    for question in QUESTIONS:
        answer = answerer.ask(question)
        print(f"\nP: {question}")
        print(answer)
        print(f"  R: {answer.text}")
        for sentence in answer.unsupported:
            print(f"  ⚠ sin respaldo en el corpus: {sentence}")

    print(
        "\n" + "-" * 74 + "\n"
        "La segunda pregunta es la importante. El modelo respondió dos frases\n"
        "correctas y agregó una tercera, fluida y falsa, sobre extender la\n"
        "garantía pagando una prima. Está bien escrita, es específica y no\n"
        "aparece en ningún documento. Ningún try/except la detecta; la\n"
        "verificación de respaldo sí, y la nombra frase por frase.\n"
    )

    _limitation_demo(answerer)
    return 0


def _limitation_demo(answerer: GroundedAnswerer) -> None:
    """Show where lexical retrieval breaks, instead of hiding it.

    The corpus says "los envíos demoran"; the question says "en cuánto
    tiempo llega un pedido". No shared vocabulary, so a lexical embedder
    ranks an unrelated chunk higher. A real embedding model closes this gap
    and nothing else in the pipeline changes.
    """
    question = "¿En cuánto tiempo llega un pedido a una ciudad principal?"
    hits = answerer.retriever.index.search(question, top_k=2)
    print("-" * 74)
    print("LÍMITE CONOCIDO — recuperación léxica, misma pregunta en otras palabras")
    print(f"\nP: {question}")
    for hit in hits:
        print(f"   {hit}")
    print(
        "\nLa respuesta correcta está en envios#0, pero la pregunta no comparte\n"
        "ni una palabra de contenido con ella. Esto no se arregla ajustando\n"
        "umbrales: se arregla cambiando HashingEmbedder por un modelo real de\n"
        "embeddings, que es exactamente para lo que existe ModelEmbedder."
    )


if __name__ == "__main__":
    raise SystemExit(main())
