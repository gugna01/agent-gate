import pytest

from agentgate.rag import (
    GroundedAnswerer,
    HashingEmbedder,
    ModelEmbedder,
    Retriever,
    VectorIndex,
    chunk_document,
    cosine,
    groundedness,
    sentence_support,
)
from agentgate.rag.answering import AnswerVerdict, _extract_citations
from agentgate.rag.embedding import char_ngrams, tokenize
from agentgate.rag.retrieval import RetrievalVerdict

CORPUS = {
    "devoluciones": (
        "Las devoluciones se aceptan dentro de los treinta días posteriores a la entrega. "
        "El producto debe estar sin uso y conservar su empaque original."
    ),
    "garantia": (
        "La garantía cubre defectos de fabricación durante doce meses. "
        "Para activarla se requiere la factura de compra."
    ),
}


def build_index() -> VectorIndex:
    index = VectorIndex(embedder=HashingEmbedder(dimensions=512))
    for doc_id, text in CORPUS.items():
        index.add(chunk_document(doc_id, text, target_chars=200))
    return index


class TestChunking:
    def test_never_cuts_mid_sentence(self):
        text = "Primera frase corta. " + ("Segunda frase bastante más larga que la anterior. " * 8)
        for chunk in chunk_document("d", text, target_chars=120):
            assert chunk.text.rstrip().endswith((".", "!", "?"))

    def test_overlap_repeats_the_previous_tail(self):
        text = " ".join(f"Frase número {i} con relleno suficiente." for i in range(1, 12))
        chunks = chunk_document("d", text, target_chars=100, overlap_sentences=1)
        assert len(chunks) > 1
        for previous, following in zip(chunks, chunks[1:]):
            assert previous.text.split(". ")[-1].strip(". ") in following.text

    def test_zero_overlap_is_allowed(self):
        chunks = chunk_document("d", "Una. Dos. Tres. Cuatro.", target_chars=100, overlap_sentences=0)
        assert chunks

    def test_empty_document_yields_nothing(self):
        assert chunk_document("d", "   ") == []

    def test_absurd_target_is_rejected(self):
        with pytest.raises(ValueError):
            chunk_document("d", "texto", target_chars=10)

    def test_chunk_id_is_traceable(self):
        assert chunk_document("politica", "Una frase.")[0].chunk_id == "politica#0"


class TestEmbedding:
    def test_accents_are_folded(self):
        assert tokenize("Análisis") == tokenize("analisis")

    def test_identical_text_is_identical_vector(self):
        e = HashingEmbedder(128)
        assert e.embed("hola mundo") == e.embed("hola mundo")

    def test_vectors_are_normalized(self):
        vec = HashingEmbedder(128).embed("una frase cualquiera con varias palabras")
        assert cosine(vec, vec) == pytest.approx(1.0, abs=1e-9)

    def test_char_ngrams_bridge_spanish_morphology(self):
        """The bug the demo exposed: 'devolver' must reach 'devoluciones'."""
        shared = set(char_ngrams("devolver")) & set(char_ngrams("devoluciones"))
        assert shared

    def test_morphological_variants_score_above_unrelated(self):
        e = HashingEmbedder(512)
        variant = cosine(e.embed("devolver un producto"), e.embed("las devoluciones del producto"))
        unrelated = cosine(e.embed("devolver un producto"), e.embed("la garantía cubre defectos"))
        assert variant > unrelated

    def test_short_tokens_are_not_fragmented(self):
        e = HashingEmbedder(512)
        assert e.embed("de la") == e.embed("de la")

    def test_empty_text_gives_zero_vector(self):
        assert set(HashingEmbedder(64).embed("   ")) == {0.0}

    def test_dimension_mismatch_is_loud(self):
        embedder = ModelEmbedder(encode=lambda _: [0.1] * 3, dimensions=4)
        with pytest.raises(ValueError, match="dimensions"):
            embedder.embed("texto")

    def test_cosine_rejects_mismatched_sizes(self):
        with pytest.raises(ValueError):
            cosine((1.0, 0.0), (1.0, 0.0, 0.0))


class TestRetrieval:
    def test_finds_the_right_document(self):
        result = Retriever(build_index()).retrieve("¿Cuántos días tengo para devolver?")
        assert result.verdict is RetrievalVerdict.GROUNDED
        assert result.hits[0].chunk.doc_id == "devoluciones"

    def test_refuses_when_nothing_is_close_enough(self):
        result = Retriever(build_index(), min_score=0.95).retrieve("¿Cuál es la capital de Mongolia?")
        assert result.verdict is RetrievalVerdict.INSUFFICIENT
        assert not result.hits

    def test_empty_question_is_insufficient(self):
        assert Retriever(build_index()).retrieve("   ").verdict is RetrievalVerdict.INSUFFICIENT

    def test_empty_index_is_insufficient(self):
        empty = VectorIndex(embedder=HashingEmbedder(128))
        assert Retriever(empty).retrieve("cualquier cosa").verdict is RetrievalVerdict.INSUFFICIENT

    def test_context_carries_chunk_ids_for_citation(self):
        result = Retriever(build_index()).retrieve("devoluciones")
        assert all(f"[{cid}]" in result.context for cid in result.sources)

    def test_invalid_min_score_is_rejected(self):
        with pytest.raises(ValueError):
            Retriever(build_index(), min_score=1.5)


class TestGroundedness:
    def test_verbatim_answer_is_fully_supported(self):
        context = "La garantía cubre defectos de fabricación durante doce meses."
        score, unsupported = groundedness(context, context)
        assert score == 1.0 and not unsupported

    def test_invented_sentence_is_named(self):
        context = "La garantía cubre defectos de fabricación durante doce meses."
        answer = (
            "La garantía cubre defectos de fabricación durante doce meses. "
            "Además puede extenderse pagando una prima adicional del quince por ciento."
        )
        score, unsupported = groundedness(answer, context)
        assert len(unsupported) == 1
        assert "prima" in unsupported[0]
        assert score < 1.0

    def test_stopword_only_sentence_asserts_nothing(self):
        assert sentence_support("Y por lo tanto, en esto y con eso.", "") == 1.0

    def test_empty_answer_scores_zero(self):
        assert groundedness("", "algo de contexto") == (0.0, ())


class TestCitations:
    def test_only_retrieved_ids_survive(self):
        assert _extract_citations("Texto [a#0] y [inventado#9].", ["a#0"]) == ["a#0"]

    def test_duplicates_collapse(self):
        assert _extract_citations("[a#0] y [a#0]", ["a#0"]) == ["a#0"]


class TestGroundedAnswerer:
    def _answerer(self, reply: str, **kwargs) -> GroundedAnswerer:
        return GroundedAnswerer(Retriever(build_index()), complete=lambda _: reply, **kwargs)

    def test_supported_answer_with_citation_is_grounded(self):
        answer = self._answerer(
            "Las devoluciones se aceptan dentro de los treinta días posteriores a la entrega. "
            "[devoluciones#0]"
        ).ask("¿Cuántos días para devolver?")
        assert answer.verdict is AnswerVerdict.GROUNDED
        assert answer.sources == ["devoluciones#0"]

    def test_partly_invented_answer_goes_to_review_not_the_bin(self):
        answer = self._answerer(
            "La garantía cubre defectos durante doce meses. "
            "También incluye reemplazo inmediato en cualquier tienda física del país. "
            "[garantia#0]"
        ).ask("¿Qué cubre la garantía?")
        assert answer.verdict is AnswerVerdict.REVIEW
        assert answer.unsupported

    def test_missing_citation_is_held_for_review(self):
        answer = self._answerer(
            "Las devoluciones se aceptan dentro de los treinta días posteriores a la entrega."
        ).ask("¿Cuántos días para devolver?")
        assert answer.verdict is AnswerVerdict.REVIEW
        assert "citation" in answer.reason

    def test_model_refusal_is_respected_not_overridden(self):
        answer = self._answerer("NO_ENCONTRADO").ask("¿Cuántos días para devolver?")
        assert answer.verdict is AnswerVerdict.REFUSED
        assert "insufficient" in answer.reason

    def test_ungrounded_answer_is_refused_outright(self):
        answer = self._answerer(
            "El plazo es de noventa días y aplica únicamente a clientes corporativos "
            "inscritos en el programa anual. [devoluciones#0]"
        ).ask("¿Cuántos días para devolver?")
        assert answer.verdict is AnswerVerdict.REFUSED

    def test_model_is_not_called_when_retrieval_fails(self):
        calls: list[str] = []

        def spy(prompt: str) -> str:
            calls.append(prompt)
            return "lo que sea"

        answerer = GroundedAnswerer(Retriever(build_index(), min_score=0.99), complete=spy)
        assert answerer.ask("¿Capital de Mongolia?").verdict is AnswerVerdict.REFUSED
        assert calls == []

    def test_inverted_thresholds_are_rejected(self):
        with pytest.raises(ValueError):
            self._answerer("x", accept=0.3, review_floor=0.9)
