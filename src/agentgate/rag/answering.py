"""Answering, and then checking that the answer is actually supported.

This module is the reason the package exists. Retrieval solves *what the
model reads*; it does not solve *what the model says*. A model handed
correct context can still add a confident sentence that appears nowhere in
it, and that sentence is indistinguishable from the rest to a reader.

So every answer is verified against the retrieved text before it is
returned, and one that cannot be traced back is not delivered as an answer.

A NOTE ON THE MEASURE, because overstating it would defeat the purpose:
groundedness here is computed from lexical overlap between each sentence of
the answer and the retrieved chunks. It reliably catches the common failure
— a model inventing a name, a figure or a rule that is simply not in the
source. It does NOT catch a faithful paraphrase being marked unsupported,
nor a claim that reuses the source's vocabulary while reversing its meaning.
A production system replaces this function with an NLI model or an
LLM-as-judge and keeps everything around it unchanged. The interface is the
part that matters; the measure is swappable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .chunking import split_sentences
from .embedding import tokenize
from .retrieval import RetrievalResult, RetrievalVerdict, Retriever

# Words that carry no evidential weight. Overlap on these means nothing, and
# counting them inflates every score toward "supported".
_STOPWORDS = frozenset(
    """de la que el en y a los del se las por un para con no una su al lo como mas pero sus
    le ya o este si porque esta entre cuando muy sin sobre tambien me hasta hay donde quien
    desde todo nos durante todos uno les ni contra otros ese eso ante ellos e esto mi antes
    algunos que unos yo otro otras otra tanto esa estos mucho quienes nada muchos cual sea
    poco ella estar haber estas estaba estamos algunas algo nosotros the of and to in is it
    for on with as be this that at by an or are from""".split()
)


class AnswerVerdict(str, Enum):
    GROUNDED = "grounded"
    REVIEW = "review"
    REFUSED = "refused"


@dataclass(frozen=True, slots=True)
class Answer:
    verdict: AnswerVerdict
    text: str
    sources: list[str]
    groundedness: float
    reason: str
    unsupported: tuple[str, ...] = ()

    def __str__(self) -> str:
        head = f"[{self.verdict.value:>9}] groundedness {self.groundedness:.2f} — {self.reason}"
        cites = f"\n  sources: {', '.join(self.sources) or 'none'}"
        return head + cites


def _content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 2}


def sentence_support(sentence: str, context: str) -> float:
    """Share of a sentence's content words that appear in the context.

    Content words only. A sentence made entirely of stopwords returns 1.0,
    which is correct: it asserts nothing, so there is nothing to contradict.
    """
    claim = _content_tokens(sentence)
    if not claim:
        return 1.0
    evidence = _content_tokens(context)
    return len(claim & evidence) / len(claim)


def groundedness(answer: str, context: str, per_sentence_floor: float = 0.6) -> tuple[float, tuple[str, ...]]:
    """Score the whole answer and name the sentences that failed.

    Reporting *which* sentence is unsupported is the difference between a
    metric and a usable signal. A single number tells a reviewer that
    something is wrong; the sentence tells them what to look at.
    """
    sentences = split_sentences(answer)
    if not sentences:
        return 0.0, ()

    scores = [(s, sentence_support(s, context)) for s in sentences]
    unsupported = tuple(s for s, score in scores if score < per_sentence_floor)
    overall = sum(score for _, score in scores) / len(scores)
    return round(overall, 3), unsupported


PROMPT = (
    "Responde la pregunta usando EXCLUSIVAMENTE el contexto entregado.\n"
    "Reglas:\n"
    "- Si el contexto no contiene la respuesta, responde exactamente: NO_ENCONTRADO\n"
    "- No agregues informacion que no este en el contexto.\n"
    "- Cita los identificadores entre corchetes de los fragmentos que uses.\n\n"
    "CONTEXTO:\n<<CONTEXT>>\n\n"
    "PREGUNTA: <<QUESTION>>\n"
)

NOT_FOUND = "NO_ENCONTRADO"


class GroundedAnswerer:
    """Retrieve, answer, verify. Three exits, no bypass — as everywhere else.

    ``accept`` and ``review_floor`` mirror the record gate deliberately: an
    engineer who has read one half of this codebase already knows how the
    other half routes.
    """

    def __init__(
        self,
        retriever: Retriever,
        complete: Callable[[str], str],
        accept: float = 0.85,
        review_floor: float = 0.60,
    ) -> None:
        if not 0.0 <= review_floor <= accept <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= review_floor <= accept <= 1")
        self.retriever = retriever
        self._complete = complete
        self.accept = accept
        self.review_floor = review_floor

    def ask(self, question: str) -> Answer:
        retrieval: RetrievalResult = self.retriever.retrieve(question)

        # Refusing before calling the model is not only safer, it is cheaper.
        # Most RAG systems discover this after their first invoice.
        if retrieval.verdict is RetrievalVerdict.INSUFFICIENT:
            return Answer(
                verdict=AnswerVerdict.REFUSED,
                text="No tengo información suficiente para responder eso.",
                sources=[],
                groundedness=0.0,
                reason=f"retrieval: {retrieval.reason}",
            )

        context = retrieval.context
        raw = self._complete(PROMPT.replace("<<CONTEXT>>", context).replace("<<QUESTION>>", question)).strip()

        # A model that follows the refusal instruction is behaving correctly.
        # Treating that as a failure teaches the next iteration to guess.
        if NOT_FOUND in raw.upper():
            return Answer(
                verdict=AnswerVerdict.REFUSED,
                text="No tengo información suficiente para responder eso.",
                sources=[],
                groundedness=0.0,
                reason="model declared the context insufficient",
            )

        cited = _extract_citations(raw, retrieval.sources)
        score, unsupported = groundedness(_strip_citations(raw), context)

        if not cited:
            return Answer(
                verdict=AnswerVerdict.REVIEW,
                text=raw,
                sources=[],
                groundedness=score,
                reason="answer carries no citation — cannot be checked by a reader",
                unsupported=unsupported,
            )

        if score >= self.accept and not unsupported:
            return Answer(AnswerVerdict.GROUNDED, raw, cited, score, f"groundedness {score:.2f} >= {self.accept}")

        # Routing on the average score alone throws away the most useful
        # thing the check produced: WHICH sentences failed. An answer with
        # three good sentences and one invented one averages down into
        # rejection, and a correct answer is discarded because a single
        # clause was fabricated — when a reviewer would simply delete that
        # clause and ship the rest.
        #
        # So refusal is reserved for answers with nothing to salvage. If any
        # sentence stands up, a person decides. Discovered by a test that
        # expected review and got a refusal.
        total = len(split_sentences(_strip_citations(raw)))
        salvageable = total - len(unsupported)

        if salvageable <= 0:
            return Answer(
                AnswerVerdict.REFUSED,
                text="No tengo información suficiente para responder eso.",
                sources=cited,
                groundedness=score,
                reason=f"no sentence is supported by the retrieved context (groundedness {score:.2f})",
                unsupported=unsupported,
            )

        if unsupported:
            return Answer(
                AnswerVerdict.REVIEW,
                raw,
                cited,
                score,
                f"{len(unsupported)} of {total} sentence(s) unsupported — the rest stands",
                unsupported,
            )

        if score >= self.review_floor:
            return Answer(AnswerVerdict.REVIEW, raw, cited, score, f"groundedness {score:.2f} below {self.accept}")

        return Answer(
            AnswerVerdict.REFUSED,
            text="No tengo información suficiente para responder eso.",
            sources=cited,
            groundedness=score,
            reason=f"groundedness {score:.2f} < {self.review_floor}",
            unsupported=unsupported,
        )


_CITATION = re.compile(r"\[([^\]\s]+)\]")


def _extract_citations(answer: str, allowed: list[str]) -> list[str]:
    """Keep only citations that point at chunks actually retrieved.

    A model will happily cite an identifier it never saw. Validating against
    the retrieved set turns the citation from decoration into a check.
    """
    seen: list[str] = []
    for ref in _CITATION.findall(answer):
        if ref in allowed and ref not in seen:
            seen.append(ref)
    return seen


def _strip_citations(answer: str) -> str:
    return _CITATION.sub(" ", answer)
