"""Orchestration.

The pipeline is intentionally thin. Every rule that matters lives in a
module that can be tested on its own; this file only decides the order.
Orchestration layers that accumulate business logic are how a system stops
being explainable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import ExtractionFailed
from .extractors import Extractor, RawDocument
from .gate import Decision, Gate, Thresholds, Verdict
from .normalize import normalize
from .queue import ReviewQueue
from .schema import Candidate


@dataclass(slots=True)
class RunReport:
    accepted: list[Candidate] = field(default_factory=list)
    rejected: list[Decision] = field(default_factory=list)
    failures: list[ExtractionFailed] = field(default_factory=list)
    queue: ReviewQueue = field(default_factory=ReviewQueue)

    @property
    def processed(self) -> int:
        return len(self.accepted) + len(self.rejected) + len(self.queue) + len(self.failures)

    def summary(self) -> str:
        return (
            f"processed {self.processed} document(s): "
            f"{len(self.accepted)} accepted, "
            f"{len(self.queue)} queued for review, "
            f"{len(self.rejected)} rejected, "
            f"{len(self.failures)} extraction failure(s)"
        )


class Pipeline:
    """extract -> normalize -> gate -> route.

    Nothing writes to a store from inside this class. The caller receives a
    report and decides. Keeping the write outside makes the whole thing
    testable without a database, and makes it obvious where the side effects
    are.
    """

    def __init__(self, extractor: Extractor, thresholds: Thresholds | None = None) -> None:
        self.extractor = extractor
        self.gate = Gate(thresholds)

    def run(self, documents: list[RawDocument]) -> RunReport:
        report = RunReport()

        for document in documents:
            try:
                candidate = self.extractor.extract(document)
            except ExtractionFailed as exc:
                # Loud, recorded, and not confused with a rejection. An
                # extractor that could not read the page is a different
                # problem from a record that failed the rules.
                report.failures.append(exc)
                continue

            decision = self.gate.evaluate(normalize(candidate))

            if decision.verdict is Verdict.ACCEPTED:
                report.accepted.append(decision.candidate)
            elif decision.verdict is Verdict.REVIEW:
                report.queue.enqueue(decision)
            else:
                report.rejected.append(decision)

        return report
