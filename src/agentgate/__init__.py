"""agent-gate — a validation gate for agent-produced records.

The failure mode this package is built around is not the crash. It is the
agent that returns something plausible and wrong, which no try/except will
ever catch. The answer is a gate that every record must pass, with three
exits and no bypass.
"""

from .errors import AgentGateError, ExtractionFailed, ProvenanceMissing, SchemaViolation
from .extractors import Extractor, HeuristicExtractor, ModelExtractor, RawDocument
from .gate import Decision, Gate, Thresholds, Verdict
from .normalize import normalize
from .pipeline import Pipeline, RunReport
from .provenance import Provenance
from .queue import ReviewItem, ReviewQueue
from .schema import Candidate, validate

__version__ = "0.1.0"

__all__ = [
    "AgentGateError",
    "Candidate",
    "Decision",
    "Extractor",
    "ExtractionFailed",
    "Gate",
    "HeuristicExtractor",
    "ModelExtractor",
    "Pipeline",
    "Provenance",
    "ProvenanceMissing",
    "RawDocument",
    "ReviewItem",
    "ReviewQueue",
    "RunReport",
    "SchemaViolation",
    "Thresholds",
    "Verdict",
    "normalize",
    "validate",
]
