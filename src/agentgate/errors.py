"""Explicit failure types.

The design rule behind this module: an agent that cannot complete its task
says so. It never returns its best guess dressed up as a result.
"""


class AgentGateError(Exception):
    """Base class for every failure this package raises on purpose."""


class ExtractionFailed(AgentGateError):
    """The extractor could not produce a candidate at all.

    This is a *good* outcome. The alternative — inventing a plausible record —
    is the failure mode that actually costs you.
    """

    def __init__(self, source_id: str, reason: str) -> None:
        super().__init__(f"extraction failed for {source_id!r}: {reason}")
        self.source_id = source_id
        self.reason = reason


class SchemaViolation(AgentGateError):
    """A candidate did not survive structural validation."""

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"field {field!r} rejected: {reason}")
        self.field = field
        self.reason = reason


class ProvenanceMissing(AgentGateError):
    """A record reached the gate without a traceable origin.

    Anything written to a store must be answerable to "where did this
    come from?". No provenance, no write.
    """
