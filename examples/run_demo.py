"""Run the pipeline end to end. No API key, no network, no setup.

    python examples/run_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentgate import HeuristicExtractor, Pipeline, Thresholds  # noqa: E402
from sample_inputs import DOCUMENTS  # noqa: E402


def main() -> int:
    pipeline = Pipeline(
        extractor=HeuristicExtractor(),
        thresholds=Thresholds(accept=0.85, review_floor=0.50),
    )
    report = pipeline.run(DOCUMENTS)

    print("=" * 68)
    print(report.summary())
    print("=" * 68)

    print("\nACCEPTED — written without a human in the loop")
    for record in report.accepted:
        print(f"  {record.name}  <{record.contact_email}>  {record.segment}  n={record.headcount}")
        print(f"    origin: {record.provenance.describe()}")

    print("\nREVIEW — held back for a person")
    print(report.queue.report())

    print("\nREJECTED — never reaches the store")
    for decision in report.rejected:
        print(f"  {decision}")

    print("\nEXTRACTION FAILURES — the agent said so instead of guessing")
    for failure in report.failures:
        print(f"  {failure.source_id}: {failure.reason}")

    print(
        "\nNote: six documents in, one written. That ratio is the point. A "
        "pipeline that accepted all six would look more productive and would "
        "be quietly poisoning the database."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
