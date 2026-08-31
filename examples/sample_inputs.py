"""Documents chosen to exercise each exit of the gate."""

DOCUMENTS: list[tuple[str, str]] = [
    # Complete and well formed -> ACCEPTED
    (
        "doc-001",
        """
        Company: Northwind Logistics Inc.
        Site: https://www.northwind-logistics.example/
        Contact: ops@northwind-logistics.example
        Segment: mid_market
        Employees: 320
        """,
    ),
    # Everything valid but headcount is absent. Coverage 4/5 = 0.80, which
    # sits inside the review band: nothing is wrong with it, and nothing
    # confirms it either. This is the case that a pipeline without a queue
    # silently guesses on.
    (
        "doc-002",
        """
        Company: Beacon Analytics
        Site: beacon-analytics.example
        Contact: hola@beacon-analytics.example
        Segment: enterprise
        """,
    ),
    # Malformed email: high field coverage, still garbage -> REJECTED on schema
    (
        "doc-003",
        """
        Company: Halyard Freight LLC
        Site: https://halyard-freight.example
        Contact: not-an-email
        Segment: small_business
        Employees: 45
        """,
    ),
    # No identity to anchor on -> extraction FAILS, loudly
    (
        "doc-004",
        """
        A promising operation in the region. Growing fast.
        Contact: hello@somewhere.example
        """,
    ),
    # Empty document -> extraction FAILS
    ("doc-005", "   "),
    # Only a name -> below the review floor, REJECTED
    ("doc-006", "Company: Cormorant Supply Co."),
]
