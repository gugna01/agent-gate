"""A small corpus and a scripted model, so the demo runs offline.

The fake model is written to reproduce three real behaviours: answering
correctly from context, following the refusal instruction, and inventing a
fluent sentence that is not in the source. The third is the one worth
catching.
"""

DOCUMENTS: dict[str, str] = {
    "politica-devoluciones": """
        Las devoluciones se aceptan dentro de los treinta días posteriores a la
        entrega del pedido. El producto debe estar sin uso y conservar su empaque
        original.

        El reembolso se procesa por el mismo medio de pago utilizado en la compra
        y demora entre cinco y diez días hábiles una vez recibido el producto en
        bodega. Los costos de envío de la devolución corren por cuenta del cliente,
        salvo cuando la devolución se origina en un defecto de fabricación.
    """,
    "garantia": """
        La garantía cubre defectos de fabricación durante doce meses desde la
        fecha de compra. No cubre daños por uso indebido, caídas, contacto con
        líquidos ni modificaciones hechas por terceros.

        Para activar la garantía se requiere la factura de compra. El proceso de
        evaluación técnica toma hasta quince días hábiles.
    """,
    "envios": """
        Los envíos a ciudades principales demoran entre dos y cuatro días hábiles.
        Los envíos a zonas rurales pueden demorar hasta ocho días hábiles.

        El despacho se realiza el mismo día para pedidos confirmados antes de las
        dos de la tarde.
    """,
}


def scripted_model(prompt: str) -> str:
    """Deterministic stand-in for an LLM. Behaviour depends on the question."""
    question = prompt.split("PREGUNTA:")[-1].strip().lower()

    if "devolver" in question or "devolucion" in question or "devolución" in question:
        return (
            "Las devoluciones se aceptan dentro de los treinta días posteriores a la "
            "entrega del pedido, y el producto debe estar sin uso y conservar su "
            "empaque original. [politica-devoluciones#0]"
        )

    if "garantía" in question or "garantia" in question:
        # Two true sentences from the source, then one invented. The
        # fabricated clause is fluent, specific and appears nowhere in the
        # corpus — the failure this whole package is built to catch.
        return (
            "La garantía cubre defectos de fabricación durante doce meses desde la fecha "
            "de compra. Para activarla se requiere la factura de compra. "
            "Además, la garantía puede extenderse veinticuatro meses adicionales pagando "
            "una prima equivalente al quince por ciento del valor del producto. [garantia#0]"
        )

    if "envio" in question or "envío" in question or "rural" in question:
        return (
            "Los envíos a zonas rurales pueden demorar hasta ocho días hábiles, "
            "mientras que los envíos a ciudades principales demoran entre dos y "
            "cuatro días hábiles. [envios#0]"
        )

    return "NO_ENCONTRADO"
