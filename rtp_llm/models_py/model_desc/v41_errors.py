"""Typed native dispatch failures for the qualified V4.1 worker."""


class V41InputRejected(ValueError):
    """All DP peers rejected this batch before entering model computation."""
