"""Shared failure for unavailable or incomplete diagnostic evidence."""


class DebugUnavailable(RuntimeError):
    """Required observation failed or is incomplete; callers map it to ERROR."""
