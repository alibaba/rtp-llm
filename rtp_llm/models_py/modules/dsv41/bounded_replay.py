"""Startup-only, opt-in decoder SWA approximation policy."""

import os


def enabled():
    return os.environ.get("DSV41_SWA_BOUNDED_REPLAY", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
