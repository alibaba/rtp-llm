"""DSv4 FP8 indexer cache layout."""

import os


def validate_indexer_cache_mode() -> None:
    mode = os.environ.get("DSV4_INDEXER_CACHE_MODE", "fp8")
    if mode != "fp8":
        raise ValueError(f"DSV4_INDEXER_CACHE_MODE supports only fp8, got {mode!r}")


def indexer_entry_bytes() -> int:
    validate_indexer_cache_mode()
    return 132
