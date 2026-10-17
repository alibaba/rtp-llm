"""Indexer representation metadata, independent of native cache descriptors."""

from enum import Enum


class Dsv4IndexerCacheMode(Enum):
    """Explicit indexer representation, independent of the attention KV dtype."""

    FOLLOW_KV = "follow_kv"
    FP8 = "fp8"
    FP4 = "fp4"
