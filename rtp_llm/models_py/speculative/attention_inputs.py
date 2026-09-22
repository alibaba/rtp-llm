"""Model-neutral normalization for speculative attention metadata."""

from collections.abc import Mapping
from typing import Any, Optional

_PREFERRED_CACHE_TAGS = (
    "default",
    "csa_kv",
    "hca_kv",
    "indexer_kv",
    "swa_kv",
    "csa_state",
    "hca_state",
    "indexer_state",
)


def primary_attention_inputs(attention_inputs: Any) -> Optional[Any]:
    """Return one entry carrying fields shared by all cache-group inputs.

    This branch normally passes one attention-input object. Newer revisions
    may pass a tag-keyed mapping whose entries differ only in group-local block
    tables. Prefer a stable known tag before falling back to insertion order.
    """
    if attention_inputs is None:
        return None
    if hasattr(attention_inputs, "attention_inputs"):
        attention_inputs = attention_inputs.attention_inputs
        if attention_inputs is None:
            return None
    if not isinstance(attention_inputs, Mapping):
        return attention_inputs
    if not attention_inputs:
        return None
    for tag in _PREFERRED_CACHE_TAGS:
        if tag in attention_inputs:
            return attention_inputs[tag]
    return next(iter(attention_inputs.values()))
