"""Attention factory module - handles different attention implementations.

Attention impl selection is driven by:
  1. Registry: AttentionImplEnum (qualname strings, no imports)
  2. Device: get_prefill_mha_priorities() etc. (returns Enum list)
  3. Selector: attn_factory.get_fmha_impl() (lazy load + support() check)
"""

from rtp_llm.models_py.modules.factory.attention.attn_factory import AttnImplFactory
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    FMHAImplBase,
    MlaImplBase,
)

__all__ = [
    "FMHAImplBase",
    "MlaImplBase",
    "AttnImplFactory",
]
