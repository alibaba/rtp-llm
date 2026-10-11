"""Rope application as a swappable module inside an attention impl.

Attention impls that can choose their rope mechanism (the py-flashinfer
prefill family) own a ``RopeApplier`` instead of a hard-wired rope operator:
the applier is the single place that knows how rope is applied, while the
attention core only consumes its output. This mirrors how vLLM and SGLang
select a rotary-embedding module independently from the attention backend, so
a new rope method composes with existing cores instead of requiring a
dedicated implementation.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional

import torch


class RopeApplier(ABC):
    """Applies rope to the packed QKV of one attention phase.

    Contract:
    - ``apply()`` consumes the packed QKV produced by the model layer and
      returns the phase-native rope output.
    - When ``fused_kv_write`` is set and a KV cache is passed to ``apply()``,
      the applier has already written K/V into that cache, and the caller must
      not write it again. Callers that own the KV write (for example CP, whose
      core stores the cache itself) pass ``kv_cache=None``.
    - ``owns_params`` marks appliers whose ``prepare()`` returns the params
      consumed by ``apply()``; otherwise the impl reuses its FMHA params, which
      carry positions and offsets for both the rope and the attention op.
    """

    fused_kv_write: ClassVar[bool] = False
    owns_params: ClassVar[bool] = False

    def prepare(self, attn_inputs: Any, forbid_reallocation: bool = False) -> Any:
        """Build the params consumed by ``apply`` (None when FMHA params are shared)."""
        return None

    def set_params(self, params: Any) -> None:
        """Attach shared FMHA params for appliers that do not own their params."""

    @abstractmethod
    def apply(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[Any] = None,
        params: Optional[Any] = None,
    ) -> Any:
        """Apply rope to the packed QKV and return the core's input."""
        raise NotImplementedError
