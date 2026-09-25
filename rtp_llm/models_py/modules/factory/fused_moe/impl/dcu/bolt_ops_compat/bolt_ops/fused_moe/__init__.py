"""``bolt_ops.fused_moe`` compatibility facade backed by aiter itself.

aiter 0.1.5+das185 expects these three helpers from ``bolt_ops.fused_moe``:

- ``moe_activation(activated_out, ffn1_out_2d, activation, is_gated,
  gemm1_alpha, gemm1_limit)``
- ``normalize_moe_activation(activation, is_gated) -> (activation, is_gated)``
- ``moe_activation_output_size(N1, is_gated) -> int``

They map 1:1 onto ``aiter.ops.triton.moe_activation`` internals.
"""

from aiter.ops.triton.moe_activation import (  # noqa: F401
    _apply_activation as moe_activation,
    _normalize_activation_and_gate as normalize_moe_activation,
    adjust_N_for_activation as moe_activation_output_size,
)
