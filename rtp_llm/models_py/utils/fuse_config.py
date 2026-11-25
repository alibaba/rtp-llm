"""Single switch for ALL Triton fuse kernels in the model_py path.

Master switch lives on ``HWKernelConfig.enable_fuse_kernels`` (default
``True``) and is settable via CLI ``--enable_fuse_kernels`` / env var
``ENABLE_FUSE_KERNELS``. Set to ``False`` to globally bypass to the
unfused baseline path (useful for debugging or precision verification
against the pre-fuse implementation).

Covered fuse paths:
  - ``DenseMLP._fuse_silu_quant`` (F2)
  - ``MlaAttention._fuse_kv_a_norm`` / ``_fuse_q_a_norm_mode``
    (DSA-F1a / F1b)
  - ``Indexer._get_logits_head_gate`` (DSA-F3 logits gate)

Not gated (always fused; no off branch):
  - ``flashmla_sparse_impl._apply_input_bmm`` (DSA-F5B
    ``strided_slice_copy_``) and ``_apply_output_bmm`` (DSA-F6a output BMM)
    — cuBLAS bmm is invariant under our layout choice, so the fused
    cuBLAS-strideC path is always preferred.

The server resolves the environment variable into ``hw_kernel_config``.
Operators used without a server config use the same default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from rtp_llm.ops import HWKernelConfig


def fuse_kernels_enabled(hw_kernel_config: Optional[HWKernelConfig] = None) -> bool:
    """Read the resolved fusion policy, defaulting to enabled."""
    return hw_kernel_config is None or hw_kernel_config.enable_fuse_kernels
