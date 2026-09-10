"""Single switch for ALL Triton fuse kernels in the model_py path.

Master switch lives on ``HWKernelConfig.enable_fuse_kernels`` (default
``True``) and is settable via CLI ``--enable_fuse_kernels`` / env var
``ENABLE_FUSE_KERNELS``. Set to ``False`` to globally bypass to the
unfused baseline path (useful for debugging or precision verification
against the pre-fuse implementation).

Covered fuse paths:
  - Qwen3.5 / Qwen3-Next decoder layer fuses
    (``_fuse_input_norm_quant{,_linear}``, ``_fuse_post_norm_quant{,_moe}``,
    ``_fuse_norm_quant`` for ``Qwen3NextGatedDeltaNet``)
  - ``CausalAttention._fuse_sigmoid_mul_quant`` (F8) AND the bf16
    ``sigmoid_mul_inplace_triton`` path — switch off restores the original
    ``attn_output * torch.sigmoid(gate)`` PyTorch baseline.
  - ``DenseMLP._fuse_silu_quant`` (F2)
  - ``MlaAttention._fuse_kv_a_norm`` / ``_fuse_q_a_norm_mode``
    (DSA-F1a / F1b)
  - ``Indexer._get_logits_head_gate`` (DSA-F3 logits gate)

Not gated (always fused; no off branch):
  - ``flashmla_sparse_impl._apply_input_bmm`` (DSA-F5B
    ``strided_slice_copy_``) and ``_apply_output_bmm`` (DSA-F6a output BMM)
    — cuBLAS bmm is invariant under our layout choice, so the fused
    cuBLAS-strideC path is always preferred.

Callers either:
  * Pass ``hw_kernel_config`` to ``fuse_kernels_enabled(hw_kernel_config)``
    when they hold a reference to it (the canonical path), OR
  * Call ``fuse_kernels_enabled()`` with no arg in modules that don't
    have access to the config — falls back to env var
    ``ENABLE_FUSE_KERNELS`` (default ``True``).
"""

from __future__ import annotations

import os
from typing import Any, Optional


def fuse_kernels_enabled(hw_kernel_config: Optional[Any] = None) -> bool:
    """Return whether Triton fuse kernels should run (vs. unfused baseline).

    Resolution order:
      1. If ``hw_kernel_config`` is provided and has the
         ``enable_fuse_kernels`` attribute, return that.
      2. Otherwise read env var ``ENABLE_FUSE_KERNELS``
         (truthy strings ``1/true/yes/on``); default ``True``.
    """
    if hw_kernel_config is not None and hasattr(
        hw_kernel_config, "enable_fuse_kernels"
    ):
        return bool(hw_kernel_config.enable_fuse_kernels)
    val = os.environ.get("ENABLE_FUSE_KERNELS")
    if val is None:
        return True
    return val.lower() in ("1", "true", "yes", "on")
