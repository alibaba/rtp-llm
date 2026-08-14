"""Switches for optional fused fast paths in the model_py path.

Master switch lives on ``HWKernelConfig.enable_fuse_kernels`` (default
``True``) and is settable via CLI ``--enable_fuse_kernels`` / env var
``ENABLE_FUSE_KERNELS``. Set to ``False`` to bypass registered optional
fast paths to their fallbacks. Some pre-existing CUDA/cuBLAS fused
operations remain enabled and are outside this switch.

``RTP_LLM_GLM5_PREFILL_REFINE`` (default ``False``) is an opt-in switch
for the GLM5 prefill refinements added together. Keep the master switch
enabled and set this switch explicitly for a same-binary strict A/B.

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
  - GLM5 prefill refine gate:
    - sparse-MLA prefill TopK index postprocessing
    - GLM5 FP8 prefill Q-RoPE direct-write into the absorbed-Q output layout
    - CP restore direct-out
    - ``GenericMoeLayer`` fused sigmoid + bias + grouped TopK

Not gated (always fused; no off branch):
  - The absorbed-Q input BMM and ``_apply_output_bmm`` (DSA-F6a output BMM)
    use the cuBLAS stride-out layout. When Q-RoPE direct-write is unavailable,
    the input path retains ``strided_slice_copy_`` as the exact fallback.

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
    """Return whether registered optional fused fast paths should run.

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


def glm5_prefill_refine_enabled(hw_kernel_config: Optional[Any] = None) -> bool:
    """Return whether the GLM5 prefill refinement group should run.

    The refinement gate is nested under ``ENABLE_FUSE_KERNELS``. With the
    master switch enabled, only explicit truthy values for
    ``RTP_LLM_GLM5_PREFILL_REFINE`` enable the refinements. When unset or
    false, the pre-refinement production paths run while preserving previously
    existing fused kernels. The gate is process-wide; for a PD prefill A/B,
    change it only in the prefill role and keep decode fixed.
    """
    if not fuse_kernels_enabled(hw_kernel_config):
        return False
    val = os.environ.get("RTP_LLM_GLM5_PREFILL_REFINE")
    if val is None:
        return False
    return val.lower() in ("1", "true", "yes", "on")
