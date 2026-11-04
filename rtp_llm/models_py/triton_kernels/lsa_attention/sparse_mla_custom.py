"""Integration point for the self-written sparse MLA prefill kernel.

The custom kernel is developed out-of-tree (``torch.utils.cpp_extension`` build
exposing ``run(q, kv, idx, out, lse)`` with ``q [T, H, 576] bf16``,
``kv [N, 576] bf16``, ``idx [T, S] int32`` (-1 marks padding) and fp32 outputs
``out [T, H, 512]`` / ``lse [T, H]``).  Point ``LSA_CUSTOM_MLA_MODULE`` at that
module (e.g. ``lsa_custom_mla``) to enable it; otherwise this function delegates
to the FlashMLA sparse path so callers get identical semantics either way.

Status (baseline-log #5.159-#5.168): the SIMT prototype is correct but far from
the prefill target; the tensor-core (tcgen05) kernel is under development.
"""

from __future__ import annotations

import importlib
import os
from typing import Optional, Tuple

import torch

__all__ = ["custom_mla_available", "sparse_mla_custom_fwd"]

_MODULE_ENV = "LSA_CUSTOM_MLA_MODULE"


def _load_custom():
    name = os.environ.get(_MODULE_ENV, "").strip()
    if not name:
        return None
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def custom_mla_available() -> bool:
    mod = _load_custom()
    return mod is not None and hasattr(mod, "run")


def sparse_mla_custom_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    idx: torch.Tensor,
    *,
    kv_lora_rank: int = 512,
    out_dtype: Optional[torch.dtype] = None,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse MLA forward using the custom kernel when configured, else FlashMLA."""
    out_dtype = out_dtype or torch.float32
    mod = _load_custom()
    if mod is not None and hasattr(mod, "run"):
        t, h = q.shape[0], q.shape[1]
        out = torch.empty(t, h, kv_lora_rank, dtype=torch.float32, device=q.device)
        lse = torch.empty(t, h, dtype=torch.float32, device=q.device)
        mod.run(q, kv, idx if idx.dim() == 2 else idx.squeeze(1), out, lse)
        return out.to(out_dtype), lse

    from rtp_llm.models_py.triton_kernels.lsa_attention.sparse_mla_flashmla import (
        sparse_flashmla_fwd,
    )

    return sparse_flashmla_fwd(
        q, kv, idx, kv_lora_rank=kv_lora_rank, out_dtype=out_dtype, scale=scale
    )
