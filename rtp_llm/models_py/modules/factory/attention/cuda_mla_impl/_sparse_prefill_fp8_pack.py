"""GLM5 sparse prefill FP8 dispatch helper.

Used by SparseMlaFp8Op._forward_gather and SparseMlaFp8CPOp._attend_gather
when RTP_LLM_GLM5_SPARSE_ATTN_DTYPE=fp8 and USE_GATHER_PATH=1.

Fast path: gather + repack the paged FP8 cache directly to a per-tensor FP8
ragged workspace in a single Triton pass, skipping the BF16 upconvert. Q is
packed by a second Triton kernel that also pre-divides RoPE by qk_scale.

Kernel contract (matches test_flash_mla_sparse_prefill_fp8.py):
  - per-tensor q_scale / k_scale: max|X_nope| / 448
  - Q_RoPE pre-divided by qk_scale to cancel the kernel's uniform multiply
  - 16B slot at [512:528] is padding (kernel does not read)

Called by the non-CP and CP non-sharded gather paths (SparseMlaFp8Op._forward_gather
and SparseMlaFp8CPOp._attend_gather); the CP kv_cache_sharded branch is a
distinct code path not exercised in production and not wired to fp8.
"""

import functools
import os
from typing import Optional

import torch

from rtp_llm.models_py.triton_kernels.sparse_mla.sparse_fp8_prefill_pack import (
    pack_q_656,
    paged_fp8_gather_pack,
)

_ENV = "RTP_LLM_GLM5_SPARSE_ATTN_DTYPE"


@functools.lru_cache(maxsize=1)
def resolve_sparse_attn_dtype() -> str:
    v = os.environ.get(_ENV, "bf16").lower()
    if v not in ("bf16", "fp8"):
        raise ValueError(f"invalid {_ENV}={v!r}; expected bf16|fp8")
    if v == "fp8":
        if not torch.cuda.is_available():
            raise RuntimeError(f"{_ENV}=fp8 requires CUDA")
        cc = torch.cuda.get_device_capability()
        if cc[0] < 10:
            raise RuntimeError(f"{_ENV}=fp8 requires Blackwell (SM100+); got {cc}")
    return v


def sparse_prefill_fp8_from_paged_cache(
    *,
    q_bf16: torch.Tensor,
    paged_u8: torch.Tensor,
    block_table: torch.Tensor,
    workspace_starts: torch.Tensor,
    seq_lens: torch.Tensor,
    batch_size: int,
    total_kv_len: int,
    tokens_per_block: int,
    global_indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    packed_kv_workspace: torch.Tensor,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Direct paged-FP8 gather + per-tensor repack, then flash_mla_sparse_fp8_fwd.

    q_bf16:               [s_q, h_q, 576] bf16 (already has RoPE)
    paged_u8:             [num_blocks, block_size, 656] uint8 paged FP8 cache
    block_table, workspace_starts, seq_lens: gather metadata
    global_indices:       [s_q, 1, topk] int32, already offset into ragged output
    packed_kv_workspace:  [total_kv_len, 656] uint8, per-forward buffer reused
                          across layers (fp8 counterpart of the bf16 fused_kv)
    """
    from flash_mla import flash_mla_sparse_fp8_fwd

    kv_pkg, k_scale = paged_fp8_gather_pack(
        paged_u8,
        block_table,
        workspace_starts,
        seq_lens,
        batch_size,
        total_kv_len,
        tokens_per_block,
        output_u8=packed_kv_workspace,
    )
    q_pkg, q_scale = pack_q_656(q_bf16, k_scale)
    out, _, _ = flash_mla_sparse_fp8_fwd(
        q_pkg,
        kv_pkg,
        global_indices,
        sm_scale,
        d_v=d_v,
        q_scale=q_scale,
        k_scale=k_scale,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )
    return out
