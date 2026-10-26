"""MiniMax-M3.1 single-token Decode data path on main's tagged cache groups."""

import torch

from rtp_llm.models_py.modules.minimax_m31.kernels.minimax_m31_cache import (
    M31CacheViews,
)
from rtp_llm.models_py.modules.minimax_m31.kernels.minimax_m31_gemma_rope import (
    minimax_m31_gemma_norm_rope_,
)
from rtp_llm.models_py.modules.minimax_m31.kernels.minimax_m31_kv_write import (
    write_m31_nvfp4_rows,
)
from rtp_llm.models_py.modules.minimax_m31.minimax_m31_addressing import (
    m31_decode_addressing,
)
from rtp_llm.models_py.modules.minimax_m31.msa.decode.minimax_m31_sparse_decode import (
    m31_sparse_decode,
)
from rtp_llm.ops.compute_ops import PyAttentionInputs


def m31_decode_projected_attention(
    qkv: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    raw_norms: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    cos_sin_cache: torch.Tensor,
    views: M31CacheViews,
    main_inputs: PyAttentionInputs,
    index_inputs: PyAttentionInputs,
    *,
    norm_eps: float,
    init_blocks: int,
    local_blocks: int,
    output_mxfp8: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Normalize projected rows, write current K/V, then attend selected pages.

    Projection output is caller-owned and may be one fused MXFP8 GEMM result.
    This path uses the official TP1 64Q/4KV/4-index-head geometry.
    """
    batch = qkv.shape[0]
    if qkv.shape != (batch, 72 * 128):
        raise ValueError("M3.1 Decode QKV requires [batch,9216]")
    if index_q.shape != (batch, 4 * 128) or index_k.shape != (batch, 128):
        raise ValueError("M3.1 Decode index projections require 4x128 Q and 128 K")
    if not qkv.is_cuda:
        raise ValueError("M3.1 Decode attention requires CUDA")
    address = m31_decode_addressing(main_inputs, index_inputs)
    if address.positions.shape != (batch,):
        raise ValueError("M3.1 Decode projection rows and requests differ")
    q8 = torch.empty((batch, 64, 128), dtype=torch.float8_e4m3fn, device=qkv.device)
    index_q8 = torch.empty(
        (batch, 4, 128), dtype=torch.float8_e4m3fn, device=qkv.device
    )
    minimax_m31_gemma_norm_rope_(
        qkv,
        index_q,
        index_k,
        raw_norms,
        address.positions,
        cos_sin_cache,
        num_q_heads=64,
        num_kv_heads=4,
        num_index_heads=4,
        eps=norm_eps,
        query_fp8_outputs=(q8, index_q8),
    )
    k = qkv[:, 64 * 128 : 68 * 128].view(batch, 4, 128)
    v = qkv[:, 68 * 128 : 72 * 128].view(batch, 4, 128)
    write_m31_nvfp4_rows(k, v, index_k, address.main_slots, address.index_slots, views)
    return m31_sparse_decode(
        q8,
        index_q8,
        views,
        address.main_table,
        address.index_table,
        address.seq_lens,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        output_mxfp8=output_mxfp8,
        valid_token_mask=address.valid_rows,
    )
