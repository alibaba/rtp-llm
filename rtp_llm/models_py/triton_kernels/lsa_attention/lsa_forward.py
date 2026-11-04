"""HFA-style overlapped LSA attention forward (SWA on a side stream + SFA on the main stream).

Implements the paper's Hybrid Sparse Attention execution idea (arXiv:2608.01662
§4.1.1): the dense sink+sliding-window part and the gathered dynamic-sparse part
are dispatched to separate non-blocking streams and recombined through the fused
online-softmax merge. The two parts read disjoint KV slot sets, so the only
cross-stream dependency is the final merge.

Stream contract: the caller's current stream keeps running the sparse part; the
SWA part runs on a cached side stream. A recorded event makes the merge wait for
both parts; the side stream waits on the caller's stream before starting so that
input writes (KV cache writes earlier in the forward) are visible. No CPU sync,
no copies, no hidden allocations inside the timed region.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from rtp_llm.models_py.triton_kernels.lsa_attention.merge_triton import (
    merge_attention_triton,
)
from rtp_llm.models_py.triton_kernels.lsa_attention.sparse_mla_triton import (
    sparse_mla_fwd,
    sparse_mla_fwd_split,
)
from rtp_llm.models_py.triton_kernels.lsa_attention.swa_dense_triton import (
    swa_dense_fwd,
)

_SIDE_STREAM: Optional[torch.cuda.Stream] = None


def _side_stream() -> torch.cuda.Stream:
    global _SIDE_STREAM
    if _SIDE_STREAM is None:
        _SIDE_STREAM = torch.cuda.Stream()
    return _SIDE_STREAM


def _sparse_single(*args, **kwargs):
    """Single-pass sparse MLA, preferring the bit-identical ``sparse_fast`` kernel
    (measured 3-7% faster on B300, baseline-log #5.89) with a fallback to the
    frozen implementation."""
    # Prefer FlashMLA when runnable (SM90+): ~2x faster on the sparse branch even
    # after head padding; the earlier "slower in the composed path" reading was an
    # artifact of the 1008-slot assertion failing and the fallback path re-running
    # the Triton kernel (baseline-log #5.136).
    try:
        from rtp_llm.models_py.triton_kernels.lsa_attention.sparse_mla_flashmla import (
            flashmla_available,
            sparse_flashmla_fwd,
        )

        # Composed-path crossover measured on B300 (#5.136): FlashMLA wins only
        # for larger query counts (prefill-scale); below that its padding/copy
        # overhead loses to sparse_fast on the critical path.
        if flashmla_available() and args and torch.is_tensor(args[0]) and args[0].shape[0] >= 512:
            return sparse_flashmla_fwd(*args, **kwargs)
    except Exception:
        pass
    try:
        from rtp_llm.models_py.triton_kernels.lsa_attention.sparse_fast_triton import (
            sparse_fast_fwd,
        )

        return sparse_fast_fwd(*args, **kwargs)
    except Exception:
        return sparse_mla_fwd(*args, **kwargs)


def lsa_attention(
    q_abs: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    dyn_indices: torch.Tensor,
    *,
    window: int = 1024,
    init_tokens: int = 16,
    kv_lora_rank: int = 512,
    scale: Optional[float] = None,
    out_dtype: Optional[torch.dtype] = None,
    splits: int = 1,
    block_n_swa: int = 32,
    block_n_sparse: int = 64,
    num_stages_swa: int = 2,
    num_stages_sparse: Optional[int] = None,
    overlap: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full LSA attention over ``S_sink ∪ S_swa ∪ S_sparse`` for each query row.

    Args:
        q_abs: ``[T, H, DQ]`` absorbed queries.
        kv: ``[N, DQ]`` latent KV rows (``[c_kv ; k_rope]``).
        positions: ``[T]`` absolute positions of the query rows.
        dyn_indices: ``[T, S]`` or ``[T, 1, S]`` dynamic-sparse slot indices
            (``-1`` padding) produced by the indexer; the sink/window slots are
            supplied by ``window``/``init_tokens`` and must not be repeated here.
        splits: split-KV factor for the sparse part. The composed path measured
            best with ``1`` (single pass) at T=64/192/2048 on B300; ``>1`` helps
            only for large sparse sets (the 2048-slot decode-only path below
            T~112, see build_logs/longcat-lsa-b300-20261017/baseline-log.md
            #5.53/#5.54).
        num_stages_sparse: pipeline stages for the sparse branch; ``None`` picks the
            measured optimum (3 for the single-pass branch, 2 for split), see
            build_logs/longcat-lsa-b300-20261017/baseline-log.md #5.106.
        overlap: run the SWA part on a side stream while the sparse part runs on
            the caller's stream (HFA). ``False`` runs both sequentially.

    Returns:
        ``(out [T, H, DV], lse [T, H])`` in ``out_dtype`` (defaults to
        ``q_abs.dtype``) and float32 natural-log LSE.
    """
    # measured on B300 (#5.106): the single-pass sparse branch prefers 3 stages
    # (prefill -2.7%, spec -2.3%) while the split branch prefers 2 (+1.5% if 3).
    if num_stages_sparse is None:
        num_stages_sparse = 3 if splits <= 1 else 2

    # Whole-attention shortcut: for prefill-scale batches the union slot list
    # (sink + window + dynamic) evaluated by ONE FlashMLA sparse call beats the
    # SWA + sparse + fused-merge composition (-24% at t=2048; MFU 15.6% vs 10.6%;
    # numerically equivalent: out 3.9e-05 / lse 1.4e-06, baseline-log #5.138).
    if q_abs.shape[0] >= 512:
        try:
            from rtp_llm.models_py.triton_kernels.lsa_attention.lsa_attention_flashmla import (
                lsa_attention_flashmla,
                lsa_flashmla_available,
            )

            if lsa_flashmla_available():
                return lsa_attention_flashmla(
                    q_abs, kv, positions, dyn_indices, window=window,
                    init_tokens=init_tokens, kv_lora_rank=kv_lora_rank, scale=scale,
                    out_dtype=out_dtype,
                )
        except Exception:
            pass

    if not overlap:
        o_swa, l_swa = swa_dense_fwd(
            q_abs, kv, positions, window=window, init_tokens=init_tokens,
            kv_lora_rank=kv_lora_rank, scale=scale, out_dtype=out_dtype,
            block_n=block_n_swa, num_stages=num_stages_swa,
        )
        if splits > 1:
            o_sp, l_sp = sparse_mla_fwd_split(
                q_abs, kv, dyn_indices, scale=scale, kv_lora_rank=kv_lora_rank,
                out_dtype=out_dtype, splits=splits, block_n=block_n_sparse,
                num_stages=num_stages_sparse,
            )
        else:
            o_sp, l_sp = _sparse_single(
                q_abs, kv, dyn_indices, scale=scale, kv_lora_rank=kv_lora_rank,
                out_dtype=out_dtype, block_n=block_n_sparse,
                num_stages=num_stages_sparse,
            )
        return merge_attention_triton([(o_swa, l_swa), (o_sp, l_sp)])

    current = torch.cuda.current_stream()
    side = _side_stream()
    ready = torch.cuda.Event()
    swa_done = torch.cuda.Event()
    ready.record(current)
    side.wait_event(ready)
    with torch.cuda.stream(side):
        o_swa, l_swa = swa_dense_fwd(
            q_abs, kv, positions, window=window, init_tokens=init_tokens,
            kv_lora_rank=kv_lora_rank, scale=scale, out_dtype=out_dtype,
            block_n=block_n_swa, num_stages=num_stages_swa,
        )
        swa_done.record(side)
    if splits > 1:
        o_sp, l_sp = sparse_mla_fwd_split(
            q_abs, kv, dyn_indices, scale=scale, kv_lora_rank=kv_lora_rank,
            out_dtype=out_dtype, splits=splits, block_n=block_n_sparse,
            num_stages=num_stages_sparse,
        )
    else:
        o_sp, l_sp = _sparse_single(
            q_abs, kv, dyn_indices, scale=scale, kv_lora_rank=kv_lora_rank,
            out_dtype=out_dtype, block_n=block_n_sparse,
            num_stages=num_stages_sparse,
        )
    current.wait_event(swa_done)
    return merge_attention_triton([(o_swa, l_swa), (o_sp, l_sp)])