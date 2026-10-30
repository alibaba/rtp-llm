"""PPU overrides; unchanged behavior is inherited from the shared DSV4 implementation."""

from __future__ import annotations
from rtp_llm.models_py.modules.dsv4.fp8.attention import (
    _use_read_from_pool,
    _use_cp_cache_hit_raw_q_merge,
    _force_cp_cache_hit_raw_q_merge,
    _force_all_cp_raw_q_merge,
    _prefill_cp_overlap_enabled,
    _prefill_cp_async_workspace_reads_enabled,
    _CP_POST_GATHER_STREAMS,
    _CP_GATHER_STREAMS,
    _CP_STREAM_CACHE_LOCK,
    _cuda_device_index,
    _get_process_cuda_stream,
    _get_cp_comm_stream,
    _get_cp_post_gather_stream,
    _flat_1d,
    _build_suffix_pool_slot_mapping,
    _build_suffix_cp_sliced_slot_mapping,
    _V4_FP8_BLOCK_CFG,
    _DSV4_FP8_KV_ENTRY_BYTES,
    _SWA_CP_RR_LOGGED_SITES,
    BIND_KEEP,
    bind_attn_cache,
    _DSV4_FP8_INDEXER_ENTRY_BYTES,
    _FLASH_MLA_SPARSE_Q_CHUNK,
    _repack_v4_fp8_scale_to_int32,
    _get_window_topk_idxs,
    _get_window_topk_idxs_cp,
    _get_window_topk_idxs_varlen,
    SwaPrefillMeta,
    WorkspaceMeta,
    CsaPrefillMeta,
    HcaPrefillMeta,
    PrefillQKV,
)
import json
import logging
import os
import threading
from contextlib import contextmanager, suppress
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union
import deep_gemm  # noqa: E402
import torch
import torch.nn as nn
import torch.nn.functional as F
from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt
from rtp_llm.models_py.modules.dsv4._fused_inv_rope_fp8_quant_triton import (
    fused_inv_rope_fp8_quant,
)
from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_fp8_quant_triton import (
    rmsnorm_fp8_quant_ue8m0,
)
from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import fused_rmsnorm_rope
from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.platforms.ppu.models.dsv4.impl.chunk_env import dsv4_chunk_tokens_from_env
from rtp_llm.models_py.modules.dsv4.cp import (
    _CP_ROLE_MAIN,
    CPContext,
    build_cp_full_prefill_positions,
    cp_actual_owned_kv_lens,
    cp_all_gather_full_varlen,
    cp_freqs_cis_local,
    cp_padded_local_kv_lens,
)
from rtp_llm.models_py.modules.dsv4.fp8._cp_attention_merge import merge_lse_output
from rtp_llm.models_py.modules.dsv4.fp8._cp_attention_shard import (
    build_swa_cp_local_indices,
    prefer_raw_q_merge_attention_conservative,
    remap_topk_to_cp_local,
)
from rtp_llm.models_py.modules.dsv4.fp8._pool_reader import (
    CompressedKPoolReader,
    LocalPoolReader,
    make_compressed_k_pool_reader,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_cp_byte_sliced import (
    CPByteSlicedSlotCompaction,
    build_cp_byte_sliced_slot_compaction,
)
from rtp_llm.platforms.ppu.models.dsv4.impl.fp8.compressor import CompressorFP8, CompressorMeta
from rtp_llm.platforms.ppu.models.dsv4.impl.fp8.indexer import IndexerFP8
from rtp_llm.platforms.ppu.models.dsv4.impl.prefill_workspace import PrefillWorkspace
from rtp_llm.models_py.modules.dsv4.rope import (
    apply_rotary_emb,
    apply_rotary_emb_batched,
    precompute_freqs_cis,
)
from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.models_py.utils.memory import dispose_tensor
from rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils import (
    pool_physical_tokens_per_block as _dsv4_pool_owner_tokens_per_block,
)
from rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils import (
    require_pool_tokens_per_block as _dsv4_pool_tokens_per_block,
)
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import kv_tag_for_compress_ratio
from rtp_llm.models_py.modules.dsv4.fp8.attention import PrefillMeta as _BasePrefillMeta
from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8 as _BaseAttentionFP8
from rtp_llm.models_py.modules.dsv4.fp8.attention import CommitOnlyAttentionFP8 as _BaseCommitOnlyAttentionFP8

def _prepare_wo_a_stacked(
    weight_fp8: torch.Tensor,
    scale_raw: torch.Tensor,
    G: int,
    R: int,
    K: int,
) -> tuple:
    """Stack V4 wo_a ckpt (``[G*R, K]`` fp8 + ``[G*R/128, K/128]`` e8m0fnu)
    into the ``fp8_einsum``-expected layout, computed once at init:

    - weight: ``[G, R, K]`` fp8 contiguous (free view)
    - scale : ``[G, R, K/512]`` int32 UE8M0 packed, MN-major TMA-aligned
      (stride ``(K/512 * tma_R, 1, tma_R)``)

    Cast e8m0fnu → fp32, row-repeat by 128 along R so
    ``get_mn_major_tma_aligned_packed_ue8m0_tensor`` (which operates on
    fp32 ``[*, mn, k/128]``) sees the full [G, R, K/128] grid.  The helper
    floor-log2 bitcasts each block scale and packs 4 UE8M0 bytes per
    int32; output shape ``[G, R, K/512]`` matches
    ``deep_gemm.fp8_einsum(..., recipe=(1, 1, 128))`` expectations."""
    from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor

    w_stk = weight_fp8.view(G, R, K).contiguous()
    scale_fp32 = scale_raw.float().view(G, R // 128, K // 128)
    idx = torch.arange(R, device=scale_raw.device) // 128
    scale_rep = scale_fp32.index_select(-2, idx).contiguous()  # [G, R, K/128]
    s_stk = get_mn_major_tma_aligned_packed_ue8m0_tensor(scale_rep)
    return w_stk, s_stk

def _v4_fp8_linear(w: torch.Tensor, s: torch.Tensor, *, platform_provider=None):
    """Build a CudaFp8DeepGEMMLinear from raw V4 FP8 weight + scale tensors.

    Repacks the UE8M0 ``float8_e8m0fnu`` scale into DeepGEMM's int32
    TMA-aligned packed layout when needed. Framework descriptor path may
    deliver the scale already packed (dtype int32) — we no-op then."""
    from rtp_llm.platforms.ppu.models.dsv4.impl.platform_provider import build_dsv4_fp8_linear

    def _default_factory(weight: torch.Tensor, scale: torch.Tensor):
        if scale.dtype == torch.float8_e8m0fnu:
            scale = _repack_v4_fp8_scale_to_int32(scale)
        # LinearFactory.create_linear_from_weights consumes a (weights_dict,
        # weight_key, scale_key) triple — feed it a one-shot dict so the
        # factory plumbing is unchanged.
        local = {"_w": weight, "_s": scale}
        return LinearFactory.create_linear_from_weights(
            local,
            "_w",
            "_s",
            quant_config=_V4_FP8_BLOCK_CFG,
        )

    assert s is not None, "expected non-null FP8 scale"
    return build_dsv4_fp8_linear(
        _default_factory, w, s, platform_provider=platform_provider
    )

def _v4_fp8_linear_from_dict(weights: dict, weight_key: str, scale_key: str):
    """Backwards-compat bridge over ``_v4_fp8_linear`` for callers that
    still pass a flat dict + keys.  Mutates ``weights[scale_key]`` to the
    packed form so subsequent callers don't repack."""
    w = weights[weight_key]
    s = weights[scale_key]
    if s.dtype == torch.float8_e8m0fnu:
        s = _repack_v4_fp8_scale_to_int32(s)
        weights[scale_key] = s
    return _v4_fp8_linear(w, s)

class PrefillMeta(NamedTuple):
    """Per-call prefill metadata, layer-invariant within a
    ``compress_ratio`` bucket. Built once per (forward, ratio) by
    :meth:`Attention._build_shared_prefill_meta` and broadcast to every
    same-ratio layer via :meth:`Attention._set_prefill_meta_shared`.

    The three sub-metadata fields are mutually exclusive — exactly one
    is non-None per layer, gated by ``compress_ratio``:
      * ``compress_ratio == 0``   → ``swa_meta`` only (SWA-only path)
      * ``compress_ratio == 4``   → ``swa_meta`` + ``csa_meta``
      * ``compress_ratio == 128`` → ``swa_meta`` + ``hca_meta``

    ``swa_meta`` is set on every FP8 layer because all paths still
    write the SWA pool for downstream decode.
    """

    seqlen: int
    seqlen_full: int  # CP-aware (== seqlen when CP off); reused by compress / pool
    rd: int
    device: torch.device
    cp_ctx: Optional[CPContext]
    cp_on: bool
    freqs_cis: torch.Tensor
    topk_idxs: torch.Tensor
    sp_int: int
    any_cont: bool
    row_seqlens_full: torch.Tensor  # [1] long — for SWA pool helpers
    # Phase-1 varlen plumbing — populated by upper-layer broadcast builder.
    # All ``None`` on the standalone (B==1) path so existing logic that
    # consults ``sp_int`` / ``seqlen`` continues to work bit-equally;
    # varlen-aware code paths added in later phases (attention / indexer /
    # compressor) prefer these tensors when present.
    #
    #   batch_size       : B — number of requests packed in this prefill
    #   cu_seqlens       : [B+1] int32/int64 — per-request token offsets
    #                      into the flat [T_total] axis
    #   input_lengths    : [B] int32 — per-request new-token count (S_b)
    #   prefix_lengths   : [B] int32 — per-request prior context length
    #                      (== absolute start_pos of each request, i.e.
    #                      ``sp_per_req``); kept under both names because
    #                      framework plumbing uses ``prefix_lengths`` and
    #                      DSV4 internals use ``sp_per_req``
    #   sp_per_req       : alias of ``prefix_lengths`` cast to int64 on
    #                      device — what the existing SWA pool helpers
    #                      consume
    #   position_ids     : [T_total] int64 — per-token global absolute
    #                      position (RoPE input)
    #   req_id_per_token : [T_total] int32 — request id each token
    #                      belongs to (== ``searchsorted(cu_seqlens, t,
    #                      right=True) - 1``); shared by indexer.prepare
    #                      / compressor.prepare_metadata as ``b_idx`` and
    #                      by the workspace path's per-request slot offsets
    #   max_seqlen_q     : max(input_lengths) — workspace sizing hint
    # ``use_varlen`` is expected to be true for every production FP8 prefill;
    # retained on the metadata object so lower helpers can assert the contract.
    use_varlen: bool = False
    sp_per_req: Optional[torch.Tensor] = None
    cu_seqlens: Optional[torch.Tensor] = None
    batch_size: int = 1
    input_lengths: Optional[torch.Tensor] = None
    prefix_lengths: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    req_id_per_token: Optional[torch.Tensor] = None
    max_seqlen_q: int = 0
    swa_meta: Optional[SwaPrefillMeta] = None
    csa_meta: Optional[CsaPrefillMeta] = None
    hca_meta: Optional[HcaPrefillMeta] = None
    # Per-forward prefill scratch (Q output buffer + CP gather/restore). Filled
    # by ``build_and_propagate_prefill_meta_fp8`` from the forward's local
    # ``PrefillWorkspace``; non-None for every production prefill call.
    workspace: Optional[PrefillWorkspace] = None
    freqs_cis_source_id: int = 0

class AttentionFP8(_BaseAttentionFP8):
    prefill_fast_protocol = "dsv4.fp8-attention.v1"

    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        q_lora_rank: int,
        head_dim: int,
        rope_head_dim: int,
        o_lora_rank: int,
        o_groups: int,
        window_size: int,
        compress_ratio: int,
        compress_rope_theta: float,
        rope_theta: float,
        rope_factor: float,
        beta_fast: int,
        beta_slow: int,
        original_seq_len: int,
        max_batch_size: int,
        max_seq_len: int,
        # Indexer
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        norm_eps: float = 1e-6,
        layer_weights: Optional[Dict[str, torch.Tensor]] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        platform_provider=None,
        indexer_factory=None,
    ):
        """``layer_weights`` is the framework's per-layer dict
        (``ModelWeights.weights[layer_id]``) keyed by ``W.v4_*`` enum.
        Reads ``W.v4_attn_*`` for dense attention weights, ``W.v4_compressor_*``
        for the outer compressor, ``W.v4_indexer_*`` (forwarded) for the
        indexer."""
        super(_BaseAttentionFP8, self).__init__()
        self._platform_provider = platform_provider
        self.layer_id = layer_id
        self.dim = dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.window_size = window_size
        self.compress_ratio = compress_ratio
        self.eps = norm_eps
        self.softmax_scale = head_dim**-0.5
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        # Per-rank head + group counts (S7a). Sharding only kicks in when
        # tp_size > 1; tp_size==1 keeps everything bit-exact unchanged.
        assert (
            n_heads % tp_size == 0
        ), f"n_heads={n_heads} not divisible by tp_size={tp_size}"
        assert (
            o_groups % tp_size == 0
        ), f"o_groups={o_groups} not divisible by tp_size={tp_size}"
        self.n_heads = n_heads // tp_size
        self.n_groups = o_groups // tp_size

        # Slices used to carve TP-local tensors out of the full ckpt.
        n_heads_local = self.n_heads
        n_groups_local = self.n_groups
        wq_b_row_slice = slice(
            tp_rank * n_heads_local * head_dim, (tp_rank + 1) * n_heads_local * head_dim
        )
        wo_a_row_slice = slice(
            tp_rank * n_groups_local * o_lora_rank,
            (tp_rank + 1) * n_groups_local * o_lora_rank,
        )
        wo_b_col_slice = slice(
            tp_rank * n_groups_local * o_lora_rank,
            (tp_rank + 1) * n_groups_local * o_lora_rank,
        )
        attn_sink_slice = slice(tp_rank * n_heads_local, (tp_rank + 1) * n_heads_local)

        from rtp_llm.utils.model_weight import W

        # Q / KV / O — FP8 linears go through LinearFactory →
        # CudaFp8DeepGEMMLinear → DeepGEMM fp8_gemm_nt.
        def _fp8_w_s(w_tag, s_tag, row_slice=None, col_slice=None):
            """Pull (weight, scale) by W tag with optional TP slicing,
            then build a CudaFp8DeepGEMMLinear via ``_v4_fp8_linear``.

            Scale slicing depends on layout:
              * legacy raw UE8M0 [N//128, K//128]: row/col strides are
                block-128 → ``slice.start // 128``.
              * framework packed int32 [N, K//128//4]: N is fully
                expanded (slice by full N stride) and K is packed 4×
                (slice by ``slice.start // 512``)."""
            w = layer_weights[w_tag]
            s = layer_weights[s_tag]
            scale_is_packed_int32 = s.dtype == torch.int32
            if row_slice is not None:
                w = w[row_slice]
                if scale_is_packed_int32:
                    s = s[row_slice]
                else:
                    s = s[row_slice.start // 128 : row_slice.stop // 128]
            if col_slice is not None:
                w = w[:, col_slice]
                if scale_is_packed_int32:
                    assert col_slice.start % 512 == 0 and col_slice.stop % 512 == 0, (
                        f"col_slice {col_slice} not aligned to 512 for "
                        f"packed int32 scale; framework path requires "
                        f"K-slices on 512-byte boundaries"
                    )
                    s = s[:, col_slice.start // 512 : col_slice.stop // 512]
                else:
                    s = s[:, col_slice.start // 128 : col_slice.stop // 128]
            if row_slice is not None or col_slice is not None:
                w = w.contiguous()
                s = s.contiguous()
            return _v4_fp8_linear(w, s, platform_provider=platform_provider)

        self.wq_a = _fp8_w_s(W.v4_attn_wq_a_w, W.v4_attn_wq_a_s)
        # wq_b is row-split along N (n_heads * head_dim)
        self.wq_b = _fp8_w_s(
            W.v4_attn_wq_b_w,
            W.v4_attn_wq_b_s,
            row_slice=wq_b_row_slice if tp_size > 1 else None,
        )
        # MQA single KV head — replicate.
        self.wkv = _fp8_w_s(W.v4_attn_wkv_w, W.v4_attn_wkv_s)

        # wo_a grouped projection: row-split along (n_groups*o_lora_rank).
        # Stored as plain ``[N, K]`` fp8 weight + UE8M0 scale tensors;
        # the ``fp8_einsum`` production path uses the pre-stacked
        # ``_wo_a_stk_w`` / ``_wo_a_stk_s`` buffers below, the BF16
        # fallback path inline-dequants from these via
        # ``dequantize_fp8_weight``.
        assert (n_heads * head_dim) % o_groups == 0
        wo_a_w = layer_weights[W.v4_attn_wo_a_w]
        wo_a_s = layer_weights[W.v4_attn_wo_a_s]
        if tp_size > 1:
            wo_a_w = wo_a_w[wo_a_row_slice].contiguous()
            if wo_a_s.dtype == torch.int32:
                # framework path: scale is already (N, K//128//4) int32
                wo_a_s = wo_a_s[wo_a_row_slice].contiguous()
            else:
                wo_a_s = wo_a_s[
                    wo_a_row_slice.start // 128 : wo_a_row_slice.stop // 128
                ].contiguous()
        self.wo_a_w = wo_a_w
        self.wo_a_s = wo_a_s
        K_local = n_heads_local * head_dim // n_groups_local
        from rtp_llm.platforms.ppu.models.dsv4.impl.platform_provider import build_dsv4_wo_a_fp8_linear

        self.wo_a = build_dsv4_wo_a_fp8_linear(
            lambda *_args, **_kwargs: None,
            wo_a_w,
            wo_a_s,
            groups=n_groups_local,
            k_local=K_local,
            platform_provider=platform_provider,
        )
        if self.wo_a is None:
            _stk_w, _stk_s = _prepare_wo_a_stacked(
                wo_a_w, wo_a_s, n_groups_local, o_lora_rank, K_local
            )
            self.register_buffer("_wo_a_stk_w", _stk_w, persistent=False)
            self.register_buffer("_wo_a_stk_s", _stk_s, persistent=False)
        else:
            self.register_buffer("_wo_a_stk_w", None, persistent=False)
            self.register_buffer("_wo_a_stk_s", None, persistent=False)

        # wo_b row-split along K (cols), all_reduce after forward
        self.wo_b = _fp8_w_s(
            W.v4_attn_wo_b_w,
            W.v4_attn_wo_b_s,
            col_slice=wo_b_col_slice if tp_size > 1 else None,
        )

        # Non-quantized norm weights — plain BF16 tensors (loader cast
        # via compute_dtype).  BF16 dtype is required by
        # FlashInfer RMSNorm (which can produce silent NaNs with fp32). attn_sink loads
        # as fp32 via descriptor data_type.
        self.q_norm = layer_weights[W.v4_attn_q_norm]
        self.kv_norm = layer_weights[W.v4_attn_kv_norm]
        attn_sink_full = layer_weights[W.v4_attn_sink]
        self.attn_sink = (
            attn_sink_full[attn_sink_slice].contiguous()
            if tp_size > 1
            else attn_sink_full
        )

        assert (n_heads * head_dim) % o_groups == 0

        # Compressor + Indexer (only for compressed layers)
        if compress_ratio:
            outer_cmp_weights = {
                "ape": layer_weights[W.v4_compressor_ape],
                "wkv": layer_weights[W.v4_compressor_wkv],
                "wgate": layer_weights[W.v4_compressor_wgate],
                "norm": layer_weights[W.v4_compressor_norm],
            }
            from rtp_llm.platforms.ppu.models.dsv4.impl.fp8.compressor import CompressorFP8

            self.compressor = CompressorFP8(
                dim=dim,
                head_dim=head_dim,
                rope_head_dim=rope_head_dim,
                compress_ratio=compress_ratio,
                max_batch_size=max_batch_size,
                cp_role=_CP_ROLE_MAIN,
                norm_eps=norm_eps,
                compressor_weights=outer_cmp_weights,
                platform_provider=platform_provider,
            )
            self.compressor._profile_label = (
                f"L{layer_id:02d}.csa_main"
                if compress_ratio == 4
                else f"L{layer_id:02d}.hca_main"
            )

            # Phase E5: Compressor.kv_cache is self-managed (was an alias
            # into ``Attention.kv_cache[:, win:]``).  Configure shape here
            # because the Compressor doesn't know ``max_seq_len``.
            self.compressor.configure_kv_cache_shape(max_seq_len // compress_ratio)
            # #50 standalone / warmup fallback: nested indexer compressor
            # shares the INDEXER_KV pool with Indexer; when pool context is
            # absent (warmup, unit tests), keep the T hint available for
            # legacy shape fallbacks.
            if compress_ratio == 4:
                from rtp_llm.platforms.ppu.models.dsv4.impl.fp8.indexer import IndexerFP8

                self.indexer = (indexer_factory or IndexerFP8)(
                    dim=dim,
                    q_lora_rank=q_lora_rank,
                    index_n_heads=index_n_heads,
                    index_head_dim=index_head_dim,
                    rope_head_dim=rope_head_dim,
                    index_topk=index_topk,
                    compress_ratio=compress_ratio,
                    max_batch_size=max_batch_size,
                    max_seq_len=max_seq_len,
                    norm_eps=norm_eps,
                    layer_weights=layer_weights,
                    platform_provider=platform_provider,
                )

                # Configure nested indexer compressor shape hint so warmup
                # (where pool context is absent) can still allocate an
                # ephemeral zero kv_cache for the write at the tail of
                # ``_forward_scalar_impl``.
                if self.indexer.compressor is not None:
                    self.indexer.compressor._profile_label = (
                        f"L{layer_id:02d}.csa_nested_indexer"
                    )
                    self.indexer.compressor.configure_kv_cache_shape(
                        max_seq_len // compress_ratio
                    )
            else:
                self.indexer = None
        else:
            self.compressor = None
            self.indexer = None

        # Phase E5b: kv_cache register_buffer retired.  KV storage now lives
        # exclusively in the framework BlockPools (SWA_KV + CSA_KV / HCA_KV);
        # prefill reads via ``_gather_kv_cache_dense_from_pool``, prefill
        # writes via ``_prefill_write_swa_fp8_paged`` + ``_prefill_paged_write_compressed``,
        # decode reads/writes via ``forward_decode``'s paged paths.
        kv_cache_size = window_size + (
            max_seq_len // compress_ratio if compress_ratio else 0
        )

        # Per-layer freqs_cis: SWA-only uses base rope_theta with no yarn,
        # CSA/HCA uses compress_rope_theta with yarn (when original_seq_len > 0).
        # Store scalars so we can re-compute after `to_empty`(meta) — otherwise
        # the buffer ends up all zeros.
        if compress_ratio:
            self._rope_base = compress_rope_theta
            self._rope_o_seq_len = original_seq_len
        else:
            self._rope_base = rope_theta
            self._rope_o_seq_len = 0
        self._rope_factor = rope_factor
        self._rope_beta_fast = beta_fast
        self._rope_beta_slow = beta_slow
        self._rope_dim = rope_head_dim
        self._rope_max_seq_len = max_seq_len
        freqs_cis = precompute_freqs_cis(
            rope_head_dim,
            max_seq_len,
            self._rope_o_seq_len,
            self._rope_base,
            rope_factor,
            beta_fast,
            beta_slow,
        )
        # Phase G: plain attr (not register_buffer).  `init_rope_cache(device)`
        # recomputes + moves to the real device after meta-to-device
        # materialization — that's the authoritative placement path; no
        # automatic `.to(device)` semantics needed.
        self.freqs_cis = freqs_cis

        # Iter2: persistent FP8 sparse decode op. Carries the FlashMLA
        # ``sched_meta`` cache (per single/dual-pool flag) so the planner
        # only runs once per layer-type per process — vs the iter1' default
        # of one planner setup per layer per step. Built lazily on the
        # first FP8 decode call so BF16-KV runs don't pay the import cost.
        self._fp8_decode_op: Optional[Any] = None

        # CP context bound per-forward by V4Transformer.  None = no CP.
        self._cp_ctx: Optional[CPContext] = None

        # Shared prefill meta (compress_ratio bucket; not layer-specific).
        # V4Transformer.forward_layers builds one per ratio in use, sets it
        # on every layer's attention via ``_set_prefill_meta_shared``, and
        # clears at end-of-forward. None ⇒ standalone path / decode →
        # ``_prefill_common_setup`` falls back to per-call build.
        self._prefill_meta_shared: Optional["PrefillMeta"] = None

        # qwen3-style: framework KVCache handle + per-request block tables
        # flow in as kwargs on ``forward`` / ``forward_decode``; these two
        # attrs are stashed only for the duration of a single forward call
        # (try/finally clears them), so pool views resolve via
        # ``self._kv_cache.get_layer_cache(layer_id, tag).kv_cache_base``
        # inside helpers without threading the handle through every method.
        self._kv_cache: Optional[Any] = None
        self._block_tables_by_type: Optional[Dict[str, torch.Tensor]] = None

        # Precomputed pool specs per cache tag — (vec_dtype, vec_dim) used
        # to reinterpret the raw ``[num_blocks, stride_elems]`` framework
        # pool tensor as a typed ``[total_slots, vec_dim]`` flat view.
        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
            CSA_KV,
            CSA_STATE,
            HCA_KV,
            HCA_STATE,
            INDEXER_KV,
            INDEXER_STATE,
            SWA_KV,
        )

        idx_hd = index_head_dim
        coff_csa = 2  # CSA overlap=True
        coff_idx = 2  # Indexer's nested Compressor overlap=True
        # HCA uses coff=1 (overlap=False) so HCA_STATE vec_dim = 2*head_dim.
        kv_spec = (torch.uint8, _DSV4_FP8_KV_ENTRY_BYTES)
        indexer_kv_spec = (
            torch.uint8,
            getattr(
                indexer_factory, "CACHE_ENTRY_BYTES", _DSV4_FP8_INDEXER_ENTRY_BYTES
            ),
        )
        self._pool_spec: Dict[str, tuple] = {
            SWA_KV: kv_spec,
            CSA_KV: kv_spec,
            HCA_KV: kv_spec,
            INDEXER_KV: indexer_kv_spec,
            CSA_STATE: (torch.float32, 2 * coff_csa * head_dim),
            HCA_STATE: (torch.float32, 2 * head_dim),
            INDEXER_STATE: (torch.float32, 2 * coff_idx * idx_hd),
        }

    def _wo_a_einsum_from_fp8(
        self, o_fp8: torch.Tensor, o_scale: torch.Tensor, B: int, S: int
    ) -> torch.Tensor:
        """One ``fp8_einsum`` call on pre-quantized activations.

        ``fused_inv_rope_fp8_quant`` emits ``(o_fp8 [M, G, K], o_scale
        [M, G, K/512])`` in the exact layout ``deep_gemm.fp8_einsum``
        consumes, so the wo_a projection is a single einsum launch.
        Matches vLLM ``deepseek_v4_attention.py:325`` (same
        ``"bhr,hdr->bhd"`` + recipe ``(1, 1, 128)`` for SM100 UE8M0)."""
        assert self.wo_a is None, "provider wo_a consumes BF16 grouped input"
        M, G, _K = o_fp8.shape
        R = self.o_lora_rank
        out = torch.empty(M, G, R, dtype=torch.bfloat16, device=o_fp8.device)
        deep_gemm.fp8_einsum(
            "bhr,hdr->bhd",
            (o_fp8, o_scale),
            (self._wo_a_stk_w, self._wo_a_stk_s),
            out,
            recipe=(1, 1, 128),
        )
        return out.view(B, S, G, R)

    def _wo_a_from_bf16(
        self,
        o: torch.Tensor,
        freqs_cis: torch.Tensor,
        B: int,
        S: int,
    ) -> torch.Tensor:
        """Inverse-RoPE then invoke a provider-owned grouped FP8 projection."""
        assert self.wo_a is not None
        o_4d = o.view(B, S, self.n_heads, self.head_dim)
        rope = o_4d[..., -self.rope_head_dim :]
        if freqs_cis.dim() == 2 and int(freqs_cis.shape[0]) == B:
            apply_rotary_emb_batched(rope, freqs_cis, inverse=True)
        else:
            apply_rotary_emb(
                rope,
                freqs_cis.reshape(-1, freqs_cis.shape[-1]).contiguous(),
                inverse=True,
            )
        grouped = o_4d.reshape(B, S, self.n_groups, -1)
        return self.wo_a(grouped)

    def _forward_decode_body(
        self,
        x: torch.Tensor,  # [B, 1, dim] bf16  (q_len=1 pure decode)
        attn_metadata: "DSv4DecodeAttnMetadataFP8",  # type: ignore[name-defined]
    ) -> torch.Tensor:
        """Decode attention body — thin dispatcher mirroring ``_forward_prefill``.

        Pipeline:
          1. Q/KV + per-request partial RoPE (``decode_compute_qkv``).
          2. FP8 SWA pool write (``decode_write_swa_fp8``).
          3. Per-``compress_ratio`` body:
             * ``0``   → :meth:`_forward_decode_swa_only`
             * ``4``   → :meth:`_forward_decode_csa`   (indexer + compressor)
             * ``128`` → :meth:`_forward_decode_hca`   (compressor, dense idx)
          4. Output projection (``decode_output_proj``).

        Compressor / Indexer freqs_cis is bound lazily on first call
        (pool context is set in :meth:`forward_decode`'s try/finally).
        """
        from rtp_llm.platforms.ppu.models.dsv4.impl.fp8.decode.compute_qkv import decode_compute_qkv
        from rtp_llm.platforms.ppu.models.dsv4.impl.fp8.decode.output_proj import decode_output_proj

        bsz, q_len, _ = x.size()
        T = bsz * q_len
        start_pos = attn_metadata.start_pos[:bsz]  # [bsz] int32
        position_ids = attn_metadata.position_ids[:T]  # [T] int32

        self._ensure_freqs_cis_bound()
        qkv = decode_compute_qkv(self, x, position_ids)
        with suppress(Exception):
            from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
                CSA_KV,
                HCA_KV,
                INDEXER_KV,
                SWA_KV,
            )

        self._decode_write_swa_fp8(qkv.kv, bsz, q_len, attn_metadata)

        if self.compress_ratio == 0:
            o = self._forward_decode_swa_only(qkv.q, bsz, q_len, attn_metadata)
        elif self.compress_ratio == 4:
            o = self._forward_decode_csa(
                x,
                qkv,
                bsz,
                q_len,
                start_pos,
                position_ids,
                attn_metadata,
            )
        elif self.compress_ratio == 128:
            o = self._forward_decode_hca(
                x,
                qkv,
                bsz,
                q_len,
                start_pos,
                position_ids,
                attn_metadata,
            )
        else:
            raise AssertionError(f"unknown compress_ratio={self.compress_ratio}")

        return decode_output_proj(self, o, qkv.freqs_cis, bsz, q_len)

    def _decode_compressor_meta_from_metadata(
        self,
        attn_metadata: "DSv4DecodeAttnMetadataFP8",  # type: ignore[name-defined]
        *,
        state_attn_type: str,
        kv_attn_type: str,
        bsz: int,
        q_len: int,
    ) -> CompressorMeta:
        """Return a CompressorMeta view backed by step-level buildmeta.

        The slot tensors are prepared once per decode step. Per-layer code
        only slices stable prefixes and then launches the compressor kernels.
        """
        assert attn_metadata.req_id_per_token is not None
        assert attn_metadata.req_id_per_token_long is not None
        assert attn_metadata.decode_seq_start_per_req is not None
        assert attn_metadata.decode_cu_seq_per_req is not None
        state_slots = attn_metadata.compressor_state_slot_mappings.get(state_attn_type)
        kv_slots = attn_metadata.pool_write_slot_mappings.get(kv_attn_type)
        assert state_slots is not None and kv_slots is not None
        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
            CSA_KV,
            HCA_KV,
            INDEXER_KV,
        )

        ratio_by_kv = {CSA_KV: 4, INDEXER_KV: 4, HCA_KV: 128}
        ratio = ratio_by_kv.get(kv_attn_type)
        compressed_lens_per_token = (
            attn_metadata.compressed_lens_per_token[ratio][:bsz, :q_len]
            if ratio in attn_metadata.compressed_lens_per_token
            else None
        )
        T = bsz * q_len
        positions = attn_metadata.position_ids_long[:T]
        b_idx = attn_metadata.req_id_per_token_long[:T]
        return CompressorMeta(
            positions=positions,
            b_idx=b_idx,
            state_slots=state_slots[:T],
            kv_slots=kv_slots[:T],
            token_to_req=attn_metadata.req_id_per_token[:T],
            has_prefix=True,
            is_batched=q_len > 1,
            seq_start_per_req=attn_metadata.decode_seq_start_per_req[:bsz],
            cu_seq_per_req=attn_metadata.decode_cu_seq_per_req[: bsz + 1],
            compressed_lens_per_token=compressed_lens_per_token,
        )

    def _forward_decode_swa_only(
        self,
        q: torch.Tensor,  # [B, 1, H, D]
        bsz: int,
        q_len: int,
        attn_metadata: "DSv4DecodeAttnMetadataFP8",  # type: ignore[name-defined]
    ) -> torch.Tensor:
        """SWA-only layer (compress_ratio == 0) — one FlashMLA call over
        the FP8 SWA pool using per-request global slot ids translated
        from ``swa_abs_idx`` through the SWA block table."""
        from rtp_llm.platforms.ppu.models.dsv4.impl.fp8.decode.attention_kernels import attn_fp8_swa_paged
        from rtp_llm.platforms.ppu.models.dsv4.impl.fp8.decode.decode_attn_metadata import get_or_build_sched_meta
        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import SWA_KV

        swa_pool_3d = self._pool_view_3d_fp8(SWA_KV)
        swa_pool_bt = (
            attn_metadata.pool_block_tables.get(SWA_KV)
            if attn_metadata.pool_block_tables
            else None
        )
        assert (
            swa_pool_3d is not None
            and swa_pool_bt is not None
            and swa_pool_bt.numel() > 0
            and attn_metadata.swa_abs_idx is not None
        ), (
            f"[DSV4 decode SWA-only] FP8 pool + block table + swa_abs_idx "
            f"required (layer={self.layer_id}); "
            f"kv_cache_bound={self._kv_cache is not None}, "
            f"pool_3d_ok={swa_pool_3d is not None}, "
            f"pool_block_tables_keys={list(attn_metadata.pool_block_tables.keys()) if attn_metadata.pool_block_tables else None}, "
            f"swa_pool_bt_shape={tuple(swa_pool_bt.shape) if swa_pool_bt is not None else None}, "
            f"swa_abs_idx_shape={tuple(attn_metadata.swa_abs_idx.shape) if attn_metadata.swa_abs_idx is not None else None}"
        )
        win = self.window_size
        T = bsz * q_len
        # FlashMLA's sparse FP8 kernel reads the packed pool via direct
        # ``pool[indices[i]]`` addressing — no block-table indirection on
        # the kernel side — so ``indices`` MUST be global slot ids, not
        # abs positions. Mirrors the CSA/HCA dual-pool path below.
        assert attn_metadata.swa_global_slots is not None
        swa_global = attn_metadata.swa_global_slots[:T]
        swa_topk_3d = swa_global.view(bsz, q_len, win).contiguous()

        sched_meta = get_or_build_sched_meta(
            attn_metadata,
            batch_size=bsz,
            q_len=q_len,
            num_heads=self.n_heads,
            topk=self.window_size,
            extra_attn_type=None,
            extra_index_width=None,
            platform_provider=self._platform_provider,
        )
        # opt_flash_mla: pass the per-request SWA effective length so FlashMLA
        # scans only ``min(window, seq_len)`` instead of the full window width.
        swa_topk_length = (
            attn_metadata.swa_topk_length[:bsz]
            if attn_metadata.swa_topk_length is not None
            else None
        )
        return attn_fp8_swa_paged(
            q=q,
            swa_pool_3d=swa_pool_3d,
            attn_sink=self.attn_sink,
            swa_topk_3d=swa_topk_3d,
            swa_block_table=swa_pool_bt[:bsz],
            sched_meta=sched_meta,
            fp8_op=self._get_fp8_decode_op(),
            topk_length=swa_topk_length,
            platform_provider=self._platform_provider,
        )

    def _decode_update_indexer(
        self, x, qr, bsz, q_len, start_pos, position_ids, attn_metadata
    ):
        """Update only the Indexer pools and raw compressed indices."""
        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
            INDEXER_KV,
            INDEXER_STATE,
        )

        assert self.indexer is not None, "CSA layer must have an indexer"
        meta = self._decode_compressor_meta_from_metadata(
            attn_metadata,
            state_attn_type=INDEXER_STATE,
            kv_attn_type=INDEXER_KV,
            bsz=bsz,
            q_len=q_len,
        )
        self.indexer.forward_decode_vectorized(
            x,
            qr,
            start_pos,
            attn_metadata.topk_buffer_compressed[:bsz],
            position_ids=position_ids,
            compressor_meta=meta,
        )

    def _decode_update_compressor(
        self, x, bsz, q_len, start_pos, position_ids, attn_metadata
    ):
        """Update the main CSA/HCA pools independently of Q and the Indexer."""
        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
            CSA_KV,
            CSA_STATE,
            HCA_KV,
            HCA_STATE,
        )

        assert self.compress_ratio in (4, 128)
        kv_tag, state_tag = (
            (CSA_KV, CSA_STATE) if self.compress_ratio == 4 else (HCA_KV, HCA_STATE)
        )
        meta = self._decode_compressor_meta_from_metadata(
            attn_metadata,
            state_attn_type=state_tag,
            kv_attn_type=kv_tag,
            bsz=bsz,
            q_len=q_len,
        )
        self.compressor.forward_decode_vectorized(
            x, start_pos, meta=meta, position_ids=position_ids
        )

    def _forward_decode_csa(
        self,
        x: torch.Tensor,
        qkv: "DecodeQKV",  # type: ignore[name-defined]
        bsz: int,
        q_len: int,
        start_pos: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: "DSv4DecodeAttnMetadataFP8",  # type: ignore[name-defined]
    ) -> torch.Tensor:
        """CSA layer (compress_ratio == 4). Indexer + main compressor
        both scatter into their pools; the indexer's topk buffer holds
        raw (pre +win) compressed local indices which the shared
        dual-pool epilogue consumes."""
        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import CSA_KV

        self._decode_update_indexer(
            x, qkv.qr, bsz, q_len, start_pos, position_ids, attn_metadata
        )
        self._decode_update_compressor(
            x, bsz, q_len, start_pos, position_ids, attn_metadata
        )
        # CSA cmp_local_raw = indexer's raw indices (the +win offset is
        # added later inside the epilogue's translate path).
        cmp_local_raw = attn_metadata.topk_buffer_compressed[:bsz]
        return self._forward_decode_compressed(
            qkv.q,
            cmp_local_raw,
            bsz,
            q_len,
            attn_metadata,
            cmp_attn_type=CSA_KV,
        )

    def _forward_decode_hca(
        self,
        x: torch.Tensor,
        qkv: "DecodeQKV",  # type: ignore[name-defined]
        bsz: int,
        q_len: int,
        start_pos: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: "DSv4DecodeAttnMetadataFP8",  # type: ignore[name-defined]
    ) -> torch.Tensor:
        """HCA layer (compress_ratio == 128). Compressor writes
        HCA_KV / HCA_STATE (no indexer). ``cmp_local_raw`` is the dense
        idx precomputed once per step by
        ``update_decode_metadata_in_place._build_dense_compressed_idxs``
        (reused across all HCA layers via ``topk_total_by_ratio[128]``)."""
        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import HCA_KV

        assert self.indexer is None, "HCA layer must not have an indexer"
        self._decode_update_compressor(
            x, bsz, q_len, start_pos, position_ids, attn_metadata
        )
        win = self.window_size
        tt_h = attn_metadata.topk_total_by_ratio.get(int(self.compress_ratio))
        assert tt_h is not None, (
            f"[DSV4 decode HCA] topk_total_by_ratio[{int(self.compress_ratio)}] "
            f"missing (layer={self.layer_id})"
        )
        cmp_local_raw = tt_h[:bsz, :, win:]
        return self._forward_decode_compressed(
            qkv.q,
            cmp_local_raw,
            bsz,
            q_len,
            attn_metadata,
            cmp_attn_type=HCA_KV,
        )

    def _forward_decode_compressed(
        self,
        q: torch.Tensor,  # [B, 1, H, D]
        cmp_local_raw: torch.Tensor,  # [B, 1, K_cmp] int32 pool-local idx
        bsz: int,
        q_len: int,
        attn_metadata: "DSv4DecodeAttnMetadataFP8",  # type: ignore[name-defined]
        cmp_attn_type: str,
    ) -> torch.Tensor:
        """Shared CSA/HCA epilogue: translate pool-local → global slots
        for both SWA and compressed pools, then one dual-pool FlashMLA
        call (``extra_k_cache`` + ``extra_indices_in_kvcache`` merges
        softmax in-kernel; mirrors vLLM ``deepseek_v4_attention.py:849-865``)."""
        from rtp_llm.platforms.ppu.models.dsv4.impl.fp8.decode.attention_kernels import attn_fp8_dual_paged
        from rtp_llm.platforms.ppu.models.dsv4.impl.fp8.decode.decode_attn_metadata import get_or_build_sched_meta
        from rtp_llm.models_py.modules.dsv4.fp8.decode.paged_topk_translator import (
            translate_local_to_global_slots,
        )
        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import SWA_KV

        win = self.window_size
        T = bsz * q_len
        K_cmp = cmp_local_raw.shape[-1]

        swa_pool_3d = self._pool_view_3d_fp8(SWA_KV)
        swa_pool_bt = attn_metadata.pool_block_tables.get(SWA_KV)
        cmp_pool_3d = self._pool_view_3d_fp8(cmp_attn_type)
        cmp_pool_bt = attn_metadata.pool_block_tables.get(cmp_attn_type)
        assert (
            swa_pool_3d is not None
            and cmp_pool_3d is not None
            and swa_pool_bt is not None
            and swa_pool_bt.numel() > 0
            and cmp_pool_bt is not None
            and cmp_pool_bt.numel() > 0
            and attn_metadata.swa_abs_idx is not None
        ), (
            f"[DSV4 decode compressed] FP8 SWA + cmp pools + block tables "
            f"required (layer={self.layer_id}, cmp_attn_type={cmp_attn_type}); "
            f"kv_cache_bound={self._kv_cache is not None}"
        )

        # Translate SWA local → global slots (iter3.3: metadata caches
        # the result once per step, shared across all 43 layers).
        assert attn_metadata.req_id_per_token is not None
        assert attn_metadata.swa_global_slots is not None
        req_id = attn_metadata.req_id_per_token[:T]
        swa_global = attn_metadata.swa_global_slots[:T]

        # Translate compressed local → global slots. HCA layers share a
        # precomputed ``hca_cmp_global_slots`` (dense idx input is
        # identical across HCA layers); CSA layers must translate per-
        # layer because ``topk_buffer_compressed`` is populated by the
        # per-layer indexer.
        if self.indexer is None and attn_metadata.hca_cmp_global_slots is not None:
            cmp_global = attn_metadata.hca_cmp_global_slots[:T]
        else:
            cmp_tokens_per_block = int(
                attn_metadata.paged_pool_tokens_per_block[cmp_attn_type]
            ) // int(self.compress_ratio)
            cmp_global = translate_local_to_global_slots(
                req_id,
                cmp_pool_bt[:bsz],
                cmp_local_raw.reshape(T, K_cmp),
                entries_per_block=self._pool_entries_per_block(cmp_attn_type),
                tokens_per_block_for_block_table=cmp_tokens_per_block,
            )

        # Iter3.2: swa_global / cmp_global are int32 from the translator,
        # so only ``.view + .contiguous`` is needed for FlashMLA's
        # [B, q_len, K] contract.
        swa_topk_3d = swa_global.view(bsz, q_len, win).contiguous()
        cmp_topk_3d = cmp_global.view(bsz, q_len, K_cmp).contiguous()

        sched_meta = get_or_build_sched_meta(
            attn_metadata,
            batch_size=bsz,
            q_len=q_len,
            num_heads=self.n_heads,
            topk=win,
            extra_attn_type=cmp_attn_type,
            extra_index_width=K_cmp,
            platform_provider=self._platform_provider,
        )
        # opt_flash_mla: pass per-request effective lengths so FlashMLA scans
        # only the true SWA / compressed widths. ``extra_topk_length`` keyed by
        # this layer's compress_ratio (4=CSA capped at index_topk, 128=HCA dense)
        # shrinks the HCA scan from the capture width (e.g. 8192) to seq_len//128.
        swa_topk_length = (
            attn_metadata.swa_topk_length[:bsz]
            if attn_metadata.swa_topk_length is not None
            else None
        )
        cmp_len_buf = attn_metadata.compressed_topk_length_by_ratio.get(
            int(self.compress_ratio)
        )
        extra_topk_length = cmp_len_buf[:bsz] if cmp_len_buf is not None else None
        return attn_fp8_dual_paged(
            q=q,
            swa_pool_3d=swa_pool_3d,
            cmp_pool_3d=cmp_pool_3d,
            compress_ratio=int(self.compress_ratio),
            cmp_tokens_per_block=int(
                self._kv_cache.get_kernel_seq_size_per_block(cmp_attn_type)
            ),
            attn_sink=self.attn_sink,
            swa_topk_3d=swa_topk_3d,
            cmp_topk_3d=cmp_topk_3d,
            swa_block_table=swa_pool_bt[:bsz],
            sched_meta=sched_meta,
            fp8_op=self._get_fp8_decode_op(),
            topk_length=swa_topk_length,
            extra_topk_length=extra_topk_length,
            platform_provider=self._platform_provider,
        )

    def _should_overlap_cp_for_prefill(self, common: PrefillMeta) -> bool:
        """Per-call gate for the CP-overlap orchestrator.

        All conditions must hold:
          * ``DSV4_PREFILL_CP_OVERLAP=1`` (default off — baseline path);
          * CP is actually active (``cp_size > 1``); no NCCL gather to
            overlap with otherwise;
          * prefill tensors are CUDA-backed — CPU / sync-reference CP
            should keep using the baseline path;
          * not inside a CUDA-graph capture — NCCL collectives are not
            capturable on this branch and ``cp_all_gather_full_async``
            calls ``work.wait()``;
          * the layer has a compressor (``compress_ratio > 0``) —
            SWA-only layers (ratio == 0) have nothing to overlap.
        """
        if not _prefill_cp_overlap_enabled():
            return False
        if not common.cp_on or common.cp_ctx is None or common.cp_ctx.cp_size <= 1:
            return False
        if self.compress_ratio == 0:
            return False
        if self.compress_ratio not in (4, 128):
            return False
        if common.device.type != "cuda":
            return False
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return False
        return True

    def _cp_kv_cache_sharded(self, common: Optional[PrefillMeta] = None) -> bool:
        cp_ctx = getattr(self, "_cp_ctx", None) or getattr(common, "cp_ctx", None)
        return bool(getattr(cp_ctx, "kv_cache_sharded", False))

    def _should_async_workspace_reads_for_prefill(self, common: PrefillMeta) -> bool:
        """Whether workspace cache reads may use side-stream async restore.

        This shares the main prefill-CP-overlap gate and only applies to the
        page-rr / KV-cache-sharded path. With ``DSV4_PREFILL_CP_OVERLAP=0``,
        workspace reads keep the original synchronous gather/restore/write
        ordering even when ``--prefill_cp_kv_cache_sharded=1`` is set.
        """
        if not _prefill_cp_async_workspace_reads_enabled():
            return False
        return self._should_overlap_cp_for_prefill(
            common
        ) and self._cp_kv_cache_sharded(common)

    def _forward_prefill(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        shared_input_quant: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Prefill body. ``x`` is flat ``[T, dim]``; output flat ``[T, dim]``.

        Three mutually-exclusive paths gated by ``compress_ratio``:
          * 0   → :meth:`_forward_prefill_swa_only`
          * 4   → :meth:`_forward_prefill_csa`   (indexer + compressor)
          * 128 → :meth:`_forward_prefill_hca`   (compressor, dense idx)

        All three share the same prologue: ``_prefill_common_setup`` →
        ``_prefill_compute_qkv`` → ``_prefill_write_swa_fp8_paged``.
        ``csa`` / ``hca`` additionally share the ``[sliding | compressed]``
        kv_cat + sparse_attn epilogue via :meth:`_forward_prefill_compressed`.

        Under ``DSV4_PREFILL_CP_OVERLAP=1`` + CP-active, CSA/HCA layers
        instead dispatch to the overlap orchestrators (which hoist the
        compressor's CP all-gather ahead of the SWA write so they can
        overlap on the default stream vs the CP communication stream). The baseline sequential path
        below stays byte-equal otherwise.

        FP8 KV-cache is asserted at the public ``forward()`` entry; this
        body assumes FP8 unconditionally.
        """
        with record_function_range("dsv4.fp8.attn.prefill.common_setup"):
            common = self._prefill_common_setup(x, positions)
        with record_function_range("dsv4.fp8.attn.prefill.compute_qkv"):
            qkv = self._prefill_compute_qkv(
                x, common, shared_input_quant=shared_input_quant
            )
        if self.compress_ratio == 4 and _rt.should_record_layer(self.layer_id):
            _rt.record_if_level(2, f"L{self.layer_id:02d}_csa_qr", qkv.qr)

        # Phase-Z overlap dispatch: hoist the SWA write into the orchestrator
        # so it can run on the default stream while the compressor NCCL
        # gather drains on the side stream. The baseline path below is
        # left untouched for non-overlap and warmup forwards.
        if self._should_overlap_cp_for_prefill(common):
            if self.compress_ratio == 128:
                with record_function_range("dsv4.fp8.attn.prefill.path_hca_overlap"):
                    return self._forward_prefill_hca_overlapped(x, qkv, common)
            if self.compress_ratio == 4:
                with record_function_range("dsv4.fp8.attn.prefill.path_csa_overlap"):
                    return self._forward_prefill_csa_overlapped(x, qkv, common)

        # SWA pool write — every FP8 layer populates the SWA pool for
        # downstream decode. Safe to do before attention because new K
        # (abs pos [sp, sp+S)) and any cont-prefill prefix tail
        # (abs pos [sp-P, sp)) target disjoint slots.
        qkv = self._ensure_prefill_kv_full(qkv, common)
        with record_function_range("dsv4.fp8.attn.prefill.swa_write"):
            self._prefill_write_swa_fp8_paged(common, qkv.kv_full)

        if self.compress_ratio == 0:
            with record_function_range("dsv4.fp8.attn.prefill.path_swa"):
                out = self._forward_prefill_swa_only(qkv, common)
        elif self.compress_ratio == 4:
            with record_function_range("dsv4.fp8.attn.prefill.path_csa"):
                out = self._forward_prefill_csa(x, qkv, common)
        elif self.compress_ratio == 128:
            with record_function_range("dsv4.fp8.attn.prefill.path_hca"):
                out = self._forward_prefill_hca(x, qkv, common)
        else:
            raise AssertionError(f"unknown compress_ratio={self.compress_ratio}")
        return out

    def _forward_prefill_swa_only(
        self, qkv: PrefillQKV, common: PrefillMeta
    ) -> torch.Tensor:
        """SWA-only path (compress_ratio == 0). Skips kv_cat + sparse_attn.
        Cold/warmup attends over BF16 ``kv_full`` directly; continuation
        builds ``[prefix_tail | new_K_bf16]`` in a workspace and runs
        chunked ``flash_mla_sparse_fwd`` over it. Each attention chunk is
        output-projected immediately, so the full ``[T, H, D]`` attention
        output is never materialized alongside Q. ``any_cont`` is
        varlen-aware (set from ``prefix_lengths.any()`` under varlen,
        ``sp_int > 0`` otherwise) so a B>1 batch with any continuation
        request takes the workspace path."""
        # No compressor on this path → Q's union slice is free; materialize now.
        qkv = self._materialize_prefill_q(qkv, common)
        if not common.any_cont or self._kv_cache is None:
            with record_function_range("dsv4.fp8.attn.swa.via_kv_full"):
                out = self._attn_fp8_swa_via_kv_full(qkv, common)
        else:
            with record_function_range("dsv4.fp8.attn.swa.via_concat"):
                out = self._attn_fp8_swa_via_concat(qkv, common)
        return out

    def _forward_prefill_csa(
        self,
        x: torch.Tensor,
        qkv: PrefillQKV,
        common: PrefillMeta,
    ) -> torch.Tensor:
        """CSA path (compress_ratio == 4). Sparse compress topk via the
        IndexerFP8 lightning indexer; main compressor writes the CSA
        pool with hoisted meta. Attention runs through the vLLM-style
        workspace path (dual FP8 dequant + BF16 overlay + flash_mla_sparse_fwd).
        Falls back to BF16 ``kv_full`` attention on warmup (workspace_meta None)."""
        from rtp_llm.platforms.ppu.models.dsv4.impl.fp8.indexer import IndexerFP8

        assert isinstance(
            self.indexer, IndexerFP8
        ), "CSA layer requires IndexerFP8 (BF16 indexer not supported on this branch)"
        assert common.csa_meta is not None, (
            "CSA prefill requires common.csa_meta — built by " "_build_csa_prefill_meta"
        )

        # Phase-3a part 3: IndexerFP8.forward + nested CompressorFP8.forward
        # are now flat-input-native (accept ``[T_total, dim]`` /
        # ``[T_total, q_lora]``). Drop the legacy ``unsqueeze(0)`` so the
        # batched flat caller hits the same code path without rewrapping.
        with record_function_range("dsv4.fp8.attn.csa.indexer"):
            raw = self.indexer(
                x, qkv.qr, common.csa_meta.indexer_meta, workspace=common.workspace
            )
        return self._forward_prefill_compressed(
            x,
            qkv,
            common,
            cmp_topk_runtime=raw,
            compressor_meta=common.csa_meta.compressor_meta,
            workspace_meta=common.csa_meta.workspace_meta,
        )

    def _forward_prefill_csa_overlapped(
        self,
        x: torch.Tensor,
        qkv: PrefillQKV,
        common: PrefillMeta,
    ) -> torch.Tensor:
        """CSA path with two CP all-gathers overlapped onto the SWA write
        + the safe prefix of indexer work.

        Phase-Z orchestrator. CSA layers issue TWO independent NCCL
        collectives per step (nested indexer compressor + main CSA
        compressor); both must share the same ``cp_gather_stream`` so
        NCCL's per-stream FIFO ordering keeps the two collectives
        rank-consistent within the ProcessGroup. Sequence:

          1. ``indexer.start_prefill_nested_compressor`` enqueues NCCL
             #1 (nested indexer compressor's fused-KV gather) on
             ``cp_gather_stream``;
          2. ``compressor.start_prefill`` enqueues NCCL #2 (main CSA
             compressor's fused-KV gather) on the SAME stream — runs after
             NCCL #1 (FIFO);
          3. ``_prefill_write_swa_fp8_paged`` on the default stream
             overlaps with both gathers above;
          4. ``indexer.forward_with_pending_nested`` waits only NCCL #1,
             writes the indexer-side pool, then runs compute_q/weights_proj;
          5. before indexer ``gather_k_cache`` / score / topk, wait only
             NCCL #2. This keeps main NCCL from running concurrently with the
             numerically sensitive indexer score/topk chain while preserving
             the baseline-visible CSA pool write order;
          6. after indexer topk, ``compressor.finish_prefill`` writes the
             CSA pool;
          7. ``_forward_prefill_compressed(_skip_compressor_write=True,
             cmp_topk_runtime=raw)`` runs workspace_attn over the
             just-written CSA pool + the indexer topk.

        Bit-equal to :meth:`_forward_prefill_csa` (sequential baseline):
        same kernel inputs, same compressor_meta, same indexer chain,
        same workspace path — only the launch ordering differs.
        """
        from rtp_llm.platforms.ppu.models.dsv4.impl.fp8.indexer import IndexerFP8

        assert isinstance(
            self.indexer, IndexerFP8
        ), "CSA overlap requires IndexerFP8 (BF16 indexer not supported)"
        assert common.csa_meta is not None, (
            "CSA overlap prefill requires common.csa_meta — built by "
            "_build_csa_prefill_meta"
        )
        assert common.cp_ctx is not None and common.cp_on, (
            "_forward_prefill_csa_overlapped invoked without CP active — "
            "_should_overlap_cp_for_prefill guards this"
        )

        cp_stream = self._get_cp_gather_stream(x.device)
        csa_meta = common.csa_meta
        layer_label = f"L{int(getattr(self, 'layer_id', 0)):02d}"
        nested_pending = None
        main_pending = None
        try:
            with record_function_range(
                "dsv4.fp8.attn.csa_overlap.start_nested_compressor"
            ):
                nested_pending = self.indexer.start_prefill_nested_compressor(
                    x,
                    csa_meta.indexer_meta.sp_int,
                    meta=csa_meta.indexer_meta.compressor_meta,
                    cp_gather_stream=cp_stream,
                    profile_label=f"{layer_label}.csa_nested_indexer",
                    workspace=common.workspace,
                )
            with record_function_range(
                "dsv4.fp8.attn.csa_overlap.start_main_compressor"
            ):
                main_pending = self.compressor.start_prefill(
                    x,
                    common.sp_int,
                    meta=csa_meta.compressor_meta,
                    cp_gather_stream=cp_stream,
                    profile_label=f"{layer_label}.csa_main",
                    workspace=common.workspace,
                )
            with record_function_range("dsv4.fp8.attn.csa_overlap.swa_write"):
                qkv = self._ensure_prefill_kv_full(qkv, common)
                self._prefill_write_swa_fp8_paged(common, qkv.kv_full)

            def wait_main_before_indexer_k() -> None:
                if main_pending is None:
                    return
                with record_function_range(
                    "dsv4.fp8.attn.csa_overlap.wait_main_before_indexer_k"
                ):
                    self.compressor.wait_prefill_gather(main_pending)

            with record_function_range("dsv4.fp8.attn.csa_overlap.indexer"):
                indexer_post_stream = (
                    _get_cp_post_gather_stream(x.device) if x.is_cuda else None
                )
                raw = self.indexer.forward_with_pending_nested(
                    x,
                    qkv.qr,
                    csa_meta.indexer_meta,
                    nested_pending,
                    before_gather_k=wait_main_before_indexer_k,
                    cp_gather_stream=cp_stream,
                    post_gather_stream=indexer_post_stream,
                )
                nested_pending = None
                if _rt.should_record_layer(self.layer_id):
                    _rt.record_if_level(
                        2, f"L{self.layer_id:02d}_csa_indexer_topk", raw
                    )
            if main_pending is not None:
                with record_function_range(
                    "dsv4.fp8.attn.csa_overlap.finish_main_compressor"
                ):
                    self.compressor.finish_prefill(main_pending)
                main_pending = None

            return self._forward_prefill_compressed(
                x,
                qkv,
                common,
                cmp_topk_runtime=raw,
                compressor_meta=csa_meta.compressor_meta,
                workspace_meta=csa_meta.workspace_meta,
                _skip_compressor_write=True,
            )
        except Exception:
            self._cleanup_pending_prefill_gather(self.compressor, main_pending)
            nested_compressor = getattr(self.indexer, "compressor", None)
            self._cleanup_pending_prefill_gather(nested_compressor, nested_pending)
            clear_nested = getattr(self.indexer, "_clear_nested_pool", None)
            if clear_nested is not None:
                with suppress(Exception):
                    clear_nested()
            raise

    def _forward_prefill_hca(
        self,
        x: torch.Tensor,
        qkv: PrefillQKV,
        common: PrefillMeta,
    ) -> torch.Tensor:
        """HCA path (compress_ratio == 128). Dense compressed indices live
        in ``workspace_meta.dense_cmp_topk``; runtime cmp_topk is None.
        Main compressor writes the HCA pool with hoisted meta."""
        assert self.indexer is None, "HCA layer must not have an indexer"
        assert common.hca_meta is not None, (
            "HCA prefill requires common.hca_meta — built by " "_build_hca_prefill_meta"
        )

        return self._forward_prefill_compressed(
            x,
            qkv,
            common,
            cmp_topk_runtime=None,
            compressor_meta=common.hca_meta.compressor_meta,
            workspace_meta=common.hca_meta.workspace_meta,
        )

    def _forward_prefill_hca_overlapped(
        self,
        x: torch.Tensor,
        qkv: PrefillQKV,
        common: PrefillMeta,
    ) -> torch.Tensor:
        """HCA path with CP all-gather overlapped onto the SWA pool write.

        Phase-Z orchestrator. Reachable only when
        ``_should_overlap_cp_for_prefill`` returned True (env on + CP
        active + not capturing). Sequence:

          1. ``compressor.start_prefill`` enqueues the fused-KV CP gather
             on ``cp_gather_stream`` (side stream — no default-stream
             dependency yet);
          2. ``_prefill_write_swa_fp8_paged`` runs on the default stream
             in parallel with the NCCL gather above (disjoint pool +
             independent input ``qkv.kv_full``);
          3. ``compressor.finish_prefill`` waits the gather + writes the
             HCA pool;
          4. ``_forward_prefill_compressed(_skip_compressor_write=True)``
             runs the workspace path over the just-written HCA pool.

        Bit-equal to :meth:`_forward_prefill_hca` (sequential baseline):
        same kernel inputs, same compressor_meta, same workspace path —
        only the launch ordering differs.
        """
        assert self.indexer is None, "HCA layer must not have an indexer"
        assert common.hca_meta is not None, (
            "HCA overlap prefill requires common.hca_meta — built by "
            "_build_hca_prefill_meta"
        )
        assert common.cp_ctx is not None and common.cp_on, (
            "_forward_prefill_hca_overlapped invoked without CP active — "
            "_should_overlap_cp_for_prefill guards this"
        )

        cp_stream = self._get_cp_gather_stream(x.device)
        layer_label = f"L{int(getattr(self, 'layer_id', 0)):02d}"
        main_pending = None
        try:
            with record_function_range("dsv4.fp8.attn.hca_overlap.start_compressor"):
                main_pending = self.compressor.start_prefill(
                    x,
                    common.sp_int,
                    meta=common.hca_meta.compressor_meta,
                    cp_gather_stream=cp_stream,
                    profile_label=f"{layer_label}.hca_main",
                    workspace=common.workspace,
                )
            with record_function_range("dsv4.fp8.attn.hca_overlap.swa_write"):
                # Default-stream work that overlaps with the NCCL gather above.
                qkv = self._ensure_prefill_kv_full(qkv, common)
                self._prefill_write_swa_fp8_paged(common, qkv.kv_full)
            with record_function_range("dsv4.fp8.attn.hca_overlap.finish_compressor"):
                self.compressor.finish_prefill(main_pending)
            main_pending = None

            return self._forward_prefill_compressed(
                x,
                qkv,
                common,
                cmp_topk_runtime=None,
                compressor_meta=common.hca_meta.compressor_meta,
                workspace_meta=common.hca_meta.workspace_meta,
                _skip_compressor_write=True,
            )
        except Exception:
            self._cleanup_pending_prefill_gather(self.compressor, main_pending)
            raise

    def _forward_prefill_compressed(
        self,
        x: torch.Tensor,
        qkv: PrefillQKV,
        common: PrefillMeta,
        cmp_topk_runtime: Optional[torch.Tensor],
        compressor_meta,
        workspace_meta: Optional[WorkspaceMeta],
        *,
        _skip_compressor_write: bool = False,
    ) -> torch.Tensor:
        """Shared CSA/HCA epilogue: write compressed-K via main compressor
        (with hoisted ``compressor_meta``), then run the workspace-path
        attention. The workspace path streams each attention chunk directly
        through output projection; warmup falls back to
        ``_attn_fp8_swa_via_kv_full`` when ``workspace_meta`` is None (pool
        context unbound).

        ``_skip_compressor_write`` is the Phase-Z overlap escape hatch:
        the orchestrator (HCA/CSA) has already drained the compressor's
        gather via ``finish_prefill``, so this method must NOT issue a
        second synchronous compressor call (which would re-do the work
        and break correctness). Baseline (non-overlap) callers leave
        the default ``False`` and the historical sequential path runs.
        """
        # Compressor write (return value is unused — compressor handles its
        # own pool dual-write; the workspace path re-reads the just-written
        # tail via dequantize_and_gather_k_cache, with BF16 overlay on top).
        # Phase-3a: ``CompressorFP8.forward`` is now flat-input-native
        # (accepts ``[T_total, dim]``). Drop the ``unsqueeze(0)`` legacy
        # rewrap so the batched (B>1) caller can reuse this code path
        # without re-shaping. B==1 reaches the same kernel — same flat
        # ``[T_total, dim]`` reshape inside ``_launch``.
        if not _skip_compressor_write:
            with record_function_range("dsv4.fp8.attn.compressed.compressor"):
                self.compressor(
                    x, common.sp_int, meta=compressor_meta, workspace=common.workspace
                )

        # Materialize the deferred dense Q now that the compressor(s) have
        # drained their CP gather/restore buffers (which alias Q's union
        # storage). Covers all four compressed paths: non-overlap wrote the
        # compressor just above; overlap already drained via finish_prefill in
        # the orchestrator before reaching here (_skip_compressor_write=True).
        qkv = self._materialize_prefill_q(qkv, common)
        if (
            self.compress_ratio == 4
            and qkv.q is not None
            and _rt.should_record_layer(self.layer_id)
        ):
            _rt.record_if_level(2, f"L{self.layer_id:02d}_csa_q_materialized", qkv.q)

        if workspace_meta is None:
            # Warmup forward: pool not bound. Fall back to BF16 ``kv_full``
            # SWA-only attention so framework shape inference still runs.
            with record_function_range("dsv4.fp8.attn.compressed.warmup_attn"):
                out = self._attn_fp8_swa_via_kv_full(qkv, common)
        else:
            with record_function_range("dsv4.fp8.attn.compressed.workspace_attn"):
                out = self._attn_via_workspace(
                    qkv, common, workspace_meta, cmp_topk_runtime
                )
        return out

    def _attn_via_workspace(
        self,
        qkv: PrefillQKV,
        common: PrefillMeta,
        workspace_meta: "WorkspaceMeta",
        cmp_topk_runtime: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """vLLM-style workspace path for CSA/HCA prefill (varlen B>=1, non-CP).

        Pipeline (kernel-only — every elementwise op pre-baked in
        :meth:`_build_workspace_meta`):
          1. Allocate ``workspace [B, M, head_dim]`` BF16 zeros.
          2. ``dequantize_and_gather_k_cache(cmp_pool, offset=0)`` →
             ``workspace[b, 0:N_b, :]`` per request (per-req ``cmp_seq_lens``
             handle the per-row variable length; ``[N_b, N_max)`` stays zero).
          3. ``dequantize_and_gather_k_cache(swa_pool, offset=N_max)`` →
             ``workspace[b, N_max:N_max+P_b, :]`` per request. SWA only
             stores the tail blocks; fresh ``S_b`` rows are supplied by step 4.
          4. BF16 overlay freshly computed new K via single ``index_copy_``
             over ``workspace.view(B*M, D)`` using ``wm.new_k_slot_in_flat``
             (already encodes ``M*req_id + N + P_b + local_pos``).
          5. ``combine_topk_swa_indices`` packs the per-query
             ``[compressed_valid | swa_valid]`` index list. The kernel
             internally computes per-token ``min((pos+1)//ratio, TOP_K)``
             so HCA's full ``arange(N_max)`` row gets masked even for tokens
             whose request has ``N_b < N_max``.
          6. ``flash_mla_sparse_fwd`` over the ``[B*M, 1, D]`` workspace view,
             chunked on Q; each chunk is immediately fed into
             :meth:`_prefill_output_proj_into` and written to the final contiguous
             ``[T, dim]`` output buffer, so no full ``[T, H, D]`` attention
             output is materialized.

        Mirrors vLLM ``DeepseekV4MultiHeadLatentAttentionWrapper._forward_prefill``.

        ``cmp_topk_runtime`` is the indexer output (CSA path); for HCA it's
        ignored and ``workspace_meta.dense_cmp_topk`` (precomputed
        ``arange(N_max)`` per token) is used instead.
        """
        assert qkv.q is not None, "_attn_via_workspace: prefill Q not materialized"

        from rtp_llm.models_py.modules.dsv4.fp8 import _swa_dequant_triton as _swa_dq
        from rtp_llm.models_py.modules.dsv4.fp8._swa_ops_triton import (
            combine_topk_swa_indices,
            combine_topk_swa_indices_cp,
        )
        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import CSA_KV, HCA_KV, SWA_KV

        ratio = self.compress_ratio
        ratio_tag = "csa" if ratio == 4 else "hca"
        cmp_at = CSA_KV if ratio == 4 else HCA_KV
        swa_byte_sliced = self._swa_cp_byte_sliced()
        swa_pool_raw = self._pool_raw_u8(SWA_KV) if swa_byte_sliced else None
        swa_pool_3d = None if swa_byte_sliced else self._pool_view_3d_fp8(SWA_KV)
        cmp_pool_3d = self._pool_view_3d_fp8(cmp_at)
        assert cmp_pool_3d is not None, "FP8 CSA/HCA pool required for workspace path"
        if swa_byte_sliced:
            assert swa_pool_raw is not None, "byte-sliced FP8 SWA pool unavailable"
        else:
            assert swa_pool_3d is not None, "FP8 SWA pool required for workspace path"

        assert workspace_meta is not None, (
            "_attn_via_workspace requires a non-None WorkspaceMeta; the caller "
            "must short-circuit to _attn_fp8_swa_via_kv_full when the pool is "
            "unbound (warmup)."
        )
        wm = workspace_meta
        # ``B`` from the per-request meta — bit-equal to legacy 1 when meta
        # was built from the scalar branch (swa_seq_lens shape == [1]).
        B = int(wm.swa_seq_lens.shape[0])
        assert B == common.batch_size, (
            f"workspace meta B ({B}) != common.batch_size ({common.batch_size}); "
            "meta builder fed an inconsistent shape — likely a varlen / legacy "
            "dispatch mismatch."
        )
        D = self.head_dim

        if wm.use_cp_raw_q_merge:
            with record_function_range("dsv4.fp8.attn.workspace.cp_raw_q_merge"):
                o = self._attn_via_workspace_cp_raw_q_merge(
                    qkv=qkv,
                    common=common,
                    workspace_meta=wm,
                    cmp_topk_runtime=cmp_topk_runtime,
                    cmp_pool_3d=cmp_pool_3d,
                    swa_pool_3d=swa_pool_3d,
                )
            with record_function_range("dsv4.fp8.attn.prefill.output_proj"):
                out = self._prefill_output_proj(o, common.freqs_cis)
            self._prefill_output_all_reduce(out)
            return out

        async_workspace_reads = self._should_async_workspace_reads_for_prefill(common)
        with record_function_range("dsv4.fp8.attn.workspace.alloc"):
            # E5b: combined indices only address rows written by compressed-K
            # gather, SWA-prefix gather, or fresh-K overlay; keep this
            # uninitialized to avoid a full workspace memset per layer.
            workspace = torch.empty(
                (B, wm.M, D), dtype=torch.bfloat16, device=qkv.q.device
            )

        cmp_pending = None
        cmp_reader_for_pending = None
        cmp_prepare_for_pending = None
        swa_prefix_pending = None
        try:
            if wm.N > 0:
                with record_function_range("dsv4.fp8.attn.workspace.gather_cmp"):
                    # Stage 5b: dispatch through the per-iteration reader. For
                    # non-CP / cp_size=1 / unsharded KV-cache this is a thin
                    # wrapper around the original ``dequantize_and_gather_k_cache``
                    # (LocalPoolReader). For CP-sharded KV-cache it gathers any
                    # nonzero compressed-K range from peer ranks before dequant.
                    cmp_reader = (
                        wm.cmp_reader
                        if wm.cmp_reader is not None
                        else LocalPoolReader()
                    )
                    start_cmp = (
                        getattr(cmp_reader, "start_fill_async", None)
                        if async_workspace_reads
                        else None
                    )
                    prepare_cmp = (
                        getattr(cmp_reader, "prepare_fill_async", None)
                        if async_workspace_reads
                        else None
                    )
                    if start_cmp is not None and prepare_cmp is not None:
                        with record_function_range(
                            "dsv4.fp8.attn.workspace.gather_cmp.prefetch_start"
                        ):
                            cmp_pending = start_cmp(
                                out=workspace,
                                k_cache=cmp_pool_3d,
                                seq_lens=wm.cmp_seq_lens,
                                gather_lens=None,
                                block_table=wm.cmp_bt_int32,
                                block_size=wm.cmp_eb,
                                offset=0,
                                stream=self._get_cp_gather_stream(qkv.q.device),
                            )
                        if cmp_pending is not None:
                            cmp_reader_for_pending = cmp_reader
                            cmp_prepare_for_pending = prepare_cmp
                    if cmp_pending is None:
                        cmp_reader.fill(
                            out=workspace,
                            k_cache=cmp_pool_3d,
                            seq_lens=wm.cmp_seq_lens,
                            gather_lens=None,
                            block_table=wm.cmp_bt_int32,
                            block_size=wm.cmp_eb,
                            offset=0,
                        )
            # SWA dequant only reads the cached prefix tail. The fresh new-K rows
            # are already available as BF16 and are overlaid below; cold prefill
            # does not need to read the SWA pool at all.
            if common.any_cont:
                with record_function_range("dsv4.fp8.attn.workspace.gather_swa_prefix"):
                    assert wm.swa_cache_slot_mapping is not None
                    if swa_byte_sliced:
                        assert common.cp_ctx is not None
                        assert wm.swa_cache_compaction is not None
                        if async_workspace_reads:
                            with record_function_range(
                                "dsv4.fp8.attn.workspace.gather_swa_prefix.prefetch_start"
                            ):
                                swa_prefix_pending = _swa_dq.start_dequantize_and_gather_k_cache_slots_cp_byte_sliced(
                                    k_cache_raw=swa_pool_raw,
                                    slot_mapping=wm.swa_cache_slot_mapping,
                                    gather_lens=wm.swa_cache_gather_lens,
                                    offset=wm.N,
                                    full_entries_per_block=wm.swa_eb,
                                    cp_rank=int(common.cp_ctx.cp_rank),
                                    cp_size=int(common.cp_ctx.cp_size),
                                    compaction=wm.swa_cache_compaction,
                                    stream=self._get_cp_gather_stream(qkv.q.device),
                                    profile_name=(
                                        f"dsv4.cp.all_gather.L{self.layer_id:02d}"
                                        ".swa_prefix"
                                    ),
                                )
                        if swa_prefix_pending is None:
                            _swa_dq.dequantize_and_gather_k_cache_slots_cp_byte_sliced(
                                out=workspace,
                                k_cache_raw=swa_pool_raw,
                                slot_mapping=wm.swa_cache_slot_mapping,
                                gather_lens=wm.swa_cache_gather_lens,
                                offset=wm.N,
                                full_entries_per_block=wm.swa_eb,
                                cp_rank=int(common.cp_ctx.cp_rank),
                                cp_size=int(common.cp_ctx.cp_size),
                                compaction=wm.swa_cache_compaction,
                            )
                    else:
                        assert swa_pool_3d is not None
                        _swa_dq.dequantize_and_gather_k_cache_slots(
                            out=workspace,
                            k_cache=swa_pool_3d,
                            slot_mapping=wm.swa_cache_slot_mapping,
                            gather_lens=wm.swa_cache_gather_lens,
                            offset=wm.N,
                        )
            # BF16 overlay of freshly computed new K — single ``index_copy_``
            # using the meta-precomputed ``new_k_slot_in_flat``. Avoids the FP8
            # round-trip loss on tokens we just wrote, while keeping the hot
            # path free of casts / gathers / per-request slicing.
            with record_function_range("dsv4.fp8.attn.workspace.overlay_new_k"):
                kv_bf16 = qkv.kv_full.to(torch.bfloat16).reshape(-1, D)
                workspace.view(B * wm.M, D).index_copy_(
                    0, wm.new_k_slot_in_flat, kv_bf16
                )
                # Free the kv_full storage before combine_topk + flash_mla_sparse_fwd.
                # After the overlay, kv_full has no remaining consumer in this
                # function or any caller; workspace attention now output-projects
                # each FlashMLA chunk immediately after attention.
                # The NamedTuple ref keeps it alive otherwise, costing ~1.1 GiB
                # of peak overlap with the sparse-attn workspace at 1M ctx.
                dispose_tensor(kv_bf16)
                dispose_tensor(qkv.kv_full)

            # combine_topk: HCA uses precomputed dense arange(N_max); CSA uses
            # the runtime indexer output (raw compressed-pool offsets in [0, N_b)).
            # Indexer output contract: ``[T_total, K] int32 contiguous`` (set in
            # ``IndexerFP8.forward`` line 754 + ``view(out_shape)``); assert here
            # so a contract drift surfaces as a loud crash instead of a silent
            # per-layer ``squeeze``/``to``/``contiguous`` retag in the hot path.
            if wm.dense_cmp_topk is not None:
                cmp_topk = wm.dense_cmp_topk
            else:
                assert (
                    cmp_topk_runtime is not None
                ), "CSA workspace path requires cmp_topk_runtime (indexer output)"
                assert (
                    cmp_topk_runtime.dim() == 2
                    and cmp_topk_runtime.dtype == torch.int32
                    and cmp_topk_runtime.is_contiguous()
                ), (
                    "cmp_topk_runtime contract violated: expected 2D int32 "
                    f"contiguous, got dim={cmp_topk_runtime.dim()} "
                    f"dtype={cmp_topk_runtime.dtype} contig={cmp_topk_runtime.is_contiguous()}"
                )
                cmp_topk = cmp_topk_runtime

            # Prepare FlashMLA's KV view before combine_topk. The
            # combine kernel is queued behind the overlay writes on the same
            # stream, so moving pure CPU/view/allocation work here lets it run
            # while the GPU drains the pre-combine dependency chain.
            kv_view = workspace.view(B * wm.M, 1, D)
            projected_out = torch.empty(
                qkv.q.shape[0],
                self.dim,
                dtype=torch.bfloat16,
                device=qkv.q.device,
            )

            if common.cp_on:
                # Phase F2/Phase-2: kernel ``combine_topk_swa_indices`` derives
                # per-Q-row ``pos = start_pos + token_idx_in_query`` assuming Q
                # is a contiguous slice. Under zigzag CP each rank's Q rows
                # have non-contiguous global positions, so use the CP fused
                # kernel that consumes explicit positions directly.
                assert common.cp_ctx is not None
                cp_ctx_local = common.cp_ctx
                legacy_prefix_length = int(cp_ctx_local.prefix_length)
                combine_kwargs = dict(
                    topk_indices=cmp_topk,
                    global_positions=_flat_1d(cp_ctx_local.global_positions),
                    sp_int=legacy_prefix_length,
                    window_size=self.window_size,
                    compress_ratio=ratio,
                    topk=int(cmp_topk.shape[-1]),
                    M=wm.M,
                    N=wm.N,
                )
                assert common.req_id_per_token is not None
                assert common.prefix_lengths is not None
                combine_kwargs.update(
                    req_id_per_token=_flat_1d(common.req_id_per_token),
                    prefix_lengths=_flat_1d(common.prefix_lengths),
                )
                with record_function_range("dsv4.fp8.attn.workspace.combine_topk_cp"):
                    combined_indices, combined_lens = combine_topk_swa_indices_cp(
                        **combine_kwargs,
                        flash_mla_indices=True,
                    )
            else:
                with record_function_range("dsv4.fp8.attn.workspace.combine_topk"):
                    combined_indices, combined_lens = combine_topk_swa_indices(
                        topk_indices=cmp_topk,
                        query_start_loc=wm.qsl,
                        seq_lens=wm.swa_seq_lens,
                        gather_lens=wm.swa_gather_lens,
                        window_size=self.window_size,
                        compress_ratio=ratio,
                        topk=int(cmp_topk.shape[-1]),
                        M=wm.M,
                        N=wm.N,
                        flash_mla_indices=True,
                    )

            post_gather_stream = None
            if cmp_pending is not None or swa_prefix_pending is not None:
                post_gather_stream = _get_cp_post_gather_stream(qkv.q.device)

            # Only the NCCL stage is pulled ahead. Local restore/dequant/scatter
            # is queued after overlay/combine and must finish before FlashMLA
            # reads workspace. The three writes target disjoint ranges
            # (cmp [0:N), SWA prefix [N:N+P), fresh K [N+P:]); if this prepare
            # point is moved earlier for more overlap, keep the Work.wait() call
            # inside the post stream helper or gathered bytes can be read early.
            if cmp_pending is not None:
                assert cmp_reader_for_pending is not None
                assert cmp_prepare_for_pending is not None
                assert post_gather_stream is not None
                with record_function_range(
                    "dsv4.fp8.attn.workspace.gather_cmp.prefetch_prepare"
                ):
                    cmp_prepare_for_pending(
                        cmp_pending,
                        stream=post_gather_stream,
                    )

            if swa_prefix_pending is not None:
                assert post_gather_stream is not None
                with record_function_range(
                    "dsv4.fp8.attn.workspace.gather_swa_prefix.prefetch_prepare"
                ):
                    _swa_dq.prepare_dequantize_and_gather_k_cache_slots_cp_byte_sliced(
                        swa_prefix_pending,
                        out=workspace,
                        stream=post_gather_stream,
                    )

            # The post-stream restore/dequant for cmp + SWA prefix can run
            # concurrently with the cmp wait_event below and with FlashMLA's
            # launch-side bookkeeping. Drain both pendings only when the
            # consumer is about to read workspace, so the prepare-side work
            # actually has a window to overlap.
            if cmp_pending is not None:
                with record_function_range(
                    "dsv4.fp8.attn.workspace.gather_cmp.prefetch_wait"
                ):
                    cmp_reader_for_pending.wait_fill_async(cmp_pending)
                cmp_pending = None

            if swa_prefix_pending is not None:
                with record_function_range(
                    "dsv4.fp8.attn.workspace.gather_swa_prefix.prefetch_wait"
                ):
                    _swa_dq.wait_dequantize_and_gather_k_cache_slots_cp_byte_sliced(
                        swa_prefix_pending
                    )
                swa_prefix_pending = None

            if self.compress_ratio == 4 and _rt.should_record_layer(self.layer_id):
                prefix = f"L{self.layer_id:02d}_csa"
                _rt.record_if_level(2, f"{prefix}_cmp_topk", cmp_topk)
                _rt.record_if_level(2, f"{prefix}_combined_indices", combined_indices)
                _rt.record_if_level(2, f"{prefix}_combined_lens", combined_lens)
                _rt.record_if_level(2, f"{prefix}_attention_workspace", workspace)

            return self._flash_mla_sparse_fwd_chunked_projected(
                q=qkv.q,
                kv=kv_view,
                indices=combined_indices,
                topk_length=combined_lens,
                freqs_cis=common.freqs_cis,
                prefill_workspace=common.workspace,
                profile_name="dsv4.fp8.attn.workspace.flash_mla_sparse_fwd",
                out=projected_out,
            )
        finally:
            if cmp_pending is not None and cmp_reader_for_pending is not None:
                cmp_reader_for_pending.discard_fill_async(cmp_pending)
            if swa_prefix_pending is not None:
                _swa_dq.discard_dequantize_and_gather_k_cache_slots_cp_byte_sliced(
                    swa_prefix_pending
                )

    @staticmethod
    def _cp_full_req_ids_and_positions(
        common: PrefillMeta,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert common.cp_ctx is not None
        cp_ctx = common.cp_ctx
        lengths = cp_ctx.input_lengths_global
        if lengths is None:
            lengths = torch.tensor(
                [int(cp_ctx.seq_len_full)], device=common.device, dtype=torch.int32
            )
        prefix = cp_ctx.prefix_lengths
        if prefix is None:
            prefix = torch.tensor(
                [int(cp_ctx.prefix_length)], device=common.device, dtype=torch.long
            )
        device = common.device
        lengths_l = lengths.to(device=device, dtype=torch.long).reshape(-1)
        prefix_l = prefix.to(device=device, dtype=torch.long).reshape(-1)
        # Vectorized: req_full = repeat_interleave(arange(B), lengths_l);
        # pos_full = prefix[req] + (arange(total) - cu_starts[req]).
        # Single .item() sync on total replaces B .item() syncs + Python loop.
        total = int(lengths_l.sum().item())
        if total == 0:
            return (
                torch.empty((0,), device=device, dtype=torch.long),
                torch.empty((0,), device=device, dtype=torch.long),
            )
        req_full = torch.repeat_interleave(
            torch.arange(int(lengths_l.numel()), device=device, dtype=torch.long),
            lengths_l,
        )
        cu_starts = torch.zeros_like(lengths_l)
        cu_starts[1:] = torch.cumsum(lengths_l[:-1], dim=0)
        local_pos = torch.arange(
            total, device=device, dtype=torch.long
        ) - cu_starts.index_select(0, req_full)
        pos_full = local_pos + prefix_l.index_select(0, req_full)
        return req_full.contiguous(), pos_full.contiguous()

    @staticmethod
    def _cp_local_full_row_indices(common: PrefillMeta) -> torch.Tensor:
        assert common.cp_ctx is not None
        cp_ctx = common.cp_ctx
        assert cp_ctx.req_id_per_token is not None
        device = common.device
        req = cp_ctx.req_id_per_token.to(device=device, dtype=torch.long).reshape(-1)
        if cp_ctx.prefix_lengths is not None:
            prefix = cp_ctx.prefix_lengths.to(device=device, dtype=torch.long).reshape(
                -1
            )
        else:
            prefix = torch.tensor(
                [int(cp_ctx.prefix_length)], device=device, dtype=torch.long
            )
        if cp_ctx.cu_seqlens_global is not None:
            cu = cp_ctx.cu_seqlens_global.to(device=device, dtype=torch.long).reshape(
                -1
            )
        else:
            cu = torch.tensor(
                [0, int(cp_ctx.seq_len_full)], device=device, dtype=torch.long
            )
        pos = cp_ctx.global_positions.to(device=device, dtype=torch.long).reshape(-1)
        local_pos = pos - prefix.index_select(0, req)
        idx = cu.index_select(0, req) + local_pos
        return idx.clamp_(min=0, max=max(int(cp_ctx.seq_len_full) - 1, 0)).contiguous()

    def _attn_via_workspace_cp_raw_q_merge(
        self,
        *,
        qkv: PrefillQKV,
        common: PrefillMeta,
        workspace_meta: WorkspaceMeta,
        cmp_topk_runtime: Optional[torch.Tensor],
        cmp_pool_3d: torch.Tensor,
        swa_pool_3d: torch.Tensor,
    ) -> torch.Tensor:
        assert (
            qkv.q is not None
        ), "_attn_via_workspace_cp_raw_q_merge: prefill Q not materialized"
        from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
        from rtp_llm.models_py.modules.dsv4.fp8 import _swa_dequant_triton as _swa_dq
        from rtp_llm.models_py.utils.flash_mla_legacy import (
            flash_mla_sparse_fwd,  # type: ignore[import-not-found]
        )

        assert common.cp_ctx is not None
        cp_ctx = common.cp_ctx
        wm = workspace_meta
        D = self.head_dim
        B = int(wm.swa_seq_lens.shape[0])
        if (
            os.environ.get("DSV4_FORWARD_TENSOR_DEBUG", "0") != "0"
            or os.environ.get("DSV4_CP_RAW_Q_MERGE_LOG", "0") != "0"
        ):
            prefix_src = (
                cp_ctx.prefix_lengths
                if cp_ctx.prefix_lengths is not None
                else common.prefix_lengths
            )
            input_src = (
                cp_ctx.input_lengths_global
                if cp_ctx.input_lengths_global is not None
                else common.input_lengths
            )
            payload = {
                "tag": "DSV4_RAW_Q_MERGE",
                "layer_id": int(self.layer_id),
                "compress_ratio": int(self.compress_ratio),
                "cp_rank": int(cp_ctx.cp_rank),
                "cp_size": int(cp_ctx.cp_size),
                "B": B,
                "prefix_lengths": (
                    prefix_src.detach().cpu().reshape(-1).tolist()
                    if prefix_src is not None
                    else None
                ),
                "input_lengths": (
                    input_src.detach().cpu().reshape(-1).tolist()
                    if input_src is not None
                    else None
                ),
            }
            print(json.dumps(payload, sort_keys=True), flush=True)

        local_cmp_lens = cp_actual_owned_kv_lens(
            wm.cmp_seq_lens.to(torch.int64),
            cp_ctx.cp_size,
            wm.cmp_eb,
            cp_ctx.cp_rank,
        ).to(device=qkv.q.device, dtype=torch.int32)
        local_N = int(local_cmp_lens.max().item()) if local_cmp_lens.numel() else 0
        gather_len_max = (
            int(wm.swa_gather_lens.max().item()) if wm.swa_gather_lens.numel() else 0
        )
        local_M = local_N + gather_len_max
        # E5b: raw-Q merge builds compact topk over written local rows only.
        workspace = torch.empty(
            (B, local_M, D), dtype=torch.bfloat16, device=qkv.q.device
        )

        if local_N > 0:
            LocalPoolReader().fill(
                out=workspace,
                k_cache=cmp_pool_3d,
                seq_lens=local_cmp_lens,
                gather_lens=None,
                block_table=wm.cmp_bt_int32,
                block_size=wm.cmp_eb,
                offset=0,
            )

        # Mirror the packed-KV workspace path: the SWA pool only needs to
        # provide cached prefix tail rows. Fresh prefill K rows come from the
        # all-gathered BF16 ``kv_full`` overlay below. Reading the whole SWA
        # stream here is both wasted work on cold prefill and unsafe when the
        # CP KV cache is sharded, because the full logical SWA block table is
        # not locally readable.
        if common.any_cont:
            _swa_dq.dequantize_and_gather_k_cache(
                out=workspace,
                k_cache=swa_pool_3d,
                seq_lens=wm.swa_cache_seq_lens,
                gather_lens=wm.swa_cache_gather_lens,
                block_table=wm.swa_bt_int32,
                block_size=wm.swa_eb,
                offset=local_N,
            )

        req_full, pos_full = self._cp_full_req_ids_and_positions(common)
        if common.cp_ctx.prefix_lengths is not None:
            prefix_lens = common.cp_ctx.prefix_lengths.to(
                device=qkv.q.device, dtype=torch.long
            )
        else:
            prefix_lens = torch.tensor(
                [int(common.cp_ctx.prefix_length)],
                device=qkv.q.device,
                dtype=torch.long,
            )
        P_per_req = torch.clamp_max(prefix_lens, self.window_size - 1)
        local_pos = pos_full - prefix_lens.index_select(0, req_full)
        new_k_slots = (
            req_full * local_M
            + local_N
            + P_per_req.index_select(0, req_full)
            + local_pos
        ).contiguous()
        workspace.view(B * local_M, D).index_copy_(
            0, new_k_slots, qkv.kv_full.to(torch.bfloat16).reshape(-1, D)
        )

        q_full = cp_all_gather_full_varlen(qkv.q, cp_ctx)
        if wm.dense_cmp_topk is not None:
            if wm.N > 0:
                dense = (
                    torch.arange(wm.N, device=qkv.q.device, dtype=torch.int64)
                    .view(1, wm.N)
                    .expand(int(q_full.shape[0]), wm.N)
                )
                dense_len = torch.clamp(
                    (pos_full + 1) // int(self.compress_ratio),
                    max=wm.N,
                ).unsqueeze(1)
                cmp_topk_full = (
                    torch.where(dense < dense_len, dense, torch.full_like(dense, -1))
                    .to(torch.int32)
                    .contiguous()
                )
            else:
                cmp_topk_full = torch.empty(
                    (int(q_full.shape[0]), 0), device=qkv.q.device, dtype=torch.int32
                )
        else:
            assert cmp_topk_runtime is not None
            cmp_topk_full = cp_all_gather_full_varlen(cmp_topk_runtime, cp_ctx)

        local_topk_compact = remap_topk_to_cp_local(
            cmp_topk_full,
            per_req_total_kv_lens=wm.cmp_seq_lens.to(torch.int64),
            cp_size=cp_ctx.cp_size,
            cp_rank=cp_ctx.cp_rank,
            block_size=wm.cmp_eb,
            req_id_per_token=req_full,
        )
        local_topk = self._compact_local_topk_to_workspace(
            local_topk_compact,
            req_id_per_token=req_full,
            per_req_total_kv_lens=wm.cmp_seq_lens.to(torch.int64),
            cp_size=cp_ctx.cp_size,
            block_size=wm.cmp_eb,
            M=local_M,
        )
        local_swa, _ = build_swa_cp_local_indices(
            pos_full,
            prefix_lengths=prefix_lens,
            cp_size=cp_ctx.cp_size,
            cp_rank=cp_ctx.cp_rank,
            window_size=self.window_size,
            M=local_M,
            N=local_N,
            req_id_per_token=req_full,
        )
        combined_indices, combined_lens = self._compact_indices([local_topk, local_swa])

        local_o, _, local_lse = flash_mla_sparse_fwd(
            q=q_full,
            kv=workspace.view(B * local_M, 1, D),
            indices=combined_indices.unsqueeze(1),
            sm_scale=self.softmax_scale,
            attn_sink=None,
            topk_length=combined_lens,
        )
        with record_function_range(
            f"dsv4.cp.all_gather.L{self.layer_id:02d}.workspace_attn.o.launch"
        ):
            gathered_o = all_gather(local_o.contiguous(), group=Group.TP).view(
                cp_ctx.cp_size, int(q_full.shape[0]), self.n_heads, D
            )
        with record_function_range(
            f"dsv4.cp.all_gather.L{self.layer_id:02d}.workspace_attn.lse.launch"
        ):
            gathered_lse = all_gather(local_lse.contiguous(), group=Group.TP).view(
                cp_ctx.cp_size, int(q_full.shape[0]), self.n_heads
            )
        merged_o, merged_lse = merge_lse_output(gathered_o, gathered_lse, dim=0)
        merged_o = self._raw_q_merge_apply_sink(merged_o, merged_lse)
        local_rows = self._cp_local_full_row_indices(common)
        return merged_o.index_select(0, local_rows).unsqueeze(0)

    def _build_shared_prefill_meta(
        self,
        x: torch.Tensor,
        positions: Union[int, torch.Tensor],
        sp_per_req: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        input_lengths: Optional[torch.Tensor] = None,
        prefix_lengths: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        req_id_per_token: Optional[torch.Tensor] = None,
        max_seqlen_q: int = 0,
        any_cont: Optional[bool] = None,
        reuse_common_meta: Optional["PrefillMeta"] = None,
        reuse_freqs_meta: Optional["PrefillMeta"] = None,
    ) -> "PrefillMeta":
        """Build the layer-invariant (within compress_ratio bucket) part
        of per-call prefill metadata. All host-side prep work that
        doesn't depend on ``self.layer_id`` lives here so the upper layer
        can run it once per ratio and broadcast to every same-ratio
        attention via :meth:`_set_prefill_meta_shared`. Standalone path
        falls back to running this per-layer.

        ``positions`` accepts either a ``[T]`` int64 tensor (from the
        normal call path) or a pre-synced int (used by upper-layer
        broadcast meta builders that already paid the sync once for
        the whole batch). When a tensor is passed we sync once here.

        ``reuse_common_meta`` is supplied only by the upper-layer broadcast
        builder. It reuses the first ratio's top-k/continuation tensors and
        SWA Group-1 write metadata while this call still builds its own
        ratio-specific CSA/HCA metadata. ``reuse_freqs_meta`` is separate:
        ratio0 uses base RoPE while ratio4/128 share compressed RoPE, so only
        a metadata object produced from the identical source table may supply
        the gathered frequencies. Omitting either input preserves the full or
        partial standalone fallback.
        """
        seqlen = int(x.shape[0])
        rd = self.rope_head_dim
        device = x.device

        # CP plumbing: this builder threads ``cp_ctx`` onto ``PrefillMeta`` so
        # downstream SWA write / sparse attention / compressor paths can switch
        # between rank-local Q metadata and the all-gathered KV/write view.
        # CP=1 collapses to the legacy single-rank values.
        cp_ctx = getattr(self, "_cp_ctx", None)
        cp_on = cp_ctx is not None and cp_ctx.cp_size > 1

        # Sync ``positions[0]`` -> int once. Tensor input pays the sync here
        # exactly once per (forward, ratio bucket); int input has already
        # been synced by the upper-layer broadcast builder. Used by the
        # kept for metadata consumers that still need the scalar absolute start.
        if isinstance(positions, torch.Tensor):
            sp_int = int(positions.reshape(-1)[0].item())
        else:
            sp_int = int(positions)

        # Single contract guard for the entire prefill stack. Every downstream
        # builder consumes per-request tensors directly; missing varlen metadata
        # is a hard configuration error.
        use_varlen = True
        win = self.window_size
        # Under CP each rank holds only ``chunk_length`` tokens locally; the
        # full prefill sequence has length ``cp_ctx.seq_len_full``. Code that
        # needs the global length (SWA pool write after all-gather, KV-side
        # cu_seqlens) reads ``seqlen_full``; rank-local code keeps using
        # ``seqlen``.
        seqlen_full = cp_ctx.seq_len_full if cp_on else seqlen

        _msg = (
            "varlen prefill requires the upper layer to populate "
            "cu_seqlens / input_lengths / prefix_lengths / position_ids / "
            "req_id_per_token / sp_per_req / batch_size / max_seqlen_q."
        )
        assert cu_seqlens is not None, _msg
        assert input_lengths is not None, _msg
        assert prefix_lengths is not None, _msg
        assert position_ids is not None, _msg
        assert req_id_per_token is not None, _msg
        assert sp_per_req is not None, _msg
        assert batch_size > 0 and max_seqlen_q > 0, _msg

        cu_seqlens = _flat_1d(cu_seqlens)
        input_lengths = _flat_1d(input_lengths)
        prefix_lengths = _flat_1d(prefix_lengths)
        position_ids = _flat_1d(position_ids)
        req_id_per_token = _flat_1d(req_id_per_token)
        sp_per_req = _flat_1d(sp_per_req)
        if any_cont is None and reuse_common_meta is not None:
            any_cont = reuse_common_meta.any_cont
        if any_cont is None:
            if prefix_lengths.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "CUDA Graph prefill metadata requires host-derived any_cont"
                )
            any_cont = bool((prefix_lengths > 0).any().item())
        else:
            any_cont = bool(any_cont)
        assert (
            position_ids.numel() == seqlen
        ), f"position_ids must be flat [T_total={seqlen}], got {position_ids.shape}"
        assert (
            req_id_per_token.numel() == seqlen
        ), f"req_id_per_token must be flat [T_total={seqlen}], got {req_id_per_token.shape}"
        assert (
            cu_seqlens.numel() == batch_size + 1
        ), f"cu_seqlens must be [B+1={batch_size + 1}], got {cu_seqlens.shape}"
        assert (
            input_lengths.numel() == batch_size
        ), f"input_lengths must be [B={batch_size}], got {input_lengths.shape}"
        assert (
            prefix_lengths.numel() == batch_size
        ), f"prefix_lengths must be [B={batch_size}], got {prefix_lengths.shape}"
        assert (
            sp_per_req.numel() == batch_size
        ), f"sp_per_req must be [B={batch_size}], got {sp_per_req.shape}"

        can_reuse_freqs = (
            reuse_freqs_meta is not None
            and reuse_freqs_meta.freqs_cis_source_id == id(self.freqs_cis)
        )
        position_ids_eff: Optional[torch.Tensor] = None
        if reuse_common_meta is None or not can_reuse_freqs:
            position_ids_eff = position_ids
            if cp_on:
                assert cp_ctx is not None
                position_ids_eff = _flat_1d(
                    cp_ctx.global_positions.to(device=device, dtype=torch.long)
                )
        if reuse_common_meta is None:
            # Per-token absolute-position RoPE gather. For B==1 contiguous this
            # is bit-equal to the retired scalar slice; for B>1 it is the only
            # correct option since requests interleave on the flat token axis.
            with record_function_range("dsv4.fp8.meta.varlen.freqs_topk"):
                assert position_ids_eff is not None
                freqs_cis = self.freqs_cis.index_select(
                    0,
                    position_ids_eff.to(device=self.freqs_cis.device, dtype=torch.long),
                )
                from rtp_llm.models_py.modules.dsv4.fp8 import (
                    _swa_ops_triton as _swa_ops,
                )

                cu_seqlens_for_k = cu_seqlens
                if cp_on:
                    assert cp_ctx is not None
                    if cp_ctx.cu_seqlens_global is not None:
                        cu_seqlens_for_k = _flat_1d(
                            cp_ctx.cu_seqlens_global.to(
                                device=device, dtype=torch.int32
                            )
                        )
                topk_idxs, topk_length_kv_full = (
                    _swa_ops.compute_window_topk_and_length_varlen(
                        win,
                        cu_seqlens_for_k,
                        position_ids_eff,
                        prefix_lengths,
                        req_id_per_token,
                    )
                )
            with record_function_range("dsv4.fp8.meta.swa_varlen"):
                swa_meta = self._build_swa_prefill_meta_varlen(
                    seqlen=seqlen,
                    device=device,
                    any_cont=any_cont,
                    batch_size=batch_size,
                    cu_seqlens=cu_seqlens,
                    input_lengths=input_lengths,
                    prefix_lengths=prefix_lengths,
                    position_ids=position_ids,
                    req_id_per_token=req_id_per_token,
                    topk_length_kv_full=topk_length_kv_full,
                )
            # A device-side fill remains valid during CUDA Graph capture.
            row_seqlens_full = torch.full(
                (1,), seqlen_full, device=device, dtype=torch.long
            )
        else:
            if self.compress_ratio == 0:
                raise ValueError(
                    "SWA-only metadata must be the common source, not a reuse target"
                )
            self._validate_reusable_prefill_common(
                reuse_common_meta,
                seqlen=seqlen,
                seqlen_full=seqlen_full,
                rd=rd,
                device=device,
                cp_ctx=cp_ctx,
                sp_int=sp_int,
                win=win,
                sp_per_req=sp_per_req,
                cu_seqlens=cu_seqlens,
                input_lengths=input_lengths,
                prefix_lengths=prefix_lengths,
                position_ids=position_ids,
                req_id_per_token=req_id_per_token,
            )
            if can_reuse_freqs:
                assert reuse_freqs_meta is not None
                freqs_cis = reuse_freqs_meta.freqs_cis
            else:
                assert position_ids_eff is not None
                with record_function_range("dsv4.fp8.meta.varlen.freqs"):
                    freqs_cis = self.freqs_cis.index_select(
                        0,
                        position_ids_eff.to(
                            device=self.freqs_cis.device, dtype=torch.long
                        ),
                    )
            topk_idxs = reuse_common_meta.topk_idxs
            any_cont = reuse_common_meta.any_cont
            row_seqlens_full = reuse_common_meta.row_seqlens_full
            source_swa = reuse_common_meta.swa_meta
            assert source_swa is not None
            swa_meta = SwaPrefillMeta(
                slot_mapping=source_swa.slot_mapping,
                query_start_loc=source_swa.query_start_loc,
                combined_seq_lens=source_swa.combined_seq_lens,
                topk_length_kv_full=source_swa.topk_length_kv_full,
                combined_gather_lens=None,
                combined_gather_len_max=0,
                M=0,
                cache_seq_lens=None,
                cache_gather_lens=None,
                prefix_len_max=0,
                combined_indices=None,
                combined_lens=None,
                slot_in_flat=None,
                cache_slot_mapping=None,
                slot_compaction=source_swa.slot_compaction,
                cache_compaction=None,
            )

        # Bind freqs_cis to this layer's compressor / indexer chain
        # (idempotent — safe to call from both standalone and meta-broadcast paths).
        self._ensure_freqs_cis_bound()

        # ``use_varlen`` stays explicit because lower builders share one
        # metadata contract.
        csa_meta: Optional[CsaPrefillMeta] = None
        hca_meta: Optional[HcaPrefillMeta] = None
        if self.compress_ratio == 4:
            with record_function_range("dsv4.fp8.meta.csa"):
                csa_meta = self._build_csa_prefill_meta(
                    seqlen,
                    sp_int,
                    device,
                    use_varlen=use_varlen,
                    batch_size=batch_size,
                    cu_seqlens=cu_seqlens,
                    input_lengths=input_lengths,
                    prefix_lengths=prefix_lengths,
                    sp_per_req=sp_per_req,
                    position_ids=position_ids,
                    req_id_per_token=req_id_per_token,
                    max_seqlen_q=max_seqlen_q,
                    has_prefix=any_cont,
                )
        elif self.compress_ratio == 128:
            with record_function_range("dsv4.fp8.meta.hca"):
                hca_meta = self._build_hca_prefill_meta(
                    seqlen,
                    sp_int,
                    device,
                    use_varlen=use_varlen,
                    batch_size=batch_size,
                    cu_seqlens=cu_seqlens,
                    input_lengths=input_lengths,
                    prefix_lengths=prefix_lengths,
                    sp_per_req=sp_per_req,
                    position_ids=position_ids,
                    req_id_per_token=req_id_per_token,
                    max_seqlen_q=max_seqlen_q,
                    has_prefix=any_cont,
                )

        return PrefillMeta(
            seqlen=seqlen,
            seqlen_full=seqlen_full,
            rd=rd,
            device=device,
            cp_ctx=cp_ctx,
            cp_on=cp_on,
            freqs_cis=freqs_cis,
            topk_idxs=topk_idxs,
            sp_int=sp_int,
            any_cont=any_cont,
            row_seqlens_full=row_seqlens_full,
            use_varlen=use_varlen,
            sp_per_req=sp_per_req,
            cu_seqlens=cu_seqlens,
            batch_size=batch_size,
            input_lengths=input_lengths,
            prefix_lengths=prefix_lengths,
            position_ids=position_ids,
            req_id_per_token=req_id_per_token,
            max_seqlen_q=max_seqlen_q,
            swa_meta=swa_meta,
            csa_meta=csa_meta,
            hca_meta=hca_meta,
            freqs_cis_source_id=id(self.freqs_cis),
        )

    def _build_csa_prefill_meta(
        self,
        seqlen: int,
        sp_int: int,
        device: torch.device,
        *,
        use_varlen: bool,
        batch_size: int = 1,
        cu_seqlens: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        prefix_lengths: Optional[torch.Tensor] = None,
        sp_per_req: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        req_id_per_token: Optional[torch.Tensor] = None,
        max_seqlen_q: int = 0,
        has_prefix: bool,
    ) -> CsaPrefillMeta:
        """Build CSA-layer per-call metadata: indexer prepare + main CSA
        compressor prepare_metadata.

        Pool context binding: this method runs from the broadcast-meta
        path (``forward_layers`` → ``_build_and_propagate_prefill_meta``)
        BEFORE the per-layer ``_set_compressor_pool_context`` would
        otherwise fire inside ``_forward_prefill_internal_wrapper``, so we
        bind here ourselves. Without this bind the indexer's
        ``self._kv_block_table`` / ``_state_block_table`` are still None
        and the hoist inside ``IndexerFP8.prepare`` silently no-ops —
        ``compressor_meta`` comes back as None and the per-call slot
        mapping (~20 small kernels per CSA layer) ends up rebuilt on the
        hot path between the FP32 SGEMM and ``_save_partial_states_kernel``.

        One bind covers both halves: ``_set_compressor_pool_context``
        wires the indexer AND the host compressor in the same call, and
        ``_build_compressor_meta`` is inlined here so we don't redundantly
        re-bind for the second meta. No try/finally — if something below
        raises we want the process to die instead of leaking a stale
        binding into the next call.
        """
        from rtp_llm.platforms.ppu.models.dsv4.impl.fp8.compressor import build_prepare_metadata_args
        from rtp_llm.platforms.ppu.models.dsv4.impl.fp8.indexer import IndexerFP8
        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import INDEXER_KV

        assert isinstance(self.indexer, IndexerFP8), "CSA layer requires IndexerFP8"

        idx_bt = (
            self._block_tables_by_type.get(INDEXER_KV)
            if self._block_tables_by_type is not None
            else None
        )
        idx_eb = self._pool_entries_per_block(INDEXER_KV)

        with record_function_range("dsv4.fp8.meta.csa.bind_pool"):
            self._set_compressor_pool_context()
        with record_function_range("dsv4.fp8.meta.csa.indexer_prepare"):
            indexer_meta = self.indexer.prepare(
                bsz=1,
                seqlen=seqlen,
                sp_int=sp_int,
                device=device,
                kv_block_table=idx_bt,
                kv_eb=idx_eb,
                use_varlen=use_varlen,
                batch_size=batch_size,
                cu_seqlens=cu_seqlens,
                input_lengths=input_lengths,
                prefix_lengths=prefix_lengths,
                position_ids=position_ids,
                req_id_per_token=req_id_per_token,
                max_seqlen_q=max_seqlen_q,
                has_prefix=has_prefix,
            )
        cp_ctx_local = getattr(self, "_cp_ctx", None)
        cp_active = cp_ctx_local is not None and cp_ctx_local.cp_size > 1
        if cp_active:
            # Under CP the main compressor all-gathers KV/score to the full
            # global sequence. Build the matching full-sequence metadata once
            # per ratio bucket here (the prefill_meta broadcast path) instead
            # of rebuilding it inside every layer's compressor.forward.
            assert cp_ctx_local is not None
            with record_function_range("dsv4.fp8.meta.csa.cp_compressor_prepare"):
                (
                    cp_positions,
                    cp_b_idx,
                    cp_seq_start_per_req,
                    cp_cu_seq_per_req,
                ) = build_cp_full_prefill_positions(cp_ctx_local, device)
                assert cp_ctx_local.input_lengths_global is not None
                compressor_meta = self.compressor.prepare_metadata(
                    cp_positions,
                    cp_b_idx,
                    has_prefix=has_prefix,
                    is_batched=True,
                    seq_start_per_req=cp_seq_start_per_req,
                    cu_seq_per_req=cp_cu_seq_per_req,
                )
        else:
            # Inline (vs. calling ``_build_compressor_meta``) is load-bearing:
            # it shares the surrounding pool bind with ``indexer.prepare`` so
            # the indexer's nested compressor-meta hoist fires under one bind.
            with record_function_range("dsv4.fp8.meta.csa.compressor_prepare"):
                cmp_args = build_prepare_metadata_args(
                    use_varlen=use_varlen,
                    has_prefix=has_prefix,
                    device=device,
                    sp_int=sp_int,
                    seqlen=seqlen,
                    position_ids=position_ids,
                    req_id_per_token=req_id_per_token,
                    seq_start_per_req=sp_per_req,
                    cu_seqlens=cu_seqlens,
                )
                compressor_meta = self.compressor.prepare_metadata(**cmp_args)
        with record_function_range("dsv4.fp8.meta.csa.clear_pool"):
            self._clear_compressor_pool_context()

        with record_function_range("dsv4.fp8.meta.csa.workspace"):
            workspace_meta = self._build_workspace_meta(
                seqlen,
                sp_int,
                device,
                with_dense_cmp_topk=False,
                use_varlen=use_varlen,
                batch_size=batch_size,
                cu_seqlens=cu_seqlens,
                input_lengths=input_lengths,
                prefix_lengths=prefix_lengths,
                sp_per_req=sp_per_req,
                position_ids=position_ids,
                req_id_per_token=req_id_per_token,
                max_seqlen_q=max_seqlen_q,
            )
        return CsaPrefillMeta(
            indexer_meta=indexer_meta,
            compressor_meta=compressor_meta,
            workspace_meta=workspace_meta,
        )

    def _build_compressor_meta(
        self,
        seqlen: int,
        sp_int: int,
        device: torch.device,
        *,
        use_varlen: bool,
        batch_size: int = 1,
        cu_seqlens: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        prefix_lengths: Optional[torch.Tensor] = None,
        sp_per_req: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        req_id_per_token: Optional[torch.Tensor] = None,
        max_seqlen_q: int = 0,
        has_prefix: bool,
    ):
        """Run the main compressor's ``prepare_metadata`` with its pool
        context temporarily bound. Returns ``CompressorMeta``. The pool
        context binding (``_set_compressor_pool_context``) reads CSA_KV/
        CSA_STATE for ratio=4 and HCA_KV/HCA_STATE for ratio=128 based
        on ``self.compress_ratio``.

        ``use_varlen`` is required — set by ``_build_shared_prefill_meta``
        (the single env-read point + contract guard for the whole prefill
        stack). UT helpers must pass it explicitly so a missing kwarg
        surfaces as ``TypeError`` instead of silently picking up the
        ambient env.
        """
        from rtp_llm.platforms.ppu.models.dsv4.impl.fp8.compressor import build_prepare_metadata_args

        cmp_args = build_prepare_metadata_args(
            use_varlen=use_varlen,
            has_prefix=has_prefix,
            device=device,
            sp_int=sp_int,
            seqlen=seqlen,
            position_ids=position_ids,
            req_id_per_token=req_id_per_token,
            seq_start_per_req=sp_per_req,
            cu_seqlens=cu_seqlens,
        )
        self._set_compressor_pool_context()
        try:
            return self.compressor.prepare_metadata(**cmp_args)
        finally:
            self._clear_compressor_pool_context()

    def _prefill_common_setup(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> PrefillMeta:
        """Return the shared prefill meta broadcast by ``forward_layers``.

        Standalone FP8 prefill (calling this layer's prefill forward WITHOUT the
        upper-layer broadcast) is UNSUPPORTED: the per-forward union
        ``PrefillWorkspace`` is allocated once in
        ``prefill/forward.py::forward_layers`` and threaded in via
        ``build_and_propagate_prefill_meta_fp8`` (``meta._replace(workspace=…)``).
        A standalone call would carry ``workspace=None`` and the deferred Q
        materialization (:meth:`_materialize_prefill_q`) would have no backing
        buffer. We fail loud HERE — at the point the contract is violated —
        rather than NoneType-deref deep in Q materialization, and we do NOT
        fabricate a fallback workspace (single authoritative buffer, owned by
        the forward; see the ``workspace`` contract in ``cp.PrefillWorkspace``).
        In production every ``AttentionFP8`` layer always receives the broadcast
        (``forward_layers`` runs it before the layer loop), so this branch is
        unreachable there.
        """
        meta = self._prefill_meta_shared
        if meta is None:
            raise RuntimeError(
                "AttentionFP8 prefill requires the per-forward workspace "
                "broadcast from forward_layers (build_and_propagate_prefill_"
                "meta_fp8); standalone prefill with no _prefill_meta_shared is "
                "unsupported."
            )
        return meta

    def _prefill_compute_qkv(
        self,
        x: torch.Tensor,
        common: PrefillMeta,
        shared_input_quant: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> PrefillQKV:
        """Q/KV path — RMSNorm + LoRA Q + KV linears + fused RMSNorm-RoPE.

        Internally uses ``[1, T, ...]`` so ``fused_rmsnorm_rope`` sees the
        ``(B, S, …)`` layout it expects. Returned tensors keep the 3D
        shape because downstream pool/compressor helpers rely on it.
        """
        x_3d = x.unsqueeze(0)
        rd = common.rd
        if self._can_reuse_qkv_input_quant():
            if shared_input_quant is None:
                with record_function_range("dsv4.fp8.attn.qkv.shared_input_quant"):
                    x_2d = x_3d.reshape(-1, x_3d.shape[-1])
                    shared_input_quant = self.wq_a.quantize_input(x_2d)
        else:
            shared_input_quant = None

        def compute_qr() -> torch.Tensor:
            # q_lora_a + norm only — small ``[1, T, q_lora_rank]`` (fed to the
            # indexer). The big ``q_lora_b`` + RoPE are deferred to
            # ``_materialize_prefill_q`` so the 16 GiB Q buffer can reuse the
            # union workspace storage after the compressors finish.
            with record_function_range("dsv4.fp8.attn.qkv.q_lora_a_norm"):
                if shared_input_quant is not None:
                    q_proj = self._lin_from_shared_quant(
                        self.wq_a, shared_input_quant, x_3d.shape
                    )
                else:
                    q_proj = self._lin(self.wq_a, x_3d)
                return self._rmsnorm_weighted(
                    q_proj, self.q_norm
                )  # [1, T, q_lora_rank]

        def compute_kv() -> torch.Tensor:
            with record_function_range("dsv4.fp8.attn.qkv.kv_proj_rope"):
                if shared_input_quant is not None:
                    kv_in = self._lin_from_shared_quant(
                        self.wkv, shared_input_quant, x_3d.shape
                    )
                else:
                    kv_in = self._lin(self.wkv, x_3d)
                return fused_rmsnorm_rope(
                    kv_in, self.kv_norm, common.freqs_cis, rd, eps=self.eps
                )

        qr = compute_qr()
        kv = compute_kv()
        if common.cp_on:
            with record_function_range("dsv4.fp8.attn.qkv.cp_gather_varlen"):
                with record_function_range(
                    "dsv4.fp8.attn.swa_kv_full.cp_gather_varlen"
                ):
                    kv_flat = kv.reshape(kv.size(0) * kv.size(1), *kv.shape[2:])
                    kv_full_flat = cp_all_gather_full_varlen(
                        kv_flat,
                        common.cp_ctx,
                        profile_name=(
                            f"dsv4.cp.all_gather.L{self.layer_id:02d}."
                            "swa_kv_full.varlen"
                        ),
                    )
                    kv_full = kv_full_flat.unsqueeze(0)
        else:
            kv_full = kv

        return PrefillQKV(
            qr=qr.squeeze(0),
            q=None,  # deferred to _materialize_prefill_q
            kv_full=kv_full.squeeze(0),
        )

    def _materialize_prefill_q(
        self, qkv: PrefillQKV, common: PrefillMeta
    ) -> PrefillQKV:
        """Compute the deferred dense Q (``q_lora_b`` + RMSNorm-RoPE) into the
        forward workspace's Q slice and return ``qkv`` with ``q`` filled.

        Deferred until both compressors have drained their CP gather/restore
        buffers — those alias the same union storage as Q (see
        ``PrefillWorkspace``). Idempotent. Mirrors the ``[1, T, ...]`` layout the
        original ``compute_q`` used so ``fused_rmsnorm_rope`` sees ``(B, S, …)``.
        """
        if qkv.q is not None:
            return qkv
        assert common.workspace is not None, "prefill workspace not bound"
        qr_3d = qkv.qr.unsqueeze(0)  # [1, T, q_lora_rank]
        with record_function_range("dsv4.fp8.attn.qkv.q_lora_b_rope"):
            seqlen = int(qr_3d.shape[1])
            q_out = common.workspace.prefill_q(seqlen)
            q_local_flat = self._lin(
                self.wq_b,
                qr_3d,
                out=q_out.view(seqlen, self.n_heads * self.head_dim),
            )
            assert q_local_flat.data_ptr() == q_out.data_ptr()
            q_local = q_local_flat.view(1, seqlen, self.n_heads, self.head_dim)
            q_local = fused_rmsnorm_rope(
                q_local, None, common.freqs_cis, common.rd, eps=self.eps, out=q_local
            )
        return qkv._replace(q=q_local.squeeze(0))

    def _ensure_prefill_kv_full(
        self, qkv: PrefillQKV, common: PrefillMeta
    ) -> PrefillQKV:
        assert qkv.kv_full is not None, "PrefillQKV must carry materialized kv_full"
        return qkv

    def _prefill_write_swa_fp8_paged(
        self, common: PrefillMeta, kv_full: torch.Tensor
    ) -> None:
        """Single-launch quantize + insert into the FP8 SWA pool, using
        the paged ``slot_mapping`` pre-built in ``common.swa_meta``.

        The paged formula matches the decode-side dequant kernel. For large
        physical blocks, only the SWA ring tail before a physical boundary or
        request end is writable; entries with ``slot=-1`` are skipped. No-op
        on warmup (``swa_meta`` write fields are ``None``).
        """
        from rtp_llm.models_py.modules.dsv4.fp8 import _swa_kv_insert_triton as _ins
        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import SWA_KV

        meta = common.swa_meta
        if meta is None or meta.slot_mapping is None:
            return
        cp_byte_sliced = self._swa_cp_byte_sliced()
        packed_3d = None if cp_byte_sliced else self._pool_view_3d_fp8(SWA_KV)
        raw_u8 = self._pool_raw_u8(SWA_KV) if cp_byte_sliced else None
        if (cp_byte_sliced and raw_u8 is None) or (
            not cp_byte_sliced and packed_3d is None
        ):
            return

        k_bf16 = kv_full.reshape(-1, self.head_dim)
        if k_bf16.dtype != torch.bfloat16:
            k_bf16 = k_bf16.to(torch.bfloat16)
        with record_function_range("dsv4.fp8.attn.swa.quant_insert"):
            if cp_byte_sliced:
                cp_ctx = common.cp_ctx
                assert cp_ctx is not None
                assert meta.slot_compaction is not None
                _ins.quantize_and_insert_k_cache_cp_byte_sliced(
                    k_bf16,
                    raw_u8,
                    meta.slot_mapping,
                    full_entries_per_block=self._swa_entries_per_block(),
                    cp_rank=int(cp_ctx.cp_rank),
                    cp_size=int(cp_ctx.cp_size),
                    compaction=meta.slot_compaction,
                )
            else:
                _ins.quantize_and_insert_k_cache(k_bf16, packed_3d, meta.slot_mapping)

    def _flash_mla_sparse_fwd_chunked_projected(
        self,
        *,
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        topk_length: torch.Tensor,
        freqs_cis: torch.Tensor,
        prefill_workspace: Optional[PrefillWorkspace],
        profile_name: str,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run sparse prefill attention in Q chunks and project immediately.

        ``q`` is the live view of the per-forward ``PrefillWorkspace`` Q
        region. Keeping the chunking here (instead of copying Q into another
        scratch buffer) preserves that storage reuse while bounding the only
        large attention temporary to ``[q_chunk, n_heads, head_dim]``. Sparse
        prefill has no dependency across Q rows, so slicing Q, indices,
        topk-length and RoPE frequencies on the same boundaries is equivalent
        to a full launch.

        The projected result is written directly into the final contiguous
        ``[T, dim]`` tensor. The unprojected full ``[T, H, D]`` output is never
        allocated. A caller may preallocate ``out`` to overlap its allocation
        with preceding GPU work; the same tensor is returned after an in-place
        all-reduce. The caller must keep ``kv`` alive until this method returns.
        """
        assert prefill_workspace is not None, "prefill workspace not bound"
        s_q = int(q.shape[0])
        assert s_q > 0, "streaming FlashMLA prefill requires at least one Q row"
        q_workspace = prefill_workspace.prefill_q(s_q)
        assert q.dtype == torch.bfloat16, f"prefill Q must be bf16, got {q.dtype}"
        assert q.is_contiguous(), "prefill Q workspace view must be contiguous"
        assert tuple(q.shape[1:]) == (self.n_heads, self.head_dim), (
            "prefill Q shape mismatch: expected "
            f"[T, {self.n_heads}, {self.head_dim}], got {tuple(q.shape)}"
        )
        assert q.numel() == q_workspace.numel(), (
            "prefill workspace Q shape mismatch: "
            f"q.numel={q.numel()} != workspace.numel={q_workspace.numel()}"
        )
        assert q.data_ptr() == q_workspace.data_ptr(), (
            "prefill Q must reuse PrefillWorkspace storage; got different "
            "base pointers"
        )
        assert (
            int(indices.shape[0]) == s_q
        ), f"indices rows ({indices.shape[0]}) != Q rows ({s_q})"
        assert topk_length.ndim == 1 and int(topk_length.shape[0]) == s_q, (
            "topk_length must be one-dimensional with one entry per Q row; "
            f"got shape {tuple(topk_length.shape)} for {s_q} Q rows"
        )
        assert (
            int(freqs_cis.shape[0]) == s_q
        ), f"RoPE rows ({freqs_cis.shape[0]}) != Q rows ({s_q})"

        from rtp_llm.models_py.utils.flash_mla_legacy import (
            flash_mla_sparse_fwd,  # type: ignore[import-not-found]
        )

        chunk_rows = min(_FLASH_MLA_SPARSE_Q_CHUNK, s_q)
        if out is None:
            out = torch.empty(s_q, self.dim, dtype=torch.bfloat16, device=q.device)
        else:
            assert (
                out.dtype == torch.bfloat16
            ), f"projected output must be bf16, got {out.dtype}"
            assert (
                out.device == q.device
            ), f"projected output device ({out.device}) != Q device ({q.device})"
            assert out.is_contiguous(), "projected output must be contiguous"
            assert tuple(out.shape) == (s_q, self.dim), (
                "projected output shape mismatch: expected "
                f"({s_q}, {self.dim}), got {tuple(out.shape)}"
            )

        for start in range(0, s_q, chunk_rows):
            end = min(start + chunk_rows, s_q)
            with record_function_range(profile_name):
                o_part, _, _ = flash_mla_sparse_fwd(
                    q=q[start:end],
                    kv=kv,
                    indices=indices[start:end],
                    sm_scale=self.softmax_scale,
                    attn_sink=self.attn_sink,
                    topk_length=topk_length[start:end],
                )
            if self.compress_ratio == 4 and _rt.should_record_layer(self.layer_id):
                _rt.record_if_level(2, f"L{self.layer_id:02d}_csa_flash_out", o_part)
            with record_function_range("dsv4.fp8.attn.prefill.output_proj"):
                self._prefill_output_proj_into(
                    o_part,
                    freqs_cis[start:end],
                    out=out[start:end, :],
                )
            dispose_tensor(o_part)

        self._prefill_output_all_reduce(out)
        if self.compress_ratio == 4 and _rt.should_record_layer(self.layer_id):
            _rt.record_if_level(2, f"L{self.layer_id:02d}_csa_projected_out", out)
        return out

    def _attn_fp8_swa_via_kv_full(
        self,
        qkv: PrefillQKV,
        common: PrefillMeta,
    ) -> torch.Tensor:
        """Chunked sparse_fwd over BF16 ``kv_full`` — no FP8 round-trip.

        Used by:
          * cold prefill (``sp == 0``) — pool capacity (``2 * eb``)
            can't hold the full prefill anyway; attend over BF16 K
            we just computed.
          * warmup forward (``self._kv_cache is None``) — pool not yet
            allocated; needed for the framework's dry-run shape inference.

        Each Q chunk is output-projected into the final ``[T, dim]`` result
        before the next chunk starts. Caller is responsible for pre-writing
        the new K to the FP8 SWA pool (via
        ``_prefill_write_swa_fp8_paged``) for future decode reads — write
        order doesn't matter here since this path doesn't read from the pool.
        """
        assert (
            qkv.q is not None
        ), "_attn_fp8_swa_via_kv_full: prefill Q not materialized"
        meta = common.swa_meta
        assert meta is not None and meta.topk_length_kv_full is not None, (
            "FP8 SWA prefill requires common.swa_meta.topk_length_kv_full; "
            "built in _prefill_common_setup for FP8 SWA-only layers."
        )

        # ``topk_idxs`` shape:
        #   * varlen path → ``[T_total, win]`` flat-KV indices (per-request
        #     window, ``cu_seqlens[b] + local_pos`` baked in).
        #   * CP path → ``[1, T, win]``.
        # Either lands at flash_mla_sparse_fwd's ``[T, 1, win]`` indices
        # contract after the same ``squeeze + unsqueeze + cast`` chain.
        #
        # CP fresh prefill alignment: under cp_size > 1 ``qkv.q`` is
        # rank-local ``[T_local, H, D]`` and ``qkv.kv_full`` is the
        # all-gathered ``[seq_len_full, D]`` in GLOBAL request order. The
        # varlen topk builder receives CP global positions plus global
        # per-request cu_seqlens, so indices address the gathered KV for B>=1.
        if common.cp_on:
            assert (
                qkv.kv_full.size(0) == common.seqlen_full
            ), "CP gather should produce kv_full sized to seq_len_full"
        ti = common.topk_idxs
        if ti.dim() == 3:
            ti = ti.squeeze(0)
        indices = ti.unsqueeze(1).to(torch.int32)

        out = self._flash_mla_sparse_fwd_chunked_projected(
            q=qkv.q,
            kv=qkv.kv_full.unsqueeze(1),
            indices=indices,
            topk_length=meta.topk_length_kv_full,
            freqs_cis=common.freqs_cis,
            prefill_workspace=common.workspace,
            profile_name="dsv4.fp8.attn.swa.flash_mla_kv_full",
        )
        # kv_full has no remaining consumer after all attention chunks drain.
        dispose_tensor(qkv.kv_full)
        return out

    def _attn_fp8_swa_via_concat(
        self,
        qkv: PrefillQKV,
        common: PrefillMeta,
    ) -> torch.Tensor:
        """Continuation prefill (any req with prefix > 0): prefix-from-cache
        + new-K-bf16 concat. Varlen-aware ``[B, M, D]`` workspace.

        Pipeline (per request b, ``S_b = input_lengths[b]``,
        ``P_b = min(prefix_lengths[b], win-1)``):
          1. ``dequantize_and_gather_k_cache`` reads each request's trailing
             ``P_b`` cached tokens (SWA prefix tail, abs pos ``[sp_b - P_b, sp_b)``)
             into ``workspace[b, :P_b, :]``. Already batched via
             ``cache_seq_lens`` / ``cache_gather_lens`` ``[B]`` tensors.
          2. Vectorized scatter places the freshly computed new K into
             ``workspace[b, P_b:P_b+S_b, :]`` — no per-request Python loop.
          3. Chunked ``flash_mla_sparse_fwd`` over
             ``workspace.view(B*M, 1, D)``; ``combined_indices`` already
             carries ``M*batch_idx + slot`` from
             ``combine_topk_swa_indices``. Each chunk is immediately output-
             projected into the final ``[T, dim]`` result.

        """
        assert qkv.q is not None, "_attn_fp8_swa_via_concat: prefill Q not materialized"
        from rtp_llm.models_py.modules.dsv4.fp8 import _swa_dequant_triton as _swa_dq
        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import SWA_KV

        meta = common.swa_meta
        assert (
            meta is not None
            and meta.combined_indices is not None
            and meta.cache_seq_lens is not None
            and meta.cache_gather_lens is not None
        ), (
            "via_concat path requires swa_meta with cache_* and combined_* "
            "fields (only built when any request has prefix > 0)"
        )

        cp_byte_sliced = self._swa_cp_byte_sliced()
        packed_3d = None if cp_byte_sliced else self._pool_view_3d_fp8(SWA_KV)
        raw_u8 = self._pool_raw_u8(SWA_KV) if cp_byte_sliced else None
        if cp_byte_sliced:
            assert raw_u8 is not None, "byte-sliced FP8 SWA pool unavailable"
        else:
            assert packed_3d is not None, "FP8 SWA pool unavailable"
        D = self.head_dim

        assert common.use_varlen, "DSV4 FP8 SWA concat requires varlen metadata"
        B = common.batch_size

        # E5b: meta.combined_indices is built from valid prefix/new-K slots only.
        workspace = torch.empty(
            (B, meta.M, D),
            dtype=torch.bfloat16,
            device=qkv.q.device,
        )

        # 1. Prefix tail dequant through precomputed global SWA slots.
        if meta.prefix_len_max > 0:
            with record_function_range("dsv4.fp8.attn.swa_concat.gather_prefix"):
                assert meta.cache_slot_mapping is not None
                if cp_byte_sliced:
                    cp_ctx = common.cp_ctx
                    assert cp_ctx is not None
                    assert meta.cache_compaction is not None
                    _swa_dq.dequantize_and_gather_k_cache_slots_cp_byte_sliced(
                        out=workspace,
                        k_cache_raw=raw_u8,
                        slot_mapping=meta.cache_slot_mapping,
                        gather_lens=meta.cache_gather_lens,
                        offset=0,
                        full_entries_per_block=self._swa_entries_per_block(),
                        cp_rank=int(cp_ctx.cp_rank),
                        cp_size=int(cp_ctx.cp_size),
                        compaction=meta.cache_compaction,
                    )
                else:
                    _swa_dq.dequantize_and_gather_k_cache_slots(
                        out=workspace,
                        k_cache=packed_3d,
                        slot_mapping=meta.cache_slot_mapping,
                        gather_lens=meta.cache_gather_lens,
                        offset=0,
                    )

        # 2. New K BF16 overlay.
        kv_bf16 = qkv.kv_full.to(torch.bfloat16)
        # ``meta.slot_in_flat`` is pre-baked in ``_build_swa_prefill_meta_varlen``
        # (= ``req_id * M + min(prefix, win-1) + local_pos``). Single
        # ``index_copy_`` here — no per-layer casts / gathers / arith.
        assert (
            meta.slot_in_flat is not None
        ), "via_concat varlen path expects pre-baked slot_in_flat in swa_meta"
        with record_function_range("dsv4.fp8.attn.swa_concat.overlay_new_k"):
            workspace.view(B * meta.M, D).index_copy_(0, meta.slot_in_flat, kv_bf16)
        # Free kv_full storage before flash_mla_sparse_fwd. After the
        # overlay nothing else reads it on this path; the NamedTuple ref
        # would otherwise keep it alive through the sparse-attn workspace
        # alloc — ~1.1 GiB peak overlap at 1M ctx.
        dispose_tensor(kv_bf16)
        dispose_tensor(qkv.kv_full)

        # 3. Chunked flash_mla_sparse_fwd over the [B*M] flat KV view.
        return self._flash_mla_sparse_fwd_chunked_projected(
            q=qkv.q,
            kv=workspace.view(B * meta.M, 1, D),
            indices=meta.combined_indices.unsqueeze(1),
            topk_length=meta.combined_lens,
            freqs_cis=common.freqs_cis,
            prefill_workspace=common.workspace,
            profile_name="dsv4.fp8.attn.swa_concat.flash_mla",
        )

    def _prefill_output_proj_into(
        self,
        o: torch.Tensor,
        freqs_cis: torch.Tensor,
        *,
        out: torch.Tensor,
    ) -> None:
        """Inverse-RoPE + grouped wo_a + wo_b into an existing ``[T, dim]`` tensor.

        The fused Triton kernel
        (``fused_inv_rope_fp8_quant``) emits the exact ``(fp8 [M,G,K],
        scale [M,G,K/512])`` layout ``deep_gemm.fp8_einsum`` consumes.
        """
        o_3d = o.view(-1, self.n_heads, self.head_dim)
        seqlen = o_3d.shape[0]
        if self.wo_a is not None:
            with record_function_range("dsv4.fp8.attn.out.provider_wo_a"):
                o_proj = self._wo_a_from_bf16(o_3d, freqs_cis, 1, seqlen)
        else:
            with record_function_range("dsv4.fp8.attn.out.fused_inv_rope_quant"):
                o_fp8, o_scale = fused_inv_rope_fp8_quant(
                    o_3d,
                    freqs_cis,
                    n_groups=self.n_groups,
                    heads_per_group=self.n_heads // self.n_groups,
                    nope_dim=self.head_dim - self.rope_head_dim,
                    rope_head_dim=self.rope_head_dim,
                )
            with record_function_range("dsv4.fp8.attn.out.wo_a_einsum"):
                o_proj = self._wo_a_einsum_from_fp8(o_fp8, o_scale, 1, seqlen)
        with record_function_range("dsv4.fp8.attn.out.wo_b"):
            wo_b_in = o_proj.flatten(2).reshape(seqlen, -1)
            self.wo_b(wo_b_in, out=out)

class CommitOnlyAttentionFP8(_BaseCommitOnlyAttentionFP8):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        q_lora_rank: int,
        head_dim: int,
        rope_head_dim: int,
        o_lora_rank: int,
        o_groups: int,
        window_size: int,
        compress_ratio: int,
        compress_rope_theta: float,
        rope_theta: float,
        rope_factor: float,
        beta_fast: int,
        beta_slow: int,
        original_seq_len: int,
        max_batch_size: int,
        max_seq_len: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        norm_eps: float = 1e-6,
        layer_weights: Optional[Dict[str, torch.Tensor]] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        platform_provider=None,
        indexer_factory=None,
    ):
        del compress_rope_theta, original_seq_len, max_batch_size, index_topk
        if layer_weights is None:
            raise ValueError("commit-only DSpARK attention requires layer weights")
        if int(compress_ratio) != 0:
            raise ValueError(
                "commit-only DSpARK attention supports SWA layers only, got "
                f"compress_ratio={compress_ratio}"
            )

        # Avoid the full constructor: proposal Q/O projections, compressor,
        # indexer, and their caches are unreachable from forward_commit.
        nn.Module.__init__(self)
        self.layer_id = int(layer_id)
        self.dim = int(dim)
        self.q_lora_rank = int(q_lora_rank)
        self.o_lora_rank = int(o_lora_rank)
        self.head_dim = int(head_dim)
        self.rope_head_dim = int(rope_head_dim)
        self.window_size = int(window_size)
        self.compress_ratio = 0
        self.eps = float(norm_eps)
        self.softmax_scale = self.head_dim**-0.5
        self.tp_size = int(tp_size)
        self.tp_rank = int(tp_rank)
        if self.tp_size <= 0:
            raise ValueError(f"invalid attention tp_size={self.tp_size}")
        if int(n_heads) % self.tp_size:
            raise ValueError(
                f"n_heads={n_heads} is not divisible by tp_size={self.tp_size}"
            )
        if int(o_groups) % self.tp_size:
            raise ValueError(
                f"o_groups={o_groups} is not divisible by tp_size={self.tp_size}"
            )
        self.n_heads = int(n_heads) // self.tp_size
        self.n_groups = int(o_groups) // self.tp_size

        from rtp_llm.utils.model_weight import W

        self.wkv = _v4_fp8_linear(
            layer_weights[W.v4_attn_wkv_w], layer_weights[W.v4_attn_wkv_s]
        )
        self.kv_norm = layer_weights[W.v4_attn_kv_norm]

        self.attn_sink = None
        self._platform_provider = platform_provider
        self._indexer_factory = indexer_factory
        self.wq_a = None
        self.wq_b = None
        self.wo_a = None
        self.wo_a_w = None
        self.wo_a_s = None
        self.wo_b = None
        self.q_norm = None
        self.compressor = None
        self.indexer = None

        self._rope_base = float(rope_theta)
        self._rope_o_seq_len = 0
        self._rope_factor = float(rope_factor)
        self._rope_beta_fast = int(beta_fast)
        self._rope_beta_slow = int(beta_slow)
        self._rope_dim = int(rope_head_dim)
        self._rope_max_seq_len = int(max_seq_len)
        self.freqs_cis = precompute_freqs_cis(
            self._rope_dim,
            self._rope_max_seq_len,
            self._rope_o_seq_len,
            self._rope_base,
            self._rope_factor,
            self._rope_beta_fast,
            self._rope_beta_slow,
        )
        self._fp8_decode_op: Optional[Any] = None
        self._cp_ctx: Optional[CPContext] = None
        self._prefill_meta_shared: Optional["PrefillMeta"] = None
        self._kv_cache: Optional[Any] = None
        self._block_tables_by_type: Optional[Dict[str, torch.Tensor]] = None

        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
            CSA_KV,
            CSA_STATE,
            HCA_KV,
            HCA_STATE,
            INDEXER_KV,
            INDEXER_STATE,
            SWA_KV,
        )

        idx_hd = int(index_head_dim)
        kv_spec = (torch.uint8, _DSV4_FP8_KV_ENTRY_BYTES)
        indexer_kv_spec = (torch.uint8, _DSV4_FP8_INDEXER_ENTRY_BYTES)
        self._pool_spec: Dict[str, tuple] = {
            SWA_KV: kv_spec,
            CSA_KV: kv_spec,
            HCA_KV: kv_spec,
            INDEXER_KV: indexer_kv_spec,
            CSA_STATE: (torch.float32, 4 * self.head_dim),
            HCA_STATE: (torch.float32, 2 * self.head_dim),
            INDEXER_STATE: (torch.float32, 4 * idx_hd),
        }
