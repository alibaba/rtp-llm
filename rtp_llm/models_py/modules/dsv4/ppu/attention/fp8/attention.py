"""PPU overrides; unchanged behavior is inherited from the shared DSV4 implementation."""

from __future__ import annotations
from rtp_llm.models_py.modules.dsv4.fp8.attention import (
    _use_cp_cache_hit_raw_q_merge,
    _force_cp_cache_hit_raw_q_merge,
    _force_all_cp_raw_q_merge,
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
    WorkspaceMeta,
    CsaPrefillMeta,
    PrefillQKV,
)
import json
import logging
import os
import threading
from contextlib import contextmanager, suppress
from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
)
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
from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.models_py.modules.dsv4.ppu.chunk_env import dsv4_chunk_tokens_from_env
from rtp_llm.models_py.modules.dsv4.cp import (
    _CP_ROLE_MAIN,
    CPContext,
    build_cp_full_prefill_positions,
    cp_freqs_cis_local,
    cp_padded_local_kv_lens,
)
from rtp_llm.models_py.modules.dsv4.fp8._cp_attention_shard import prefer_raw_q_merge_attention_conservative
from rtp_llm.models_py.modules.dsv4.fp8._pool_reader import (
    CompressedKPoolReader,
    LocalPoolReader,
    make_compressed_k_pool_reader,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_cp_byte_sliced import (
    CPByteSlicedSlotCompaction,
    build_cp_byte_sliced_slot_compaction,
)
from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.compressor import CompressorFP8
from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.indexer import IndexerFP8
from rtp_llm.models_py.modules.dsv4.prefill_workspace import PrefillWorkspace
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
from rtp_llm.models_py.modules.dsv4.fp8.attention import PrefillMeta, _prepare_wo_a_stacked
from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8 as _BaseAttentionFP8
from rtp_llm.models_py.modules.dsv4.fp8.attention import CommitOnlyAttentionFP8 as _BaseCommitOnlyAttentionFP8

def _v4_fp8_linear(w: torch.Tensor, s: torch.Tensor, *, platform_provider=None):
    """Build a CudaFp8DeepGEMMLinear from raw V4 FP8 weight + scale tensors.

    Repacks the UE8M0 ``float8_e8m0fnu`` scale into DeepGEMM's int32
    TMA-aligned packed layout when needed. Framework descriptor path may
    deliver the scale already packed (dtype int32) — we no-op then."""

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
    return platform_provider.build_fp8_linear(_default_factory, w, s)

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

        self.wo_a = platform_provider.build_wo_a_fp8_linear(lambda *_args, **_kwargs: None, wo_a_w, wo_a_s, groups=n_groups_local, k_local=K_local)
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
            from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.compressor import CompressorFP8

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
                from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.indexer import IndexerFP8

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
        from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.decode.compute_qkv import decode_compute_qkv
        from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.decode.output_proj import decode_output_proj

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
        from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.decode.attention_kernels import attn_fp8_swa_paged
        from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.decode.decode_attn_metadata import get_or_build_sched_meta
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
        from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.decode.attention_kernels import attn_fp8_dual_paged
        from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.decode.decode_attn_metadata import get_or_build_sched_meta
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
        from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.indexer import IndexerFP8

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
        from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.indexer import IndexerFP8

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
        from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.compressor import build_prepare_metadata_args
        from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.indexer import IndexerFP8
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
        from rtp_llm.models_py.modules.dsv4.ppu.attention.fp8.compressor import build_prepare_metadata_args

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
        # ``_attn_fp8_swa_via_concat`` (inherited) always passes this; its
        # packed-prefix fields are SM120-only, so they stay empty and are
        # ignored here.
        backend_inputs=None,
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
