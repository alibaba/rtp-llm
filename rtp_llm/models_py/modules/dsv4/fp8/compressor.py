"""DeepSeek-V4 Compressor — FP8 KV pool path (per-token state pool).

Companion to ``compressor.py`` (BF16 path). Single class for both pool
flavors; ``head_dim`` selects the writer kernel + FP8 KV slot layout:

  * ``head_dim == 512`` (CSA / HCA): per-slot 584B striped layout
    (448 fp8 NoPE + 64 bf16 RoPE + 8 UE8M0 scales). Reader:
    ``flash_mla_sparse_fwd`` after dequant.

  * ``head_dim == 128`` (indexer compressor): per-slot 132B grouped
    layout (128 fp8 K + 4-byte fp32 scale). Reader: DeepGEMM
    ``fp8_paged_mqa_logits``.

Post-commit ``e76867719`` ("fix - align state size to 256") the C++
state pools (INDEXER_STATE / CSA_STATE / HCA_STATE) all use
``entries_per_block=256`` with ``fixed_blocks_per_req=2``: every token
gets its own slot. We mirror vLLM's ``DeepseekCompressor`` flow:

  1. ``_save_partial_states_kernel`` writes per-token (kv | score+ape)
     into the framework-allocated state pool.
  2. ``_fused_kv_compress_norm_rope_insert_*_attn`` self-skips
     non-boundary tokens, otherwise gathers the ``(1+overlap)*ratio``
     window from the state pool, does softmax → RMSNorm → RoPE → FP8
     UE8M0 quant → KV-pool slot store.

Public API kept stable so ``attention.py`` / ``indexer_fp8.py`` call
sites do not change:
  * ``set_pool_context(kv_view, kv_bt, kv_eb, state_view, state_bt,
    state_eb)`` — same 6-arg shape as the old ``PoolBackedModule``.
  * ``forward(x, start_pos, sequence_lengths=None)`` for prefill.
  * ``forward_decode_vectorized(x, start_pos)`` for batched decode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.models_py.modules.dsv4.cp import (
    CPContext,
    cp_all_gather_full_async,
    cp_should_gather,
    cp_wait_gather_full,
)
from rtp_llm.models_py.modules.dsv4.fp8._compressor_consts import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
    KV_ENTRY_BYTES,
    KV_HEAD_DIM,
)
from rtp_llm.models_py.modules.dsv4.fp8._compressor_vllm_triton import (
    build_cos_sin_cache,
    run_fused_compress_kv_write,
    run_save_partial_states,
)


@dataclass(frozen=True)
class CompressorMeta:
    """Pre-computed per-token launch metadata.

    Built once per (state_block_table, kv_block_table, positions, b_idx)
    tuple — typically by the attention layer just after ``set_pool_context``,
    so the math is amortized across both the host compressor and any nested
    indexer compressor that shares the same positions/b_idx layout.

    Fields are device tensors of length ``N_tok``:
      * ``positions``    : int64 absolute token positions
      * ``b_idx``        : int64 request index per token
      * ``state_slots``  : int64 state-pool slot per token (-1 = skip)
      * ``kv_slots``     : int64 KV-pool slot per token (-1 if non-boundary
                           or unallocated)
      * ``token_to_req`` : int32 alias of ``b_idx`` for the fused KV writer
      * ``is_batched``   : True when this meta packs multiple requests
                           (B>1) — the kernel's scalar ``seq_start`` raw
                           path is unsafe and ``_launch`` must disable it.
                           Validated byte-equal to raw_path=True for short
                           prefill by ``test_compressor_disable_raw_path``.
    """

    positions: torch.Tensor
    b_idx: torch.Tensor
    state_slots: torch.Tensor
    kv_slots: torch.Tensor
    token_to_req: torch.Tensor
    is_batched: bool = False
    # Phase-3a part 4c — varlen raw path. Populated only when is_batched;
    # otherwise the legacy scalar-seq_start path is used.
    #   seq_start_per_req[b] = abs position of req b's first new token (sp_b)
    #   cu_seq_per_req[b+1]  = end offset of req b in flat kv_flat axis
    seq_start_per_req: Optional[torch.Tensor] = None
    cu_seq_per_req: Optional[torch.Tensor] = None


class _CompressorNorm(nn.Module):
    """RMSNorm weight holder — bf16 (vLLM kernel reads bf16 weight)."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))


class CompressorFP8(nn.Module):
    """FP8 KV cache compressor — vLLM-aligned per-token state pool path.

    Compress ratio: CSA uses ratio=4 (overlap=True), HCA uses ratio=128
    (overlap=False). The indexer compressor follows the host attention
    layer's ratio (typically 4, overlap=True).
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        rope_head_dim: int,
        compress_ratio: int,
        max_batch_size: int,
        norm_eps: float = 1e-6,
        rotate: bool = False,
        compressor_weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """``compressor_weights`` is a 4-key dict ``{"ape", "wkv", "wgate",
        "norm"}`` extracted by the caller from ``layer_weights[W.v4_*compressor_*]``."""
        super().__init__()
        assert head_dim in (KV_HEAD_DIM, INDEXER_HEAD_DIM), (
            f"CompressorFP8 supports head_dim in {{{KV_HEAD_DIM}, "
            f"{INDEXER_HEAD_DIM}}}; got {head_dim}"
        )
        assert compressor_weights is not None, (
            "CompressorFP8 requires compressor_weights — meta-tensor / "
            "stand-alone construction is not supported (use the BF16 path)."
        )
        self.dim = dim
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        self.norm_eps = norm_eps
        coff = 1 + self.overlap
        self.coff = coff
        # Per-slot bytes of the FP8 KV pool entry (selects layout/kernel).
        self._pool_entry_bytes = (
            KV_ENTRY_BYTES if head_dim == KV_HEAD_DIM else INDEXER_ENTRY_BYTES
        )

        # ape/wkv/wgate stay FP32 — accumulation happens in FP32 inside the
        # state pool; vLLM keeps the same convention. Register ape as a
        # non-trainable Parameter so .to(device) follows the module.
        self.ape = nn.Parameter(
            compressor_weights["ape"].float().contiguous(), requires_grad=False
        )
        self.wkv = nn.Linear(dim, coff * head_dim, bias=False)
        self.wgate = nn.Linear(dim, coff * head_dim, bias=False)
        with torch.no_grad():
            self.wkv.weight = nn.Parameter(
                compressor_weights["wkv"].to(torch.bfloat16), requires_grad=False
            )
            self.wgate.weight = nn.Parameter(
                compressor_weights["wgate"].to(torch.bfloat16), requires_grad=False
            )
        self.norm = _CompressorNorm(head_dim)
        self.norm.weight = nn.Parameter(
            compressor_weights["norm"].to(torch.bfloat16),
            requires_grad=False,
        )

        # Fuse wkv + wgate into one bf16 weight matrix; saves one
        # GEMM launch per compressor decode call (~92/step at bs16).
        self._wkv_wgate_fused: Optional[torch.Tensor] = None
        self._fuse_wkv_wgate(coff)

        # Pool context — populated by attention's _set_compressor_pool_context.
        self._state_pool_3d: Optional[torch.Tensor] = None
        self._state_block_table: Optional[torch.Tensor] = None
        self._state_eb: int = 0
        self._kv_pool_3d: Optional[torch.Tensor] = None
        self._kv_block_table: Optional[torch.Tensor] = None
        self._kv_eb: int = 0

        # Legacy attribute kept for attention.py's cmp_T fallback (line 1583).
        self._kv_cache_t: int = 0

        # Cached cos_sin cache built from self.freqs_cis at first forward.
        self.freqs_cis: Optional[torch.Tensor] = None
        self._cos_sin_cache: Optional[torch.Tensor] = None
        self._cp_ctx: Optional[CPContext] = None
        # MOEDBG: caller (Attention / IndexerFP8) sets this to a name
        # prefix like ``"L02_attn_cmp"`` before forward and clears after;
        # _forward_prefill_body uses it as the rec name root. None / empty
        # string suppresses recording.
        self._dbg_prefix: Optional[str] = None

    def _fuse_wkv_wgate(self, coff: int) -> None:
        """Concat wkv + wgate along out-dim into one fused bf16 weight,
        then re-point ``wkv.weight`` / ``wgate.weight`` to views of the
        fused storage (zero memory overhead)."""
        with torch.no_grad():
            fused = torch.cat(
                [self.wkv.weight.data, self.wgate.weight.data], dim=0
            ).contiguous()
            self._wkv_wgate_fused = fused
            out_dim = coff * self.head_dim
            self.wkv.weight = nn.Parameter(fused[:out_dim], requires_grad=False)
            self.wgate.weight = nn.Parameter(fused[out_dim:], requires_grad=False)

    # ----------------------------------------------------------------------
    # Compatibility shims (kept to match the BF16 ``Compressor`` API surface
    # so callers don't need to special-case the FP8 class).
    # ----------------------------------------------------------------------
    def configure_kv_cache_shape(self, kv_cache_t: int) -> None:
        """Stores ``_kv_cache_t`` only as informational metadata so legacy
        readers (e.g. ``attention.py:1583`` ``cmp_T`` fallback) keep working.
        The FP8 path does NOT allocate any per-step bf16 cache from this."""
        self._kv_cache_t = int(kv_cache_t)

    def set_cp_ctx(self, cp_ctx: Optional[CPContext]) -> None:
        self._cp_ctx = cp_ctx

    # ----------------------------------------------------------------------
    # Pool context lifecycle (6-arg signature matches PoolBackedModule)
    # ----------------------------------------------------------------------
    def set_pool_context(
        self,
        kv_pool_view: Optional[torch.Tensor],
        kv_block_table: Optional[torch.Tensor],
        kv_eb: int,
        state_pool_view: Optional[torch.Tensor],
        state_block_table: Optional[torch.Tensor],
        state_eb: int,
    ) -> None:
        """Install framework pool views.

        ``kv_pool_view``    : 3D ``[num_blocks, kv_eb, ENTRY_BYTES]`` uint8
                              (FP8 KV pool, already TMA-padded if 584B).
        ``state_pool_view`` : 2D flat ``[total_slots, 2*coff*head_dim]`` fp32
                              from ``_pool_view``. Reshaped here to
                              ``[num_blocks, state_eb, 2*coff*head_dim]``.
        """
        self._kv_pool_3d = kv_pool_view
        self._kv_block_table = kv_block_table
        self._kv_eb = kv_eb

        if state_pool_view is not None and state_eb > 0:
            assert (
                state_pool_view.dim() == 2
            ), f"expected flat 2D state pool view, got {state_pool_view.shape}"
            total_slots, hidden = state_pool_view.shape
            assert total_slots % state_eb == 0, (
                f"state pool total_slots={total_slots} not divisible by "
                f"state_eb={state_eb}"
            )
            num_blocks = total_slots // state_eb
            self._state_pool_3d = state_pool_view.view(num_blocks, state_eb, hidden)
        else:
            self._state_pool_3d = None
        self._state_block_table = state_block_table
        self._state_eb = state_eb

    def clear_pool_context(self) -> None:
        self._state_pool_3d = None
        self._state_block_table = None
        self._state_eb = 0
        self._kv_pool_3d = None
        self._kv_block_table = None
        self._kv_eb = 0

    # ----------------------------------------------------------------------
    # Metadata preparation (call once per forward, OFF the hot path)
    # ----------------------------------------------------------------------
    def prepare_metadata(
        self,
        positions: torch.Tensor,  # [N] int64
        b_idx: torch.Tensor,  # [N] int64
        is_batched: bool = False,
        seq_start_per_req: Optional[torch.Tensor] = None,
        cu_seq_per_req: Optional[torch.Tensor] = None,
    ) -> CompressorMeta:
        """Compute slot mappings + token_to_req from current pool context.

        Pure function of ``(positions, b_idx, self._state_block_table,
        self._kv_block_table, self.compress_ratio, self._state_eb,
        self._kv_eb)`` — safe to call once per attention forward and reuse
        across the host compressor and any nested indexer compressor that
        shares the same positions/b_idx (when their pool context is bound).
        """
        assert (
            positions.dim() == 1
        ), f"positions must be flat [N], got {positions.shape}"
        assert b_idx.dim() == 1, f"b_idx must be flat [N], got {b_idx.shape}"
        assert (
            positions.numel() == b_idx.numel()
        ), f"positions/b_idx length mismatch: {positions.numel()} vs {b_idx.numel()}"
        with record_function_range("dsv4.fp8.compressor.meta.state_slots"):
            state_slots = self._compute_state_slot_mapping(positions, b_idx)
        with record_function_range("dsv4.fp8.compressor.meta.kv_slots"):
            kv_slots = self._compute_kv_slot_mapping(positions, b_idx)
        with record_function_range("dsv4.fp8.compressor.meta.token_to_req"):
            token_to_req = b_idx.to(torch.int32)
        return CompressorMeta(
            positions=positions,
            b_idx=b_idx,
            state_slots=state_slots,
            kv_slots=kv_slots,
            token_to_req=token_to_req,
            is_batched=is_batched,
            seq_start_per_req=seq_start_per_req,
            cu_seq_per_req=cu_seq_per_req,
        )

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------
    def _ensure_cos_sin_cache(self, device: torch.device) -> torch.Tensor:
        if self._cos_sin_cache is None or self._cos_sin_cache.device != device:
            assert (
                self.freqs_cis is not None
            ), "CompressorFP8.freqs_cis must be bound before forward"
            cache, _ = build_cos_sin_cache(self.freqs_cis.to(device))
            self._cos_sin_cache = cache
        return self._cos_sin_cache

    def _compute_state_slot_mapping(
        self,
        positions: torch.Tensor,  # [N] int64
        b_idx: torch.Tensor,  # [N] int64
    ) -> torch.Tensor:
        """state_slot[t] = state_block_table[b, (pos//eb) % max_blocks] * eb + pos%eb.
        Returns -1 where the chosen block id is unallocated (==0 sentinel)."""
        bt = self._state_block_table
        eb = self._state_eb
        assert bt is not None and eb > 0, "state pool context unbound"
        bt_long = bt.to(torch.long)
        max_blocks = int(bt_long.shape[1])
        block_in_seq = (positions // eb) % max_blocks
        in_block = positions % eb
        block_id = bt_long[b_idx, block_in_seq]
        valid = block_id > 0
        slot = block_id * eb + in_block
        return torch.where(valid, slot, torch.full_like(slot, -1))

    def _compute_kv_slot_mapping(
        self,
        positions: torch.Tensor,  # [N] int64
        b_idx: torch.Tensor,  # [N] int64
    ) -> torch.Tensor:
        """KV-pool slot for each token. -1 unless (pos+1) % ratio == 0
        (i.e. boundary token that produces a compressed entry).

        Block addressing follows the framework convention: ALL DSV4 pools
        use ``seq_size_per_block = TOKENS_PER_BLOCK = 256`` for block_table
        indexing — i.e. the block_table is indexed in *token* space, not
        compressed-entry space. The KV pool's per-block entry count is
        ``kv_eb = TOKENS_PER_BLOCK / ratio``, so the in-block offset is the
        compressed-entry offset *within that token block*.

          block_in_seq = pos // TOKENS_PER_BLOCK              # token -> block
          in_block     = (pos % TOKENS_PER_BLOCK) // ratio    # compressed offset
          slot         = block_id * kv_eb + in_block

        Also masks out any slot that would land past the pool's row count
        (same overflow guard upstream 2184f972 added to the legacy
        ``_compute_pool_slots`` — a malformed block_table can otherwise
        produce a slot above ``pool_view.shape[0]`` and silently corrupt
        an unrelated pool entry).
        """
        bt = self._kv_block_table
        kv_eb = self._kv_eb
        ratio = self.compress_ratio
        if bt is None or kv_eb <= 0:
            return torch.full_like(positions, -1)
        # Recover seq_size_per_block from the pool spec invariant
        # (kv_eb * ratio); avoids hard-coding 256 here.
        tokens_per_block = kv_eb * ratio
        bt_long = bt.to(torch.long)
        max_blocks = int(bt_long.shape[1])
        if max_blocks <= 0:
            return torch.full_like(positions, -1)

        boundary = ((positions + 1) % ratio) == 0
        block_in_seq = positions // tokens_per_block
        in_block = (positions % tokens_per_block) // ratio
        in_capacity = block_in_seq < max_blocks
        safe_block_in_seq = block_in_seq.clamp(min=0, max=max_blocks - 1)
        block_id = bt_long[b_idx, safe_block_in_seq]
        slot = block_id * kv_eb + in_block
        valid = boundary & in_capacity & (block_id > 0)
        if self._kv_pool_3d is not None:
            pool_rows = int(self._kv_pool_3d.numel() // self._kv_pool_3d.shape[-1])
            valid = valid & (slot < pool_rows)
        return torch.where(valid, slot, torch.full_like(slot, -1))

    def _launch(
        self,
        kv_flat: torch.Tensor,  # [N, coff*head_dim] fp32
        score_flat: torch.Tensor,  # [N, coff*head_dim] fp32
        meta: CompressorMeta,
        seq_start: Optional[int] = None,
    ) -> None:
        """Launch the two vLLM kernels (state write + boundary KV write).

        ``seq_start`` is the absolute position of ``kv_flat[0]`` for
        sequentially-laid-out batches (prefill: ``sp_int``). When provided
        the fused kernel reads any overlap-window position with
        ``flat_idx = pos - seq_start in [0, N)`` directly from
        ``kv_flat / score_flat`` instead of the cyclic state pool, which
        only retains the latest ~512 tokens per request and would have
        been overwritten within this same launch by ``run_save_partial_states``.

        Pass ``None`` to disable the raw path (decode: ``kv_flat`` is
        indexed by ``req_idx``, not by absolute position offset).

        All slot-mapping math is consumed from ``meta`` — this method only
        does kernel launches and the cos_sin_cache lazy build. Designed to
        stay branch-light so it composes cleanly with CUDA graph capture.
        """
        if (
            self._state_pool_3d is None
            or self._kv_pool_3d is None
            or self._state_block_table is None
            or self._kv_block_table is None
        ):
            # Warmup / unbound: nothing to write.
            return

        N = int(meta.positions.shape[0])
        if N == 0:
            return

        with record_function_range("dsv4.fp8.compressor.launch.cos_sin_cache"):
            cos_sin_cache = self._ensure_cos_sin_cache(kv_flat.device)

        with record_function_range("dsv4.fp8.compressor.launch.save_partial_states"):
            run_save_partial_states(
                kv_flat,
                score_flat,
                self.ape,
                meta.positions,
                self._state_pool_3d,
                meta.state_slots,
                compress_ratio=self.compress_ratio,
            )

        # Decode path passes seq_start=None: disable the raw branch so the
        # kernel only reads state_cache. seq_start value is then irrelevant.
        # Phase-3a part 4c: when meta carries per-request raw windows, the
        # kernel uses native varlen raw path (one address per request) and
        # the scalar ``seq_start`` is unused.
        use_varlen_raw = (
            meta.is_batched
            and meta.seq_start_per_req is not None
            and meta.cu_seq_per_req is not None
        )
        raw_disabled = (seq_start is None) and not use_varlen_raw
        with record_function_range("dsv4.fp8.compressor.launch.compress_kv_write"):
            run_fused_compress_kv_write(
                self._state_pool_3d,
                meta.token_to_req,
                meta.positions,
                meta.state_slots,
                self._state_block_table.to(torch.int32),
                self.norm.weight,
                self.norm_eps,
                cos_sin_cache,
                self._kv_pool_3d,
                meta.kv_slots,
                kv_flat,
                score_flat,
                self.ape,
                0 if (raw_disabled or use_varlen_raw) else seq_start,
                disable_raw_path=raw_disabled,
                head_dim=self.head_dim,
                rope_head_dim=self.rope_head_dim,
                compress_ratio=self.compress_ratio,
                overlap=self.overlap,
                seq_start_per_req=meta.seq_start_per_req if use_varlen_raw else None,
                cu_seq_per_req=meta.cu_seq_per_req if use_varlen_raw else None,
            )

    # ----------------------------------------------------------------------
    # Forward (prefill)
    # ----------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        start_pos,
        sequence_lengths: Optional[torch.Tensor] = None,
        meta: Optional[CompressorMeta] = None,
    ) -> Optional[torch.Tensor]:
        """Prefill entry. ``bsz==1`` (FIFO scheduler).

        Returns ``None`` — downstream readers gather compressed K from the
        FP8 KV pool directly.

        ``meta`` lets the caller (typically ``attention.py``) hoist the
        slot-mapping compute out of the per-layer hot path and amortize it
        across the host compressor and any nested indexer compressor that
        share the same positions/b_idx. When ``None`` the compressor falls
        back to the in-body compute path (warmup / standalone / UT).
        """
        del sequence_lengths  # not needed: positions derived from start_pos+arange
        # Phase-3a: accept either flat ``[T_total, dim]`` (vLLM-native /
        # batched prefill) or legacy ``[B, S, dim]``. CP gather still
        # produces a 3D tensor downstream so we leave the post-gather
        # (bsz, seqlen) recompute alone — the dim-2 fast path only fires
        # when the *input* arrives flat.
        if x.dim() == 2:
            bsz = 1
            seqlen = int(x.size(0))
        else:
            bsz, seqlen, _ = x.size()
        sp = (
            int(start_pos.item())
            if isinstance(start_pos, torch.Tensor)
            else int(start_pos)
        )
        # Warmup forward (no pool bound by framework): no-op.
        if self._state_pool_3d is None or self._kv_pool_3d is None or self._kv_eb <= 0:
            return None

        device = x.device
        out_dim = (1 + self.overlap) * self.head_dim
        with record_function_range("dsv4.fp8.compressor.prefill.fused_linear"):
            fused_out = torch.nn.functional.linear(
                x.to(self._wkv_wgate_fused.dtype), self._wkv_wgate_fused
            )
            kv, score = fused_out[..., :out_dim], fused_out[..., out_dim:]

        cp_ctx = self._cp_ctx
        cp_gather = cp_should_gather(cp_ctx, start_pos)
        kv_gather_handle = None
        score_gather_handle = None
        if cp_gather:
            assert cp_ctx is not None
            # Current CP gather API is flattened token-major 2D. Production
            # prefill flattens requests to [T_total, dim]; accept the legacy
            # [1, T_total, dim] wrapper by squeezing it at the module boundary.
            if kv.dim() == 3:
                assert kv.size(0) == 1, "CompressorFP8 CP expects flattened input"
                kv_local = kv.squeeze(0)
                score_local = score.squeeze(0)
            else:
                kv_local = kv
                score_local = score
            gather_stream = (
                torch.cuda.Stream(device=kv_local.device) if kv_local.is_cuda else None
            )
            # Start both collectives early so their NCCL work can overlap with
            # downstream CPU/Python setup before the gathered tensors are needed.
            with record_function_range("dsv4.fp8.compressor.prefill.cp_gather_kv"):
                kv_gather_handle = cp_all_gather_full_async(
                    kv_local, cp_ctx, stream=gather_stream
                )
            with record_function_range("dsv4.fp8.compressor.prefill.cp_gather_score"):
                score_gather_handle = cp_all_gather_full_async(
                    score_local, cp_ctx, stream=gather_stream
                )
            bsz = 1
            seqlen = int(cp_ctx.seq_len_full)
            # The post-gather kv/score tensors are sized for the GLOBAL
            # ``seq_len_full``. The matching slot metadata is layer-invariant
            # and must be built in the prefill-meta broadcast path before this
            # hot path runs.
            # ``bsz`` here is the leading dim of the compressor input, NOT
            # the number of prefill requests — production prefill always
            # flattens multi-request input to ``[1, T_total, dim]`` (with
            # T_total being the sum of per-request lengths). Multi-request CP
            # keeps this convention and makes
            # ``seqlen == seq_len_full == sum_i L_i_global``.
            assert meta is not None, (
                "CompressorFP8 CP prefill requires full-sequence metadata from "
                "rtp_llm.models_py.modules.dsv4.fp8.prefill_meta; rebuilding it "
                "inside compressor.forward is intentionally disabled."
            )
            assert int(meta.positions.numel()) == seqlen, (
                f"CP compressor meta/token length mismatch: meta={meta.positions.numel()} "
                f"seq_len_full={seqlen}"
            )
            assert kv_gather_handle is not None
            assert score_gather_handle is not None
            with record_function_range("dsv4.fp8.compressor.prefill.cp_wait_kv"):
                kv = cp_wait_gather_full(kv_gather_handle).unsqueeze(0)
            with record_function_range("dsv4.fp8.compressor.prefill.cp_wait_score"):
                score = cp_wait_gather_full(score_gather_handle).unsqueeze(0)

        N = bsz * seqlen
        # ``reshape(N, -1)`` collapses dim-0/1 for 3D, no-op for 2D — same
        # contiguous flat layout the kernel expects in both cases.
        with record_function_range("dsv4.fp8.compressor.prefill.flatten"):
            kv_flat = kv.reshape(N, -1).contiguous()
            score_flat = score.reshape(N, -1).contiguous()
        if meta is None:
            with record_function_range("dsv4.fp8.compressor.prefill.build_meta"):
                positions, b_idx = _build_prefill_positions(sp, bsz, seqlen, device)
                meta = self.prepare_metadata(positions, b_idx)
        # Phase-3a part 4b/4c: B>1 batched prefill — each request has its
        # own absolute start_pos so the scalar ``seq_start`` raw path is
        # unsafe. When the meta carries per-request varlen raw arrays
        # (4c), ``_launch`` routes through the kernel's BATCHED branch.
        # Otherwise (4b fallback) ``seq_start=None`` disables the raw
        # path and reads from state_cache instead. Both validated byte-
        # equal vs raw_path=True for short prefill by
        # ``test_compressor_disable_raw_path``.
        seq_start = None if meta.is_batched else sp
        with record_function_range("dsv4.fp8.compressor.prefill.launch"):
            self._launch(kv_flat, score_flat, meta, seq_start=seq_start)
        return None

    # ----------------------------------------------------------------------
    # Forward (decode, vectorized over B)
    # ----------------------------------------------------------------------
    def forward_decode_vectorized(
        self,
        x: torch.Tensor,
        start_pos: torch.Tensor,
        meta: Optional[CompressorMeta] = None,
    ) -> Optional[torch.Tensor]:
        """Batched decode entry. q_len == 1 per request."""
        assert x.shape[1] == 1, "decode-only: q_len must be 1"
        bsz = x.size(0)
        if self._state_pool_3d is None or self._kv_pool_3d is None or self._kv_eb <= 0:
            return None

        device = x.device
        out_dim = (1 + self.overlap) * self.head_dim
        fused_out = torch.nn.functional.linear(
            x.to(self._wkv_wgate_fused.dtype), self._wkv_wgate_fused
        )
        kv, score = fused_out[..., :out_dim], fused_out[..., out_dim:]

        kv_flat = kv.reshape(bsz, -1).contiguous()
        score_flat = score.reshape(bsz, -1).contiguous()
        if meta is None:
            positions = start_pos.to(device=device, dtype=torch.long).reshape(bsz)
            b_idx = torch.arange(bsz, device=device, dtype=torch.long)
            meta = self.prepare_metadata(positions, b_idx)
        self._launch(kv_flat, score_flat, meta)
        return None


# ---------------------------------------------------------------------------
# Free helpers — exposed so the attention layer can build positions/b_idx
# once per forward and feed them through ``prepare_metadata`` / ``forward``.
# ---------------------------------------------------------------------------
def _build_prefill_positions(
    sp: int, bsz: int, seqlen: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """``positions = sp + arange(seqlen)`` flat ``[seqlen]``; ``b_idx``
    all-zeros ``[seqlen]``.

    Legacy B==1 / ``DSV4_VARLEN_PREFILL=0`` helper only — varlen feeds
    ``(position_ids, req_id_per_token)`` straight into
    ``compressor.prepare_metadata`` so this builder must NEVER be reached
    from a varlen call site (positions would collapse to a single
    contiguous ``[sp, sp+T_total)`` range and ``b_idx`` to all-zeros,
    silently mapping every token onto request-0's block_table).
    """
    assert bsz == 1, (
        f"_build_prefill_positions is the legacy B==1 helper; got bsz={bsz}. "
        "Varlen callers must use position_ids / req_id_per_token directly."
    )
    positions = torch.arange(sp, sp + seqlen, device=device, dtype=torch.long)
    b_idx = torch.zeros(seqlen, device=device, dtype=torch.long)
    return positions, b_idx


def build_prefill_metadata(
    compressor: "CompressorFP8", sp: int, bsz: int, seqlen: int, device: torch.device
) -> CompressorMeta:
    """Convenience: build positions/b_idx + ``CompressorMeta`` in one call."""
    positions, b_idx = _build_prefill_positions(sp, bsz, seqlen, device)
    return compressor.prepare_metadata(positions, b_idx)


def build_prepare_metadata_args(
    *,
    use_varlen: bool,
    device: torch.device,
    sp_int: int,
    seqlen: int,
    position_ids: Optional[torch.Tensor] = None,
    req_id_per_token: Optional[torch.Tensor] = None,
    seq_start_per_req: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Return the kwargs dict for ``CompressorFP8.prepare_metadata``,
    branching on ``use_varlen``. Single source of truth for the three
    call sites that used to inline this dispatch:

      * ``Attention._build_compressor_meta`` (HCA + standalone CSA path)
      * ``Attention._build_csa_prefill_meta`` (CSA inline that shares the
        pool bind with ``IndexerFP8.prepare``)
      * ``IndexerFP8.prepare`` nested compressor hoist

    Under varlen we pass the upper-layer-derived per-request tensors
    straight through; the legacy B==1 path collapses to the same
    ``(arange(sp, sp+T), zeros)`` pair ``_build_prefill_positions`` produces
    so bisecting between the two stays bit-equal.

    ``seq_start_per_req`` accepts either ``sp_per_req`` (Attention) or
    ``prefix_lengths`` (Indexer) — both index32-cast to the same int32
    layout the compressor's varlen raw kernel consumes.
    """
    if use_varlen:
        assert (
            position_ids is not None
            and req_id_per_token is not None
            and seq_start_per_req is not None
            and cu_seqlens is not None
        ), "varlen dispatch requires position_ids/req_id_per_token/seq_start_per_req/cu_seqlens"
        return dict(
            positions=position_ids.to(device=device, dtype=torch.long)
            .reshape(-1)
            .contiguous(),
            b_idx=req_id_per_token.to(device=device, dtype=torch.long)
            .reshape(-1)
            .contiguous(),
            is_batched=True,
            seq_start_per_req=seq_start_per_req.to(device=device, dtype=torch.int32)
            .reshape(-1)
            .contiguous(),
            cu_seq_per_req=cu_seqlens.to(device=device, dtype=torch.int32)
            .reshape(-1)
            .contiguous(),
        )
    positions, b_idx = _build_prefill_positions(sp_int, 1, seqlen, device)
    return dict(
        positions=positions,
        b_idx=b_idx,
        is_batched=False,
        seq_start_per_req=None,
        cu_seq_per_req=None,
    )


def build_decode_metadata(
    compressor: "CompressorFP8", start_pos: torch.Tensor, bsz: int
) -> CompressorMeta:
    device = start_pos.device
    positions = start_pos.to(device=device, dtype=torch.long).reshape(bsz)
    b_idx = torch.arange(bsz, device=device, dtype=torch.long)
    return compressor.prepare_metadata(positions, b_idx)
