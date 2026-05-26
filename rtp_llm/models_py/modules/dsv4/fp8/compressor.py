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
``entries_per_block=256``: every token gets its own slot. We mirror
vLLM's ``DeepseekCompressor`` flow:

  1. ``_save_partial_states_kernel`` writes per-token (kv | score+ape)
     into the framework-allocated state pool.
  2. ``_fused_kv_compress_norm_rope_insert_*_attn`` self-skips
     non-boundary tokens, otherwise gathers the ``(1+overlap)*ratio``
     window from the state pool, does softmax → RMSNorm → RoPE → FP8
     UE8M0 quant → KV-pool slot store.

Public API kept stable so ``attention.py`` / ``indexer_fp8.py`` call
sites do not change:
  * ``set_pool_context(kv_view, kv_bt, kv_eb, state_view, state_bt,
    state_eb)`` — shared with ``PoolBackedModule``.
  * ``forward(x, start_pos, sequence_lengths=None)`` for prefill.
  * ``forward_decode_vectorized(x, start_pos)`` for batched decode.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.ops.compute_ops import rtp_llm_ops

_CUBLAS_GEMM_BF16_BF16_FP32 = getattr(rtp_llm_ops, "cublas_gemm_bf16_bf16_fp32", None)


def _compressor_meta_fused_enabled() -> bool:
    """Env-gated fused compressor prepare_metadata.

    Default-on: decode now feeds metadata from buildmeta, and the remaining
    prepare_metadata callers should use the fused integer slot builder unless
    explicitly disabled for debugging with ``DSV4_FUSED_PREPARE=0``.
    """

    return os.environ.get("DSV4_FUSED_PREPARE", "1").strip() == "1"


def _linear_bf16_bf16_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """F.linear(x, weight) with BF16 operands and FP32 accumulation/output."""
    assert x.dtype == torch.bfloat16, f"expected BF16 input, got {x.dtype}"
    assert weight.dtype == torch.bfloat16, f"expected BF16 weight, got {weight.dtype}"
    assert x.is_contiguous(), "expected contiguous input"
    assert weight.is_contiguous(), "expected contiguous weight"
    assert (
        _CUBLAS_GEMM_BF16_BF16_FP32 is not None
    ), "cublas_gemm_bf16_bf16_fp32 op is not built"
    leading_shape = x.shape[:-1]
    x_2d = x.reshape(-1, x.shape[-1])
    out_2d = _CUBLAS_GEMM_BF16_BF16_FP32(x_2d, weight)
    return out_2d.reshape(*leading_shape, weight.shape[0])


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
from rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils import PoolBackedModule

# Process-local cache for the device-side cos_sin tensor derived from a
# given freqs_cis source. DSV4 has ~91 CompressorFP8 instances (main +
# indexer + indexer.compressor per layer), each holding its own 256 MiB
# cos_sin_cache at 1M seq len → 22.75 GiB of duplicated baseline memory.
# Since reset_rope_cache now binds the memoized shared freqs_cis tensor
# (see rope.py), all instances see the same ``id(freqs_cis)`` → one
# shared entry instead of 91. Saves ~22.5 GiB of persistent GPU memory
# per rank during 1M prefill.
_SHARED_COS_SIN_CACHE: Dict[Tuple[int, torch.device], torch.Tensor] = {}


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
      * ``is_batched``   : True when this meta uses the varlen/per-request
                           raw path. This includes B==1 CP prefill, where
                           keeping the same path avoids reintroducing a
                           scalar-B special case.
    """

    positions: torch.Tensor
    b_idx: torch.Tensor
    state_slots: torch.Tensor
    kv_slots: torch.Tensor
    token_to_req: torch.Tensor
    is_batched: bool = False
    # Phase-3a part 4c — varlen raw path. Populated when is_batched;
    # otherwise the legacy scalar-seq_start path is used.
    #   seq_start_per_req[b] = abs position of req b's first new token (sp_b)
    #   cu_seq_per_req[b+1]  = end offset of req b in flat kv_flat axis
    seq_start_per_req: Optional[torch.Tensor] = None
    cu_seq_per_req: Optional[torch.Tensor] = None
    # Decode indexer hot path: per-token compressed length
    # ``floor((position + 1) / ratio)``. Built once in decode metadata so
    # CSA indexer layers do not relaunch tiny add/div kernels.
    compressed_lens_per_token: Optional[torch.Tensor] = None


class _CompressorNorm(nn.Module):
    """RMSNorm weight holder — bf16 (vLLM kernel reads bf16 weight)."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))


class CompressorFP8(PoolBackedModule):
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

        # ape stays FP32. wkv/wgate are stored as BF16, while the fused
        # projection below uses BF16 operands with FP32 accumulation/output.
        # Register ape as a non-trainable Parameter so .to(device) follows the
        # module.
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

        # Legacy attribute kept for attention.py's cmp_T fallback (line 1583).
        self._kv_cache_t: int = 0

        # Cached cos_sin cache built from self.freqs_cis at first forward.
        # ``_cos_sin_cache_device`` is the cached device sibling so the hot
        # path avoids ``tensor.device`` property construction (~70 ns/call).
        self.freqs_cis: Optional[torch.Tensor] = None
        self._cos_sin_cache: Optional[torch.Tensor] = None
        self._cos_sin_cache_device: Optional[torch.device] = None
        self._cp_ctx: Optional[CPContext] = None
        self._cp_gather_stream: Optional[Any] = None
        self._kv_cache_sharded: bool = False
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
        # When ``kv_cache_sharded=True`` flows in via the CPContext, only the
        # paged KV-pool writer uses CP-aware slot mapping. STATE pools are not
        # sharded and must keep the full slot mapping on every rank.
        self._kv_cache_sharded = bool(
            cp_ctx is not None
            and getattr(cp_ctx, "kv_cache_sharded", False)
            and cp_ctx.cp_size > 1
        )

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

        # Warmup: pool context unbound — return None-slotted meta so
        # callers (compressor.forward, _forward_prefill_compressed) can
        # short-circuit without crashing.
        if self._state_block_table is None or self._state_eb <= 0:
            return CompressorMeta(
                positions=positions,
                b_idx=b_idx,
                state_slots=None,
                kv_slots=None,
                token_to_req=b_idx.to(torch.int32),
                is_batched=is_batched,
                seq_start_per_req=seq_start_per_req,
                cu_seq_per_req=cu_seq_per_req,
            )

        if _compressor_meta_fused_enabled() and not self._kv_cache_sharded:
            from rtp_llm.models_py.modules.dsv4.fp8 import _fused_compressor_meta_triton

            if _fused_compressor_meta_triton._TRITON_AVAILABLE:
                pool_rows = 0
                if self._kv_pool_3d is not None:
                    pool_rows = int(
                        self._kv_pool_3d.numel() // self._kv_pool_3d.shape[-1]
                    )
                (
                    state_slots,
                    kv_slots,
                    token_to_req,
                ) = _fused_compressor_meta_triton.fused_compressor_slot_mapping(
                    positions,
                    b_idx,
                    self._state_block_table,
                    self._state_eb,
                    self._kv_block_table,
                    self._kv_eb,
                    self.compress_ratio,
                    pool_rows=pool_rows,
                )
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
        cached = self._cos_sin_cache
        # Compare against cached device sibling (a plain torch.device): hot
        # path avoids the ``cached.device`` property which constructs a fresh
        # torch.device per access (~70 ns/call on this code path).
        if cached is not None and self._cos_sin_cache_device == device:
            return cached
        assert (
            self.freqs_cis is not None
        ), "CompressorFP8.freqs_cis must be bound before forward"
        # Dedup at module level by source freqs_cis identity. After the
        # rope.py memoization, all CompressorFP8 instances binding the same
        # rope params share one freqs_cis object → one cos_sin_cache.
        key = (id(self.freqs_cis), device)
        shared = _SHARED_COS_SIN_CACHE.get(key)
        if shared is None or shared.device != device:
            shared, _ = build_cos_sin_cache(self.freqs_cis.to(device))
            _SHARED_COS_SIN_CACHE[key] = shared
        self._cos_sin_cache = shared
        self._cos_sin_cache_device = device
        return shared

    def _compute_state_slot_mapping(
        self,
        positions: torch.Tensor,  # [N] int64
        b_idx: torch.Tensor,  # [N] int64
    ) -> torch.Tensor:
        """state_slot[t] = state_block_table[b, pos//eb] * eb + pos%eb.
        Returns -1 where the logical block is absent or unallocated."""
        bt = self._state_block_table
        eb = self._state_eb
        assert bt is not None and eb > 0, "state pool context unbound"
        # STATE pools (CSA_STATE / HCA_STATE) are NOT cp-sharded — each rank
        # holds the full block_table. Applying cp ownership mask here would
        # leave half the entries as -1 sentinels, the writer would skip them,
        # and decode would read garbage at non-owned positions. Always use
        # the full slot mapping.
        bt_long = bt.to(torch.long)
        max_blocks = int(bt_long.shape[1])
        if max_blocks <= 0:
            return torch.full_like(positions, -1)
        block_in_seq = positions // eb
        in_block = positions % eb
        in_capacity = block_in_seq < max_blocks
        safe_block_in_seq = block_in_seq.clamp(min=0, max=max_blocks - 1)
        block_id = bt_long[b_idx, safe_block_in_seq]
        valid = in_capacity & (block_id >= 0)
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

        Also masks out any slot that would land past the pool's row count; a
        malformed block_table can otherwise produce a slot above
        ``pool_view.shape[0]`` and silently corrupt an unrelated pool entry.
        """
        bt = self._kv_block_table
        kv_eb = self._kv_eb
        ratio = self.compress_ratio
        if bt is None or kv_eb <= 0:
            return torch.full_like(positions, -1)
        # Recover seq_size_per_block from the pool spec invariant
        # (kv_eb * ratio); avoids hard-coding 256 here.
        tokens_per_block = kv_eb * ratio
        if self._kv_cache_sharded and self._cp_ctx is not None:
            from rtp_llm.models_py.modules.dsv4.fp8._cp_slot_mapping import (
                cp_kv_slot_mapping,
            )

            slot = cp_kv_slot_mapping(
                positions,
                bt,
                b_idx,
                tokens_per_block,
                kv_eb,
                ratio,
                self._cp_ctx.cp_size,
                self._cp_ctx.cp_rank,
            )
            if self._kv_pool_3d is not None:
                pool_rows = int(self._kv_pool_3d.numel() // self._kv_pool_3d.shape[-1])
                slot = torch.where(slot < pool_rows, slot, torch.full_like(slot, -1))
            return slot
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
        valid = boundary & in_capacity & (block_id >= 0)
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
        ``kv_flat / score_flat`` instead of reading back through the state
        pool, where current-launch writes are still in flight.

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
                self._state_block_table,
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
        # batched prefill) or legacy ``[B, S, dim]``. The compressor kernels
        # consume token-major 2D tensors, so the prefill path flattens once
        # immediately after the fused projection and never reintroduces B.
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
            fused_out = _linear_bf16_bf16_fp32(x, self._wkv_wgate_fused)
            N = bsz * seqlen
            fused_flat = fused_out.reshape(N, -1)

        cp_ctx = self._cp_ctx
        cp_gather = cp_should_gather(cp_ctx, start_pos)
        fused_gather_handle = None
        if cp_gather:
            assert cp_ctx is not None
            gather_stream = (
                torch.cuda.Stream(device=fused_flat.device)
                if fused_flat.is_cuda
                else None
            )
            # Gather the fused ``[kv | score]`` projection once. This removes
            # one NCCL launch and keeps the CP path token-major 2D; the split
            # below is a strided view and the Triton wrappers pass row strides
            # explicitly.
            with record_function_range(
                "dsv4.fp8.compressor.prefill.cp_gather_kv_score"
            ):
                fused_gather_handle = cp_all_gather_full_async(
                    fused_flat, cp_ctx, stream=gather_stream
                )
            assert meta is not None, (
                "CompressorFP8 CP prefill requires full-sequence metadata from "
                "rtp_llm.models_py.modules.dsv4.fp8.prefill_meta; rebuilding it "
                "inside compressor.forward is intentionally disabled."
            )
            N = int(cp_ctx.seq_len_full)
            assert int(meta.positions.numel()) == N, (
                f"CP compressor meta/token length mismatch: meta={meta.positions.numel()} "
                f"seq_len_full={N}"
            )
            assert fused_gather_handle is not None
            with record_function_range("dsv4.fp8.compressor.prefill.cp_wait_kv_score"):
                fused_flat = cp_wait_gather_full(fused_gather_handle)

        with record_function_range("dsv4.fp8.compressor.prefill.split_kv_score"):
            assert fused_flat.dim() == 2, (
                f"CompressorFP8 prefill expects flat fused projection, got "
                f"{tuple(fused_flat.shape)}"
            )
            assert fused_flat.size(1) == 2 * out_dim, (
                f"CompressorFP8 fused hidden mismatch: got {fused_flat.size(1)}, "
                f"expected {2 * out_dim}"
            )
            kv_flat = fused_flat[:, :out_dim]
            score_flat = fused_flat[:, out_dim:]
        if meta is None:
            with record_function_range("dsv4.fp8.compressor.prefill.build_meta"):
                positions, b_idx = _build_prefill_positions(sp, bsz, seqlen, device)
                meta = self.prepare_metadata(positions, b_idx)
        # Varlen prefill carries per-request raw arrays, so ``_launch`` routes
        # through the kernel's BATCHED branch even when CP has only one
        # request. Legacy scalar metadata keeps the old ``seq_start`` path.
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
        position_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Batched decode entry. ``position_ids`` enables q_len > 1 verify."""
        bsz, q_len = int(x.size(0)), int(x.size(1))
        T = bsz * q_len
        if position_ids is None:
            assert q_len == 1, "decode q_len > 1 requires flat position_ids"
        if self._state_pool_3d is None or self._kv_pool_3d is None or self._kv_eb <= 0:
            return None

        device = x.device
        out_dim = (1 + self.overlap) * self.head_dim
        fused_out = _linear_bf16_bf16_fp32(x, self._wkv_wgate_fused)
        kv, score = fused_out[..., :out_dim], fused_out[..., out_dim:]

        # ``kv`` and ``score`` are last-dim slices of ``fused_out``. Keep
        # them as strided row views instead of materialising two BF16 copy
        # kernels; the Triton writer consumes row stride explicitly.
        kv_flat = kv.view(T, -1)
        score_flat = score.view(T, -1)
        if meta is None:
            if position_ids is None:
                positions = start_pos.to(device=device, dtype=torch.long).reshape(bsz)
                b_idx = torch.arange(bsz, device=device, dtype=torch.long)
                meta = self.prepare_metadata(positions, b_idx)
            else:
                positions = (
                    position_ids.to(device=device, dtype=torch.long)
                    .reshape(T)
                    .contiguous()
                )
                b_idx = torch.arange(bsz, device=device, dtype=torch.long)
                b_idx = b_idx.repeat_interleave(q_len).contiguous()
                position_ids_2d = positions.view(bsz, q_len)
                cu_seq_per_req = torch.arange(
                    0,
                    (bsz + 1) * q_len,
                    q_len,
                    device=device,
                    dtype=torch.int32,
                )
                meta = self.prepare_metadata(
                    positions,
                    b_idx,
                    is_batched=q_len > 1,
                    seq_start_per_req=position_ids_2d[:, 0]
                    .to(torch.int32)
                    .contiguous(),
                    cu_seq_per_req=cu_seq_per_req,
                )
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
