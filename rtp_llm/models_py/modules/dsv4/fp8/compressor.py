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

Public API:
  * ``set_pool_context(kv_view, kv_bt, kv_eb, state_view, state_bt,
    state_eb, state_tokens_per_block, kv_tokens_per_block)`` — shared with
    ``PoolBackedModule``.
  * ``forward(x, start_pos, sequence_lengths=None)`` for prefill.
  * ``forward_decode_vectorized(x, start_pos, meta)`` for batched decode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.ops.compute_ops import rtp_llm_ops

_CUBLAS_GEMM_BF16_BF16_FP32 = getattr(rtp_llm_ops, "cublas_gemm_bf16_bf16_fp32", None)


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
_SHARED_COS_SIN_CACHE: Dict[
    Tuple[int, torch.device, Tuple[int, ...], torch.dtype], torch.Tensor
] = {}


def _build_cp_full_state_read_cache(
    state_cache: torch.Tensor,
    block_table: torch.Tensor,
    cp_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather CP-sliced fixed-state blocks into a compact full-ring read cache."""
    if cp_size <= 1:
        return state_cache, block_table
    if block_table is None or int(block_table.numel()) == 0:
        return state_cache, block_table
    if not state_cache.is_cuda:
        raise RuntimeError("CP-sliced DSV4 state-cache read requires CUDA state pools")

    bt = block_table.to(device=state_cache.device, dtype=torch.long).contiguous()
    rows = int(bt.numel())
    if rows == 0:
        return state_cache, block_table

    local_eb = int(state_cache.shape[1])
    hidden = int(state_cache.shape[2])
    flat_ids = bt.reshape(-1)
    valid = flat_ids > 0
    safe_ids = torch.where(valid, flat_ids, torch.zeros_like(flat_ids))

    local_blocks = state_cache.index_select(0, safe_ids)
    local_blocks = torch.where(
        valid.view(rows, 1, 1),
        local_blocks,
        torch.zeros((), dtype=state_cache.dtype, device=state_cache.device),
    )

    gathered = all_gather(
        local_blocks.reshape(rows * local_eb, hidden).contiguous(),
        group=Group.TP,
    )
    full = (
        gathered.view(cp_size, rows, local_eb, hidden)
        .permute(1, 0, 2, 3)
        .reshape(rows, cp_size * local_eb, hidden)
        .contiguous()
    )

    zero = torch.zeros(
        (1, cp_size * local_eb, hidden),
        dtype=state_cache.dtype,
        device=state_cache.device,
    )
    read_cache = torch.cat([zero, full], dim=0)

    compact = torch.arange(1, rows + 1, device=bt.device, dtype=bt.dtype).view_as(bt)
    read_bt = torch.where(bt > 0, compact, torch.zeros_like(compact))
    return read_cache, read_bt.to(dtype=block_table.dtype)


def _cp_sliced_state_read_needed(
    meta: "CompressorMeta", cp_ctx: Optional[CPContext]
) -> bool:
    if cp_ctx is None or cp_ctx.cp_size <= 1:
        return False
    if not getattr(cp_ctx, "kv_cache_sharded", False):
        return False
    if meta.seq_start_per_req is None:
        return False
    return bool(torch.any(meta.seq_start_per_req > 0).item())


def _validate_per_request_metadata(
    positions: torch.Tensor,
    b_idx: torch.Tensor,
    seq_start_per_req: Optional[torch.Tensor],
    cu_seq_per_req: Optional[torch.Tensor],
) -> None:
    if seq_start_per_req is None or cu_seq_per_req is None:
        raise RuntimeError(
            "CompressorFP8 requires per-request metadata: "
            "seq_start_per_req/cu_seq_per_req"
        )
    if seq_start_per_req.dim() != 1:
        raise RuntimeError(
            f"seq_start_per_req must be 1D, got {tuple(seq_start_per_req.shape)}"
        )
    if cu_seq_per_req.dim() != 1:
        raise RuntimeError(
            f"cu_seq_per_req must be 1D, got {tuple(cu_seq_per_req.shape)}"
        )
    req_count = int(seq_start_per_req.numel())
    if int(cu_seq_per_req.numel()) != req_count + 1:
        raise RuntimeError(
            "cu_seq_per_req length must equal seq_start_per_req length + 1, "
            f"got {cu_seq_per_req.numel()} vs {req_count}"
        )
    if positions.numel() == 0:
        return
    if req_count == 0:
        raise RuntimeError("seq_start_per_req must not be empty when positions exist")
    min_req = int(b_idx.min().item())
    max_req = int(b_idx.max().item())
    if min_req < 0 or max_req >= req_count:
        raise RuntimeError(
            f"b_idx out of range for per-request metadata: "
            f"min={min_req} max={max_req} req_count={req_count}"
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
    """

    positions: torch.Tensor
    b_idx: torch.Tensor
    state_slots: torch.Tensor
    kv_slots: torch.Tensor
    token_to_req: torch.Tensor
    # Varlen raw path metadata. Required by the compressor writer once pools
    # are bound; B==1 uses the same per-request arrays as larger batches.
    #   seq_start_per_req[b] = abs position of req b's first new token (sp_b)
    #   cu_seq_per_req[b+1]  = end offset of req b in flat kv_flat axis
    seq_start_per_req: Optional[torch.Tensor] = None
    cu_seq_per_req: Optional[torch.Tensor] = None
    # Decode indexer hot path: per-token compressed length
    # ``floor((position + 1) / ratio)``. Built once in decode metadata so
    # CSA indexer layers do not relaunch tiny add/div kernels.
    compressed_lens_per_token: Optional[torch.Tensor] = None


@dataclass
class _CompressorPending:
    """In-flight state between :meth:`CompressorFP8.start_prefill` and
    :meth:`CompressorFP8.finish_prefill`.

    Captured at ``start_prefill`` time so the orchestrator (attention
    overlap path) can interleave other work on the default stream while
    the CP all-gather drains on ``cp_gather_stream``. Holds:

    * ``fused_flat`` — rank-local ``[T_local, 2*out_dim]`` fused KV/gate
      projection. When CP is on, this is the source of the NCCL gather
      (``fused_gather_handle``) and ``finish_prefill`` overwrites the
      local reference with the gathered full-seq tensor.
    * ``fused_gather_handle`` — ``CPCudaAsyncGatherHandle`` (or sync handle)
      when CP is active; ``None`` when CP is off (single-rank prefill).
    * ``meta`` — caller-provided ``CompressorMeta``. The attention layer
      builds it before the compressor write so slot mapping is not rebuilt
      inside the hot path.
    * ``out_dim`` — ``(1 + overlap) * head_dim``. Captured at start_prefill
      time so finish_prefill stays pure (no self peek).
    * ``restored_buf`` — optional full-sequence destination used by the
      non-prefix CP restore path. It is kept on the pending object so the
      restored fused tensor's storage stays owned through ``finish_prefill``.

    The overlap orchestrator may call ``wait_prefill_gather`` before
    ``finish_prefill`` to fence NCCL without making the compressed-pool write
    visible early. The baseline ``forward`` path is unchanged.
    """

    fused_flat: torch.Tensor
    fused_gather_handle: Optional[Any]
    meta: CompressorMeta
    out_dim: int
    profile_label: Optional[str] = None
    restored_buf: Optional[torch.Tensor] = None


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
        self._cos_sin_cache_key: Optional[
            Tuple[int, torch.device, Tuple[int, ...], torch.dtype]
        ] = None
        self._state_tokens_per_block: int = 0
        self._cp_ctx: Optional[CPContext] = None
        self._cp_gather_stream: Optional[Any] = None
        self._kv_cache_sharded: bool = False
        self._profile_label: Optional[str] = None
        # MOEDBG: caller (Attention / IndexerFP8) sets this to a name
        # prefix like ``"L02_attn_cmp"`` before forward and clears after;
        # _forward_prefill_body uses it as the rec name root. None / empty
        # string suppresses recording.
        self._dbg_prefix: Optional[str] = None

    def _cp_profile_name(self, profile_label: Optional[str]) -> str:
        label = profile_label or self._profile_label
        if not label:
            role = "indexer" if self.head_dim == INDEXER_HEAD_DIM else "attn"
            label = f"{role}.ratio{self.compress_ratio}.hd{self.head_dim}"
        return f"dsv4.cp.all_gather.{label}.kv_score"

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
        # When ``kv_cache_sharded=True`` flows in via the CPContext, paged KV
        # pools use page-RR ownership and fixed STATE pools use intra-block
        # slices. Both writers therefore need CP-aware slot mappings.
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
        _validate_per_request_metadata(
            positions, b_idx, seq_start_per_req, cu_seq_per_req
        )

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
                seq_start_per_req=seq_start_per_req,
                cu_seq_per_req=cu_seq_per_req,
            )

        input_lens = (cu_seq_per_req[1:] - cu_seq_per_req[:-1]).to(torch.long)
        seq_end_per_req = seq_start_per_req.to(torch.long) + input_lens

        if not self._kv_cache_sharded:
            from rtp_llm.models_py.modules.dsv4.fp8 import _fused_compressor_meta_triton

            if not _fused_compressor_meta_triton._TRITON_AVAILABLE:
                raise RuntimeError(
                    "DSV4 FP8 compressor requires fused Triton metadata preparation"
                )
            pool_rows = 0
            if self._kv_pool_view is not None:
                pool_rows = int(
                    self._kv_pool_view.numel() // self._kv_pool_view.shape[-1]
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
                seq_end_per_req,
                self._state_tokens_per_block,
                pool_rows=pool_rows,
            )
            return CompressorMeta(
                positions=positions,
                b_idx=b_idx,
                state_slots=state_slots,
                kv_slots=kv_slots,
                token_to_req=token_to_req,
                seq_start_per_req=seq_start_per_req,
                cu_seq_per_req=cu_seq_per_req,
            )

        with record_function_range("dsv4.fp8.compressor.meta.state_slots"):
            state_slots = self._compute_state_slot_mapping(
                positions, b_idx, seq_end_per_req
            )
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
            seq_start_per_req=seq_start_per_req,
            cu_seq_per_req=cu_seq_per_req,
        )

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------
    def _ensure_cos_sin_cache(self, device: torch.device) -> torch.Tensor:
        assert (
            self.freqs_cis is not None
        ), "CompressorFP8.freqs_cis must be bound before forward"
        key = (
            id(self.freqs_cis),
            device,
            tuple(int(v) for v in self.freqs_cis.shape),
            self.freqs_cis.dtype,
        )
        cached = self._cos_sin_cache
        # Compare against a key that includes source freqs_cis identity. A
        # reset may bind a new shared freqs tensor on the same device.
        if cached is not None and self._cos_sin_cache_key == key:
            return cached
        # Dedup at module level by source freqs_cis identity. After the
        # rope.py memoization, all CompressorFP8 instances binding the same
        # rope params share one freqs_cis object → one cos_sin_cache.
        shared = _SHARED_COS_SIN_CACHE.get(key)
        if shared is None or shared.device != device:
            shared, _ = build_cos_sin_cache(self.freqs_cis.to(device))
            _SHARED_COS_SIN_CACHE[key] = shared
        self._cos_sin_cache = shared
        self._cos_sin_cache_device = device
        self._cos_sin_cache_key = key
        return shared

    def _compute_state_slot_mapping(
        self,
        positions: torch.Tensor,  # [N] int64
        b_idx: torch.Tensor,  # [N] int64
        seq_end_per_req: torch.Tensor,  # [B] int64
    ) -> torch.Tensor:
        """state_slot[t] = state_block_table[b, (pos//tpb)%max_blocks] * eb + pos%eb.

        State pools are SWA-type ring tables. Block-table indexing uses
        ``_state_tokens_per_block`` (physical block size) with modulo
        wrapping; in-block ring offset uses ``_state_eb``.
        Returns -1 where the resolved block_id is negative (unallocated).

        Ring write mask: only the last R positions before each block boundary
        (or sequence end) actually write. Earlier positions whose ring entries
        would be overwritten by later tokens in the same block are masked to -1.
        """
        bt = self._state_block_table
        eb = self._state_eb
        tpb = self._state_tokens_per_block
        assert bt is not None and eb > 0, "state pool context unbound"
        if self._kv_cache_sharded and self._cp_ctx is not None:
            from rtp_llm.models_py.modules.dsv4.fp8._cp_slot_mapping import (
                cp_state_slot_mapping,
            )

            return cp_state_slot_mapping(
                positions,
                bt,
                b_idx,
                eb,
                tpb,
                self._cp_ctx.cp_size,
                self._cp_ctx.cp_rank,
                seq_end_per_req=seq_end_per_req,
            )
        bt_long = bt.to(torch.long)
        max_blocks = int(bt_long.shape[1])
        if max_blocks <= 0:
            return torch.full_like(positions, -1)
        block_in_seq = (positions // tpb) % max_blocks
        in_block = positions % eb
        block_id = bt_long[b_idx, block_in_seq]
        valid = block_id > 0
        block_end = (positions // tpb + 1) * tpb
        seq_end = seq_end_per_req[b_idx]
        effective_end = torch.minimum(block_end, seq_end)
        valid = valid & ((positions + eb) >= effective_end)
        slot = block_id * eb + in_block
        return torch.where(valid, slot, torch.full_like(slot, -1))

    def _compute_kv_slot_mapping(
        self,
        positions: torch.Tensor,  # [N] int64
        b_idx: torch.Tensor,  # [N] int64
    ) -> torch.Tensor:
        """KV-pool slot for each token. -1 unless (pos+1) % ratio == 0
        (i.e. boundary token that produces a compressed entry).

        Block addressing follows the framework convention for FULL paged
        pools: the block_table is indexed in raw-token space using
        ``_kv_tokens_per_block``. The KV pool's per-block entry count is
        ``kv_eb = kernel_tokens_per_block / ratio``, so the in-block offset
        is the compressed-entry offset within that raw-token block.

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
        tokens_per_block = self._kv_tokens_per_block
        if bt is None or kv_eb <= 0 or tokens_per_block <= 0:
            return torch.full_like(positions, -1)
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
                owner_tokens_per_block=self._state_tokens_per_block,
            )
            if self._kv_pool_view is not None:
                pool_rows = int(
                    self._kv_pool_view.numel() // self._kv_pool_view.shape[-1]
                )
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
        if self._kv_pool_view is not None:
            pool_rows = int(self._kv_pool_view.numel() // self._kv_pool_view.shape[-1])
            valid = valid & (slot < pool_rows)
        return torch.where(valid, slot, torch.full_like(slot, -1))

    def _launch(
        self,
        kv_flat: torch.Tensor,  # [N, coff*head_dim] fp32
        score_flat: torch.Tensor,  # [N, coff*head_dim] fp32
        meta: CompressorMeta,
    ) -> None:
        """Launch the two vLLM kernels (state write + boundary KV write).

        All slot-mapping math is consumed from ``meta`` — this method only
        does kernel launches and the cos_sin_cache lazy build. The fused
        writer always consumes per-request raw-window metadata, including the
        B==1 path, so there is no scalar ``seq_start`` fallback.
        """
        if (
            self._state_pool_3d is None
            or self._kv_pool_view is None
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

        if meta.seq_start_per_req is None or meta.cu_seq_per_req is None:
            raise RuntimeError(
                "CompressorFP8 requires per-request raw metadata: "
                "seq_start_per_req/cu_seq_per_req"
            )
        state_cache_for_read = self._state_pool_3d
        state_block_table_for_read = self._state_block_table
        if _cp_sliced_state_read_needed(meta, self._cp_ctx):
            with record_function_range(
                "dsv4.fp8.compressor.launch.cp_gather_state_read_cache"
            ):
                state_cache_for_read, state_block_table_for_read = (
                    _build_cp_full_state_read_cache(
                        self._state_pool_3d,
                        self._state_block_table,
                        int(self._cp_ctx.cp_size),
                    )
                )

        with record_function_range("dsv4.fp8.compressor.launch.compress_kv_write"):
            run_fused_compress_kv_write(
                state_cache_for_read,
                meta.token_to_req,
                meta.positions,
                meta.state_slots,
                state_block_table_for_read,
                self.norm.weight,
                self.norm_eps,
                cos_sin_cache,
                self._kv_pool_view,
                meta.kv_slots,
                kv_flat,
                score_flat,
                self.ape,
                0,
                disable_raw_path=False,
                head_dim=self.head_dim,
                rope_head_dim=self.rope_head_dim,
                compress_ratio=self.compress_ratio,
                overlap=self.overlap,
                seq_start_per_req=meta.seq_start_per_req,
                cu_seq_per_req=meta.cu_seq_per_req,
                state_tokens_per_block=self._state_tokens_per_block,
            )

    # ----------------------------------------------------------------------
    # Overlap orchestration: split-phase prefill (start / finish).
    #
    # The baseline :meth:`forward` below is a strict superset of these two
    # split-phase methods (start + finish performed sequentially). The
    # overlap orchestrator in attention.py uses the split-phase variant so
    # it can interleave the CP all-gather (queued on a side stream by
    # ``start_prefill``) with default-stream compute (SWA pool write,
    # indexer compute_q) before waiting on the gather inside
    # ``finish_prefill``.
    #
    # Contract:
    #   * Every ``start_prefill`` call must be paired with exactly one
    #     ``finish_prefill`` call on the returned handle.
    #   * On warmup (no pool bound) ``start_prefill`` returns ``None`` and
    #     ``finish_prefill(None)`` is a no-op — callers can chain
    #     unconditionally without warmup branching.
    #   * ``cp_gather_stream`` is the caller-owned CP gather stream. Sharing
    #     one stream across multiple compressor calls in the same layer
    #     (e.g. CSA: nested indexer + main) is required for FIFO ordering
    #     of NCCL collectives within the ProcessGroup.
    # ----------------------------------------------------------------------
    def start_prefill(
        self,
        x: torch.Tensor,
        start_pos,
        *,
        meta: CompressorMeta,
        cp_gather_stream: Optional[Any] = None,
        profile_label: Optional[str] = None,
    ) -> Optional[_CompressorPending]:
        """Begin a prefill compressor launch without waiting on the CP gather.

        Steps performed eagerly on the **default** stream:
          * shape resolve (mirrors ``forward``);
          * warmup early return → ``None`` (no pool bound by framework);
          * fused KV/gate projection (``_linear_bf16_bf16_fp32``).

        Steps deferred to :meth:`finish_prefill`:
          * waiting the CP all-gather (queued here on
            ``cp_gather_stream`` when CP is active);
          * splitting fused → kv_flat + score_flat;
          * the ``_launch`` writer kernel.
        """
        if x.dim() == 2:
            bsz = 1
            seqlen = int(x.size(0))
        else:
            bsz, seqlen, _ = x.size()
        if (
            self._state_pool_3d is None
            or self._kv_pool_view is None
            or self._kv_eb <= 0
        ):
            return None
        assert meta is not None, "CompressorFP8.start_prefill requires CompressorMeta"

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
            N_full = int(cp_ctx.seq_len_full)
            assert int(meta.positions.numel()) == N_full, (
                f"CP compressor meta/token length mismatch: "
                f"meta={meta.positions.numel()} seq_len_full={N_full}"
            )
            restored_buf = None
            if not cp_ctx.unpad_restore_is_prefix:
                restored_buf = torch.empty(
                    (N_full, int(fused_flat.size(1))),
                    dtype=fused_flat.dtype,
                    device=fused_flat.device,
                )
            gather_stream = cp_gather_stream
            if gather_stream is None and fused_flat.is_cuda:
                gather_stream = torch.cuda.Stream(device=fused_flat.device)
            profile_name = self._cp_profile_name(profile_label)
            gather_range = "dsv4.fp8.compressor.prefill.cp_gather_kv_score"
            if profile_label:
                gather_range = (
                    f"dsv4.fp8.compressor.prefill.{profile_label}.cp_gather_kv_score"
                )
            with record_function_range(gather_range):
                fused_gather_handle = cp_all_gather_full_async(
                    fused_flat,
                    cp_ctx,
                    stream=gather_stream,
                    restored_buf=restored_buf,
                    profile_name=profile_name,
                )
        else:
            restored_buf = None

        return _CompressorPending(
            fused_flat=fused_flat,
            fused_gather_handle=fused_gather_handle,
            meta=meta,
            out_dim=out_dim,
            profile_label=profile_label,
            restored_buf=restored_buf,
        )

    def wait_prefill_gather(self, pending: Optional[_CompressorPending]) -> None:
        """Fence the split prefill CP gather without writing the FP8 pool.

        CSA overlap uses this to ensure the main compressor's NCCL gather has
        completed before indexer score/topk, while preserving the baseline
        order where the main CSA pool write happens after indexer topk.
        """
        if pending is None or pending.fused_gather_handle is None:
            return
        wait_range = "dsv4.fp8.compressor.prefill.cp_wait_kv_score"
        if pending.profile_label:
            wait_range = (
                f"dsv4.fp8.compressor.prefill.{pending.profile_label}.cp_wait_kv_score"
            )
        with record_function_range(wait_range):
            pending.fused_flat = cp_wait_gather_full(pending.fused_gather_handle)
            pending.fused_gather_handle = None

    def finish_prefill(self, pending: Optional[_CompressorPending]) -> None:
        """Drain a :meth:`start_prefill` and write the FP8 KV pool.

        Mirrors the second half of :meth:`forward` (wait → split → launch).
        ``pending=None`` is a no-op (warmup early-return path from
        ``start_prefill``).
        """
        if pending is None:
            return  # warmup, mirrors forward()'s early return
        self.wait_prefill_gather(pending)
        fused_flat = pending.fused_flat
        out_dim = pending.out_dim
        meta = pending.meta
        assert meta is not None, "CompressorFP8.finish_prefill requires CompressorMeta"

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

        with record_function_range("dsv4.fp8.compressor.prefill.launch"):
            self._launch(kv_flat, score_flat, meta)

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
        """Prefill entry.

        Returns ``None`` — downstream readers gather compressed K from the
        FP8 KV pool directly.

        ``meta`` lets the caller (typically ``attention.py``) hoist the
        slot-mapping compute out of the per-layer hot path and amortize it
        across the host compressor and any nested indexer compressor that
        share the same positions/b_idx. Once pools are bound, callers must
        provide this metadata; warmup still exits before metadata is used.
        """
        del sequence_lengths  # not needed: positions derived from start_pos+arange
        # Accept either flat ``[T_total, dim]`` or ``[B, S, dim]``. The
        # compressor kernels consume token-major 2D tensors, so prefill
        # flattens once after the fused projection and never reintroduces B.
        if x.dim() == 2:
            bsz = 1
            seqlen = int(x.size(0))
        else:
            bsz, seqlen, _ = x.size()
        # Warmup forward (no pool bound by framework): no-op.
        if (
            self._state_pool_3d is None
            or self._kv_pool_view is None
            or self._kv_eb <= 0
        ):
            return None
        assert meta is not None, "CompressorFP8.forward requires CompressorMeta"

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
                    fused_flat,
                    cp_ctx,
                    stream=gather_stream,
                    profile_name=self._cp_profile_name(None),
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
        with record_function_range("dsv4.fp8.compressor.prefill.launch"):
            self._launch(kv_flat, score_flat, meta)
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
        if (
            self._state_pool_3d is None
            or self._kv_pool_view is None
            or self._kv_eb <= 0
        ):
            return None
        assert (
            meta is not None
        ), "CompressorFP8.forward_decode_vectorized requires CompressorMeta"

        out_dim = (1 + self.overlap) * self.head_dim
        fused_out = _linear_bf16_bf16_fp32(x, self._wkv_wgate_fused)
        kv, score = fused_out[..., :out_dim], fused_out[..., out_dim:]

        # ``kv`` and ``score`` are last-dim slices of ``fused_out``. Keep
        # them as strided row views instead of materialising two BF16 copy
        # kernels; the Triton writer consumes row stride explicitly.
        kv_flat = kv.view(T, -1)
        score_flat = score.view(T, -1)
        self._launch(kv_flat, score_flat, meta)
        return None


def build_prepare_metadata_args(
    *,
    device: torch.device,
    position_ids: Optional[torch.Tensor] = None,
    req_id_per_token: Optional[torch.Tensor] = None,
    seq_start_per_req: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Return the kwargs dict for ``CompressorFP8.prepare_metadata``.

    Per-request metadata is required. Single source of truth for the three
    call sites that used to inline this dispatch:

      * ``Attention._build_compressor_meta`` (HCA + standalone CSA path)
      * ``Attention._build_csa_prefill_meta`` (CSA inline that shares the
        pool bind with ``IndexerFP8.prepare``)
      * ``IndexerFP8.prepare`` nested compressor hoist

    ``seq_start_per_req`` accepts either ``sp_per_req`` (Attention) or
    ``prefix_lengths`` (Indexer) — both index32-cast to the same int32
    layout the compressor's varlen raw kernel consumes.
    """
    if (
        position_ids is None
        or req_id_per_token is None
        or seq_start_per_req is None
        or cu_seqlens is None
    ):
        raise RuntimeError(
            "CompressorFP8 requires varlen metadata: "
            "position_ids/req_id_per_token/seq_start_per_req/cu_seqlens"
        )
    return dict(
        positions=position_ids.to(device=device, dtype=torch.long)
        .reshape(-1)
        .contiguous(),
        b_idx=req_id_per_token.to(device=device, dtype=torch.long)
        .reshape(-1)
        .contiguous(),
        seq_start_per_req=seq_start_per_req.to(device=device, dtype=torch.int32)
        .reshape(-1)
        .contiguous(),
        cu_seq_per_req=cu_seqlens.to(device=device, dtype=torch.int32)
        .reshape(-1)
        .contiguous(),
    )


def build_decode_metadata(
    compressor: "CompressorFP8", start_pos: torch.Tensor, bsz: int
) -> CompressorMeta:
    device = start_pos.device
    positions = start_pos.to(device=device, dtype=torch.long).reshape(bsz)
    b_idx = torch.arange(bsz, device=device, dtype=torch.long)
    cu_seq_per_req = torch.arange(0, bsz + 1, device=device, dtype=torch.int64)
    return compressor.prepare_metadata(
        positions,
        b_idx,
        seq_start_per_req=positions,
        cu_seq_per_req=cu_seq_per_req,
    )
