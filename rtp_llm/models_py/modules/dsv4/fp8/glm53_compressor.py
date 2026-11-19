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
  * ``forward_decode_vectorized(x, start_pos)`` for batched decode.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.distributed.glm53_collective_gemm import all_gather
from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.models_py.modules.dsv4.fp8._compressor_consts import INDEXER_HEAD_DIM
from rtp_llm.models_py.modules.dsv4.fp8.compressor import (
    CompressorFP8 as _CompressorFP8,
)
from rtp_llm.models_py.modules.dsv4.fp8.compressor import (
    CompressorMeta,
    _cp_sliced_state_read_needed,
    _linear_bf16_bf16_fp32,
)
from rtp_llm.models_py.modules.dsv4.fp8.glm53_compressor_triton import (
    build_cos_sin_cache,
    run_fused_compress_kv_write,
    run_fused_kpool_decode_update,
    run_save_partial_states,
)

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
    """Gather CP-sliced fixed-state blocks into a compact full-ring read cache.

    CP prefill stores INDEXER/CSA/HCA state as intra-block slices: each rank
    owns ``local_eb`` rows of the full ``local_eb * cp_size`` state ring.
    The fused compressor may still need to read prefix-cache history that
    spans all state-ring rows.  For that read path only, gather the rows for
    the current request block table into compact block ids while keeping the
    write path local/sliced.
    """
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


class CompressorFP8(_CompressorFP8):
    """KPool normalization, cache metadata and writes for GLM's FP8 indexer."""

    _prefill_request_index_dtype = torch.int32

    def __init__(
        self,
        dim: int,
        head_dim: int,
        rope_head_dim: int,
        compress_ratio: int,
        max_batch_size: int,
        *,
        cp_role: str,
        norm_eps: float = 1e-6,
        rotate: bool = False,
        compressor_weights: Optional[Dict[str, torch.Tensor]] = None,
        overlap: Optional[bool] = None,
        kpool_mode: bool = False,
        pre_norm_weight: Optional[torch.Tensor] = None,
        pre_norm_bias: Optional[torch.Tensor] = None,
    ):
        """``compressor_weights`` is a 4-key dict ``{"ape", "wkv", "wgate",
        "norm"}`` extracted by the caller from ``layer_weights[W.v4_*compressor_*]``."""
        resolved_overlap = compress_ratio == 4 if overlap is None else bool(overlap)
        if kpool_mode and (
            head_dim != INDEXER_HEAD_DIM or resolved_overlap or rope_head_dim != 0
        ):
            raise ValueError(
                "GLM-5.3-Flash KPool requires head_dim=128, rope_head_dim=0, "
                "and overlap disabled"
            )
        if kpool_mode and (pre_norm_weight is None or pre_norm_bias is None):
            raise ValueError("GLM-5.3-Flash KPool requires LayerNorm weight and bias")
        super().__init__(
            dim,
            head_dim,
            rope_head_dim,
            compress_ratio,
            max_batch_size,
            cp_role=cp_role,
            norm_eps=norm_eps,
            rotate=rotate,
            compressor_weights=compressor_weights,
            overlap=resolved_overlap,
        )
        self.kpool_mode = bool(kpool_mode)
        self.pre_norm_weight = (
            nn.Parameter(pre_norm_weight.to(torch.bfloat16), requires_grad=False)
            if pre_norm_weight is not None
            else None
        )
        self.pre_norm_bias = (
            nn.Parameter(pre_norm_bias.to(torch.bfloat16), requires_grad=False)
            if pre_norm_bias is not None
            else None
        )

        self._cos_sin_cache_key: Optional[
            Tuple[int, torch.device, Tuple[int, ...], torch.dtype]
        ] = None

    def _normalize_kpool_key(self, key: torch.Tensor) -> torch.Tensor:
        """Apply the checkpoint's per-token LayerNorm before KPool."""
        if not self.kpool_mode:
            return key
        assert self.pre_norm_weight is not None and self.pre_norm_bias is not None
        return F.layer_norm(
            key.to(torch.bfloat16),
            (self.head_dim,),
            self.pre_norm_weight,
            self.pre_norm_bias,
            self.norm_eps,
        ).float()

    def prepare_metadata(
        self,
        positions: torch.Tensor,  # [N] int64
        b_idx: torch.Tensor,  # [N] int64
        has_prefix: bool,
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
                has_prefix=has_prefix,
                is_batched=is_batched,
                seq_start_per_req=seq_start_per_req,
                cu_seq_per_req=cu_seq_per_req,
            )

        assert seq_start_per_req is not None and cu_seq_per_req is not None, (
            "seq_start_per_req and cu_seq_per_req are required for ring write mask; "
            "caller must supply per-request metadata"
        )
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
                seq_start_per_req,
                cu_seq_per_req,
                self._state_tokens_per_block,
                pool_rows=pool_rows,
                kv_tokens_per_block=self._kv_tokens_per_block,
            )
            return CompressorMeta(
                positions=positions,
                b_idx=b_idx,
                state_slots=state_slots,
                kv_slots=kv_slots,
                token_to_req=token_to_req,
                has_prefix=has_prefix,
                is_batched=is_batched,
                seq_start_per_req=seq_start_per_req,
                cu_seq_per_req=cu_seq_per_req,
            )

        input_lens = (cu_seq_per_req[1:] - cu_seq_per_req[:-1]).to(torch.long)
        seq_end_per_req = seq_start_per_req.to(torch.long) + input_lens
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
            has_prefix=has_prefix,
            is_batched=is_batched,
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
        self._cos_sin_cache_key = key
        return shared

    def _launch(
        self,
        kv_flat: torch.Tensor,  # [N, coff*head_dim] fp32
        score_flat: torch.Tensor,  # [N, coff*head_dim] fp32
        meta: CompressorMeta,
        seq_start: Optional[int] = None,
        *,
        is_decode: bool = False,
    ) -> None:
        """Launch the two vLLM kernels (state write + boundary KV write).

        ``seq_start`` is the absolute position of ``kv_flat[0]`` for
        sequentially-laid-out batches (prefill: ``sp_int``). When provided
        the fused kernel reads any overlap-window position with
        ``flat_idx = pos - seq_start in [0, N)`` directly from
        ``kv_flat / score_flat`` instead of reading back through the state
        pool, where current-launch writes are still in flight.

        Pass ``None`` to disable the raw path (decode and batched prefill use
        different layouts that both require it). ``is_decode`` is explicit so
        batched prefill can never enter the decode-only KPool fusion.

        All slot-mapping math is consumed from ``meta`` — this method only
        does kernel launches and the cos_sin_cache lazy build. Designed to
        stay branch-light so it composes cleanly with CUDA graph capture.
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

        kv_flat = self._normalize_kpool_key(kv_flat)
        use_fused_kpool_decode = (
            self.kpool_mode
            and is_decode
            and seq_start is None
            and meta.seq_start_per_req is not None
            and meta.cu_seq_per_req is not None
        )
        if use_fused_kpool_decode:
            batch_size = int(meta.seq_start_per_req.numel())
            if batch_size <= 0 or N % batch_size:
                raise RuntimeError(
                    "fused KPool decode requires uniform per-request tokens: "
                    f"tokens={N}, batch={batch_size}"
                )
            q_len = N // batch_size
            with record_function_range(
                "dsv4.fp8.compressor.launch.fused_kpool_decode_update"
            ):
                run_fused_kpool_decode_update(
                    self._state_pool_3d,
                    self._state_block_table,
                    meta.state_slots.view(batch_size, q_len),
                    self._kv_pool_view,
                    meta.kv_slots.view(batch_size, q_len),
                    kv_flat.view(batch_size, q_len, self.head_dim),
                    score_flat.view(batch_size, q_len, self.head_dim),
                    self.ape,
                    meta.positions.view(batch_size, q_len),
                    state_tokens_per_block=self._state_tokens_per_block,
                    pool_size=self.compress_ratio,
                )
            return

        with record_function_range("dsv4.fp8.compressor.launch.cos_sin_cache"):
            cos_sin_cache = (
                kv_flat.new_empty((1, 0), dtype=torch.float32)
                if self.kpool_mode
                else self._ensure_cos_sin_cache(kv_flat.device)
            )

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
        state_cache_for_read = self._state_pool_3d
        state_block_table_for_read = self._state_block_table
        if _cp_sliced_state_read_needed(meta, self._cp_ctx, raw_disabled):
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
                0 if (raw_disabled or use_varlen_raw) else seq_start,
                disable_raw_path=raw_disabled,
                head_dim=self.head_dim,
                rope_head_dim=self.rope_head_dim,
                compress_ratio=self.compress_ratio,
                overlap=self.overlap,
                seq_start_per_req=meta.seq_start_per_req if use_varlen_raw else None,
                cu_seq_per_req=meta.cu_seq_per_req if use_varlen_raw else None,
                state_tokens_per_block=self._state_tokens_per_block,
                kpool_mode=self.kpool_mode,
            )

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
                cu_seq_per_req = torch.arange(
                    0, bsz + 1, device=device, dtype=torch.int64
                )
                meta = self.prepare_metadata(
                    positions,
                    b_idx,
                    has_prefix=True,
                    seq_start_per_req=positions,
                    cu_seq_per_req=cu_seq_per_req,
                )
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
                    has_prefix=True,
                    is_batched=q_len > 1,
                    seq_start_per_req=position_ids_2d[:, 0]
                    .to(torch.int32)
                    .contiguous(),
                    cu_seq_per_req=cu_seq_per_req,
                )
        self._launch(kv_flat, score_flat, meta, is_decode=True)
        return None
