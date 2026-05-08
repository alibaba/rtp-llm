"""DeepSeek-V4 Compressor — vLLM per-token state-pool flow, BF16 KV pool.

Adaptation of ``new_ft/.../dsv4/fp8/compressor.py`` (per-token state
pool + fused boundary writer Triton kernels) that emits BF16 directly to
the same ``[total_slots, head_dim]`` BF16 KV pool that the existing
``Compressor`` writes — i.e. **no FP8 quantization, no UE8M0 scales, no
584B / 132B FP8 slot layout**. Drop-in replacement for ``Compressor``
when ``DSV4_COMPRESSOR_VLLM=1`` is set.

Why this exists: the vLLM-style flow trades a complex per-logical-block
state buffer for two single-launch Triton kernels (per-token state write
+ fused boundary compress→norm→rope→store). For the BF16 path we keep
exactly the same outer flow but swap the FP8 quant tail for a single
``store(bf16)`` to the existing BF16 KV pool, so downstream readers
(``flash_mla_sparse_fwd`` for CSA / dense MQA for HCA / DeepGEMM
``fp8_paged_mqa_logits`` indexer reader's BF16 sibling) work un
changed.

Public API mirrors :class:`Compressor` so attention.py / indexer.py
construction sites can swap classes without touching call sites:
  * ``set_pool_context(kv_view, kv_bt, kv_eb, state_view, state_bt, state_eb)``
  * ``set_cp_ctx(cp_ctx)``, ``configure_kv_cache_shape(kv_cache_t)``
  * ``forward(x, start_pos, sequence_lengths=None)`` (prefill)
  * ``forward_decode_vectorized(x, start_pos)`` and
    ``forward_decode(x, start_pos)`` aliases (decode)
  * ``freqs_cis`` attribute bound by Attention / Indexer
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4._compressor_vllm_triton import (
    build_cos_sin_cache,
    run_fused_compress_kv_write_bf16,
    run_save_partial_states,
)
from rtp_llm.models_py.modules.dsv4.cp import (
    CPContext,
    cp_all_gather_full_async,
    cp_should_gather,
    cp_wait_gather_full,
)


@dataclass(frozen=True)
class CompressorMeta:
    """Pre-computed per-token launch metadata.

    Built once per (state_block_table, kv_block_table, positions, b_idx)
    tuple — typically by the attention layer just after
    ``set_pool_context``, so the math is amortized across both the host
    compressor and any nested indexer compressor that shares the same
    positions/b_idx layout.

    Fields are device tensors of length ``N_tok``:
      * ``positions``    : int64 absolute token positions
      * ``b_idx``        : int64 request index per token
      * ``state_slots``  : int64 state-pool slot per token (-1 = skip)
      * ``kv_slots``     : int64 KV-pool slot per token (-1 if non-boundary
                           or unallocated)
      * ``token_to_req`` : int32 alias of ``b_idx`` for the fused KV writer
      * ``boundary_token_indices`` : int64 token indices that write KV slots
    """

    positions: torch.Tensor
    b_idx: torch.Tensor
    state_slots: torch.Tensor
    kv_slots: torch.Tensor
    token_to_req: torch.Tensor
    boundary_token_indices: torch.Tensor


class _CompressorNorm(nn.Module):
    """RMSNorm weight holder — bf16 (Triton kernel reads bf16 weight)."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))


class CompressorVLLM(nn.Module):
    """vLLM-style per-token state-pool compressor with BF16 KV-pool writeback.

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
        "norm"}`` extracted by the caller from
        ``layer_weights[W.v4_*compressor_*]``."""
        super().__init__()
        assert head_dim in (
            128,
            512,
        ), f"CompressorVLLM supports head_dim in {{128, 512}}; got {head_dim}"
        assert compressor_weights is not None, (
            "CompressorVLLM requires compressor_weights — meta-tensor / "
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

        # ape / wkv / wgate stay FP32 — accumulation happens in FP32 inside
        # the state pool; vLLM keeps the same convention. Register ape as a
        # non-trainable Parameter so .to(device) follows the module.
        self.ape = nn.Parameter(
            compressor_weights["ape"].float().contiguous(), requires_grad=False
        )
        self.wkv = nn.Linear(dim, coff * head_dim, bias=False)
        self.wgate = nn.Linear(dim, coff * head_dim, bias=False)
        with torch.no_grad():
            self.wkv.weight = nn.Parameter(
                compressor_weights["wkv"].float(), requires_grad=False
            )
            self.wgate.weight = nn.Parameter(
                compressor_weights["wgate"].float(), requires_grad=False
            )
        self.norm = _CompressorNorm(head_dim)
        self.norm.weight = nn.Parameter(
            compressor_weights["norm"].to(torch.bfloat16),
            requires_grad=False,
        )

        # Fuse wkv + wgate into one fp32 weight matrix; saves one SIMT
        # SGEMM launch per compressor decode call.
        self._wkv_wgate_fused: Optional[torch.Tensor] = None
        self._fuse_wkv_wgate(coff)

        # Pool context — populated by attention's _set_compressor_pool_context.
        self._state_pool_3d: Optional[torch.Tensor] = None
        self._state_block_table: Optional[torch.Tensor] = None
        self._state_eb: int = 0
        self._kv_pool_view: Optional[torch.Tensor] = None
        self._kv_block_table: Optional[torch.Tensor] = None
        self._kv_eb: int = 0

        # Legacy attribute kept so attention.py's cmp_T fallback keeps
        # working when probing a CompressorVLLM instance.
        self._kv_cache_t: int = 0

        # Cached cos_sin cache built from self.freqs_cis at first forward.
        self.freqs_cis: Optional[torch.Tensor] = None
        self._cos_sin_cache: Optional[torch.Tensor] = None
        self._cp_ctx: Optional[CPContext] = None
        self._dbg_prefix: Optional[str] = None

    def _fuse_wkv_wgate(self, coff: int) -> None:
        """Concat wkv + wgate along out-dim into one fused fp32 weight,
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
    # Compatibility shims — match the BF16 ``Compressor`` API surface.
    # ----------------------------------------------------------------------
    def configure_kv_cache_shape(self, kv_cache_t: int) -> None:
        """Stores ``_kv_cache_t`` only as informational metadata so legacy
        readers (e.g. ``attention.py`` ``cmp_T`` fallback) keep working.
        The vLLM path does NOT allocate any per-step bf16 cache from this."""
        self._kv_cache_t = int(kv_cache_t)

    def set_cp_ctx(self, cp_ctx: Optional[CPContext]) -> None:
        self._cp_ctx = cp_ctx

    # ----------------------------------------------------------------------
    # Pool context lifecycle (6-arg signature matches Compressor)
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

        ``kv_pool_view``    : 2D ``[total_slots, head_dim]`` bf16 (BF16 KV pool).
        ``state_pool_view`` : 2D flat ``[total_slots, 2*coff*head_dim]`` fp32
                              from ``_pool_view``. Reshaped here to
                              ``[num_blocks, state_eb, 2*coff*head_dim]``.
        """
        self._kv_pool_view = kv_pool_view
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
        self._kv_pool_view = None
        self._kv_block_table = None
        self._kv_eb = 0

    # ----------------------------------------------------------------------
    # Metadata preparation (call once per forward, OFF the hot path)
    # ----------------------------------------------------------------------
    def prepare_metadata(
        self,
        positions: torch.Tensor,  # [N] int64
        b_idx: torch.Tensor,  # [N] int64
    ) -> CompressorMeta:
        """Compute slot mappings + token_to_req from current pool context."""
        state_slots = self._compute_state_slot_mapping(positions, b_idx)
        kv_slots = self._compute_kv_slot_mapping(positions, b_idx)
        token_to_req = b_idx.to(torch.int32)
        boundary_token_indices = torch.nonzero(kv_slots >= 0, as_tuple=False).flatten()
        return CompressorMeta(
            positions=positions,
            b_idx=b_idx,
            state_slots=state_slots,
            kv_slots=kv_slots,
            token_to_req=token_to_req,
            boundary_token_indices=boundary_token_indices,
        )

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------
    def _ensure_cos_sin_cache(self, device: torch.device) -> torch.Tensor:
        if self._cos_sin_cache is None or self._cos_sin_cache.device != device:
            assert (
                self.freqs_cis is not None
            ), "CompressorVLLM.freqs_cis must be bound before forward"
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

        Block addressing follows the framework convention: DSV4 KV pools
        use ``seq_size_per_block = TOKENS_PER_BLOCK`` for block_table
        indexing — i.e. the block_table is indexed in *token* space, not
        compressed-entry space. The KV pool's per-block entry count is
        ``kv_eb = TOKENS_PER_BLOCK / ratio``, so the in-block offset is the
        compressed-entry offset *within that token block*.
        """
        bt = self._kv_block_table
        kv_eb = self._kv_eb
        ratio = self.compress_ratio
        if bt is None or kv_eb <= 0:
            return torch.full_like(positions, -1)
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
        valid = boundary & in_capacity & (block_id > 0)
        slot = block_id * kv_eb + in_block
        return torch.where(valid, slot, torch.full_like(slot, -1))

    def _launch(
        self,
        kv_flat: torch.Tensor,  # [N, coff*head_dim] fp32
        score_flat: torch.Tensor,  # [N, coff*head_dim] fp32
        meta: CompressorMeta,
        seq_start: Optional[int] = None,
    ) -> None:
        """Launch the two Triton kernels (state write + boundary KV write).

        ``seq_start`` is the absolute position of ``kv_flat[0]`` for
        sequentially-laid-out batches (prefill). When provided the fused
        kernel reads any overlap-window position with
        ``flat_idx = pos - seq_start in [0, N)`` directly from
        ``kv_flat / score_flat`` instead of the cyclic state pool, which
        only retains the latest few hundred tokens per request and would
        have been overwritten within this same launch by
        ``run_save_partial_states``.

        Pass ``None`` to disable the raw path (decode: ``kv_flat`` is
        indexed by ``req_idx``, not by absolute position offset).
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

        cos_sin_cache = self._ensure_cos_sin_cache(kv_flat.device)

        run_save_partial_states(
            kv_flat,
            score_flat,
            self.ape,
            meta.positions,
            self._state_pool_3d,
            meta.state_slots,
            compress_ratio=self.compress_ratio,
        )

        raw_disabled = seq_start is None
        run_fused_compress_kv_write_bf16(
            self._state_pool_3d,
            meta.token_to_req,
            meta.positions,
            meta.state_slots,
            self._state_block_table.to(torch.int32),
            self.norm.weight,
            self.norm_eps,
            cos_sin_cache,
            self._kv_pool_view,
            meta.kv_slots,
            kv_flat,
            score_flat,
            self.ape,
            0 if raw_disabled else int(seq_start),
            disable_raw_path=raw_disabled,
            boundary_token_indices=meta.boundary_token_indices,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            compress_ratio=self.compress_ratio,
            overlap=self.overlap,
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
        BF16 KV pool directly (same pool layout as the existing
        :class:`Compressor` writes).
        """
        del sequence_lengths  # not needed: positions derived from start_pos+arange
        bsz, seqlen, _ = x.size()

        # CP prefill: absolute prefix start is cp_ctx.prefix_length, NOT
        # the per-rank start_pos (which is the local-chunk offset).
        cp_ctx = self._cp_ctx
        if cp_ctx is not None and cp_ctx.cp_size > 1:
            sp = int(cp_ctx.prefix_length)
        elif isinstance(start_pos, torch.Tensor):
            sp = int(start_pos.item()) if start_pos.numel() == 1 else 0
        else:
            sp = int(start_pos)

        # Warmup forward (no pool bound by framework): no-op.
        if (
            self._state_pool_3d is None
            or self._kv_pool_view is None
            or self._kv_eb <= 0
        ):
            return None

        device = x.device
        x32 = x.float()
        out_dim = (1 + self.overlap) * self.head_dim
        fused_out = torch.nn.functional.linear(x32, self._wkv_wgate_fused)
        kv, score = fused_out[..., :out_dim], fused_out[..., out_dim:]

        cp_gather = cp_should_gather(cp_ctx, start_pos)
        kv_gather_handle = None
        score_gather_handle = None
        if cp_gather:
            assert cp_ctx is not None
            if kv.dim() != 3 or kv.size(0) != 1:
                raise RuntimeError(
                    f"vLLM CP compressor KV expects [1, T_local, H], got {tuple(kv.shape)}"
                )
            if score.dim() != 3 or score.size(0) != 1:
                raise RuntimeError(
                    f"vLLM CP compressor score expects [1, T_local, H], got {tuple(score.shape)}"
                )
            gather_stream = torch.cuda.Stream(device=kv.device) if kv.is_cuda else None
            kv_gather_handle = cp_all_gather_full_async(
                kv.squeeze(0), cp_ctx, stream=gather_stream
            )
            score_gather_handle = cp_all_gather_full_async(
                score.squeeze(0), cp_ctx, stream=gather_stream
            )
            bsz, seqlen = 1, cp_ctx.seq_len_full

        if meta is None:
            positions, b_idx = _build_prefill_positions(sp, bsz, seqlen, device)
            meta = self.prepare_metadata(positions, b_idx)

        if cp_gather:
            assert kv_gather_handle is not None
            assert score_gather_handle is not None
            kv = cp_wait_gather_full(kv_gather_handle).unsqueeze(0)
            score = cp_wait_gather_full(score_gather_handle).unsqueeze(0)

        N = bsz * seqlen
        kv_flat = kv.reshape(N, -1).contiguous()
        score_flat = score.reshape(N, -1).contiguous()
        self._launch(kv_flat, score_flat, meta, seq_start=sp)
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
        if (
            self._state_pool_3d is None
            or self._kv_pool_view is None
            or self._kv_eb <= 0
        ):
            return None

        device = x.device
        x32 = x.float()
        out_dim = (1 + self.overlap) * self.head_dim
        fused_out = torch.nn.functional.linear(x32, self._wkv_wgate_fused)
        kv, score = fused_out[..., :out_dim], fused_out[..., out_dim:]

        kv_flat = kv.reshape(bsz, -1).contiguous()
        score_flat = score.reshape(bsz, -1).contiguous()
        if meta is None:
            positions = start_pos.to(device=device, dtype=torch.long).reshape(bsz)
            b_idx = torch.arange(bsz, device=device, dtype=torch.long)
            meta = self.prepare_metadata(positions, b_idx)
        self._launch(kv_flat, score_flat, meta)
        return None

    # API parity with Compressor: scalar-loop forward_decode aliases the
    # vectorized form (the vLLM path does not need a separate Python loop).
    def forward_decode(
        self,
        x: torch.Tensor,
        start_pos: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return self.forward_decode_vectorized(x, start_pos)


# ---------------------------------------------------------------------------
# Free helpers — exposed so the attention layer can build positions/b_idx
# once per forward and feed them through ``prepare_metadata`` / ``forward``.
# ---------------------------------------------------------------------------
def _build_prefill_positions(
    sp: int, bsz: int, seqlen: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``positions = sp + arange(seqlen)`` broadcast to ``[bsz, seqlen]`` →
    flat ``[bsz*seqlen]``; ``b_idx = repeat_interleave(arange(bsz))``."""
    N = bsz * seqlen
    positions = (
        (torch.arange(seqlen, device=device, dtype=torch.long).unsqueeze(0) + sp)
        .expand(bsz, -1)
        .reshape(N)
        .contiguous()
    )
    b_idx = (
        torch.arange(bsz, device=device, dtype=torch.long)
        .unsqueeze(1)
        .expand(-1, seqlen)
        .reshape(N)
        .contiguous()
    )
    return positions, b_idx


def build_prefill_metadata(
    compressor: "CompressorVLLM",
    sp: int,
    bsz: int,
    seqlen: int,
    device: torch.device,
) -> CompressorMeta:
    """Convenience: build positions/b_idx + ``CompressorMeta`` in one call."""
    positions, b_idx = _build_prefill_positions(sp, bsz, seqlen, device)
    return compressor.prepare_metadata(positions, b_idx)


def build_decode_metadata(
    compressor: "CompressorVLLM", start_pos: torch.Tensor, bsz: int
) -> CompressorMeta:
    device = start_pos.device
    positions = start_pos.to(device=device, dtype=torch.long).reshape(bsz)
    b_idx = torch.arange(bsz, device=device, dtype=torch.long)
    return compressor.prepare_metadata(positions, b_idx)
