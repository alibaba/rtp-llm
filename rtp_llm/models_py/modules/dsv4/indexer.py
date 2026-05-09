"""DeepSeek-V4 lightning Indexer for CSA.

Faithful BF16 port of `inference/model.py:Indexer`. Skips Hadamard rotation
+ FP4 quant (BF16-only path for M2/M3 correctness validation).

Has its own dedicated Compressor (rotate=True in official code; we keep
the parameter for ckpt-loader symmetry but don't apply Hadamard).
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4.compressor import Compressor


def _use_rope_only_kernel() -> bool:
    import os

    return os.environ.get("DSV4_INDEXER_ROPE_ONLY", "1") != "0"


from rtp_llm.models_py.modules.dsv4._metadata_triton import build_pool_slots
from rtp_llm.models_py.modules.dsv4._profiler import record_function_range
from rtp_llm.models_py.modules.dsv4.cp import CPContext, cp_freqs_cis_local
from rtp_llm.models_py.modules.dsv4.indexer_topk import select_indexer_topk
from rtp_llm.models_py.modules.dsv4.qlinear import QuantizedLinear
from rtp_llm.models_py.modules.dsv4.rope import (
    apply_rotary_emb,
    apply_rotary_emb_batched,
)


class Indexer(nn.Module):
    def __init__(
        self,
        dim: int,
        q_lora_rank: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        index_topk: int,
        compress_ratio: int,
        max_batch_size: int,
        max_seq_len: int,
        norm_eps: float = 1e-6,
        layer_weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """``layer_weights`` is the framework's per-layer dict
        (``ModelWeights.weights[layer_id]``), keyed by ``W.v4_*`` enum.
        Indexer reads ``W.v4_indexer_wq_b_{w,s}``, ``W.v4_indexer_weights_proj_w``,
        and the four ``W.v4_indexer_compressor_*`` keys for its nested
        ``Compressor``."""
        super().__init__()
        self.dim = dim
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        self.softmax_scale = self.head_dim**-0.5
        self.compress_ratio = compress_ratio

        from rtp_llm.models_py.modules.dsv4.attention import _v4_fp8_linear
        from rtp_llm.utils.model_weight import W

        self.wq_b = _v4_fp8_linear(
            layer_weights[W.v4_indexer_wq_b_w],
            layer_weights[W.v4_indexer_wq_b_s],
        )
        # weights_proj is a plain BF16 weight tensor; forward calls
        # ``F.linear(x, self.weights_proj)`` directly.
        self.weights_proj = layer_weights[W.v4_indexer_weights_proj_w]

        inner_cmp_weights = {
            "ape": layer_weights[W.v4_indexer_compressor_ape],
            "wkv": layer_weights[W.v4_indexer_compressor_wkv],
            "wgate": layer_weights[W.v4_indexer_compressor_wgate],
            "norm": layer_weights[W.v4_indexer_compressor_norm],
        }
        self.compressor = Compressor(
            dim=dim,
            head_dim=index_head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            max_batch_size=max_batch_size,
            norm_eps=norm_eps,
            rotate=True,
            compressor_weights=inner_cmp_weights,
        )
        # Pool-only (#50): self.kv_cache is NEVER a persistent Python-owned
        # tensor.  Each forward / forward_decode call gathers the indexer's
        # prefix compressed-K rows from the INDEXER_KV pool on entry, binds
        # onto this ephemeral attr for the scoring body, and clears to None
        # on exit.  The nested compressor.forward_* does its own bind/scatter
        # against INDEXER_KV + INDEXER_STATE (pool context set by this
        # Indexer in its own wrapper), so no scatter is needed here.
        self.max_batch_size = max_batch_size
        self._kv_cache_t = max_seq_len // compress_ratio
        self._kv_cache_d = index_head_dim
        self.kv_cache: Optional[torch.Tensor] = None
        self.freqs_cis: Optional[torch.Tensor] = None
        # CP context bound per-forward by V4Transformer; None = no CP.
        self._cp_ctx: Optional[CPContext] = None
        # Optional per-forward debug prefix (e.g. "L01_attn_idx"); when set,
        # internal forward checkpoints are recorded via _record_tensor.
        # Also propagated to the nested compressor as f"{prefix}_cmp".
        self._dbg_prefix: Optional[str] = None

        # Pool context — bound per forward call by Attention via
        # ``set_pool_context`` and cleared via ``clear_pool_context``.  When
        # unbound (standalone tests) the helpers fall back to zeros.
        self._kv_pool_view: Optional[torch.Tensor] = None
        self._kv_block_table: Optional[torch.Tensor] = None
        self._kv_eb: int = 0
        self._state_pool_view: Optional[torch.Tensor] = None
        self._state_block_table: Optional[torch.Tensor] = None
        self._state_eb: int = 0

    # ------------------------------------------------------------------
    # #50 — pool-only lifecycle helpers.
    # ------------------------------------------------------------------

    def set_pool_context(
        self,
        kv_pool_view: Optional[torch.Tensor],
        kv_block_table: Optional[torch.Tensor],
        kv_eb: int,
        state_pool_view: Optional[torch.Tensor],
        state_block_table: Optional[torch.Tensor],
        state_eb: int,
    ) -> None:
        """Bind INDEXER_KV + INDEXER_STATE views / block tables for one
        forward call.  Attention sets this before ``forward`` /
        ``forward_decode*``; ``clear_pool_context`` clears after."""
        self._kv_pool_view = kv_pool_view
        self._kv_block_table = kv_block_table
        self._kv_eb = kv_eb
        self._state_pool_view = state_pool_view
        self._state_block_table = state_block_table
        self._state_eb = state_eb

    def clear_pool_context(self) -> None:
        self._kv_pool_view = None
        self._kv_block_table = None
        self._kv_eb = 0
        self._state_pool_view = None
        self._state_block_table = None
        self._state_eb = 0

    def _propagate_pool_to_nested(self) -> None:
        """Hand the same INDEXER_KV + INDEXER_STATE pool context to the
        nested compressor so its own forward_* wrappers bind / write /
        scatter compressed-K + state into the pool we share."""
        if self.compressor is None:
            return
        self.compressor.set_pool_context(
            self._kv_pool_view,
            self._kv_block_table,
            self._kv_eb,
            self._state_pool_view,
            self._state_block_table,
            self._state_eb,
        )

    def _clear_nested_pool(self) -> None:
        if self.compressor is None:
            return
        self.compressor.clear_pool_context()

    def _bind_kv_cache_from_pool(
        self,
        bsz: int,
        is_fresh_prefill: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Populate ``self.kv_cache`` at the start of a forward call.
        Fresh prefill → zeros.  Decode / continuation → gather from
        INDEXER_KV pool (zero for sentinel slots).  Standalone-test
        fallback (no pool context) also returns zeros.
        """
        T = self._kv_cache_t
        D = self._kv_cache_d
        if (
            is_fresh_prefill
            or self._kv_pool_view is None
            or self._kv_block_table is None
            or self._kv_eb <= 0
        ):
            self.kv_cache = torch.zeros(bsz, T, D, dtype=dtype, device=device)
            return
        eb = self._kv_eb
        with record_function_range("dsv4.indexer.pool_bind_slot_meta"):
            fast = build_pool_slots(self._kv_block_table, bsz=bsz, T=T, eb=eb)
            if fast is None:
                max_blocks = self._kv_block_table.shape[1]
                pool_capacity = max_blocks * eb
                pos = torch.arange(T, device=device, dtype=torch.long)
                in_capacity_row = pos < pool_capacity
                safe_pos = torch.where(in_capacity_row, pos, torch.zeros_like(pos))
                block_in_seq = safe_pos // eb
                in_block = safe_pos % eb
                bt_long = self._kv_block_table.to(torch.long)
                b_idx = torch.arange(bsz, device=device, dtype=torch.long).unsqueeze(1)
                block_id = bt_long[:bsz][b_idx, block_in_seq.unsqueeze(0)]
                in_capacity = in_capacity_row.unsqueeze(0).expand(bsz, -1)
                valid = (block_id > 0) & in_capacity
                safe_slot = torch.where(
                    valid,
                    block_id * eb + in_block.unsqueeze(0),
                    torch.zeros_like(block_id),
                )
            else:
                valid, safe_slot = fast
        from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt

        if self._dbg_prefix is not None and _rt.ENABLED:
            _rt.record_if_level(2, f"{self._dbg_prefix}_kv_bind_slot", safe_slot)
            _rt.record_if_level(
                2, f"{self._dbg_prefix}_kv_bind_valid", valid.to(torch.int32)
            )
        with record_function_range("dsv4.indexer.pool_bind_kv"):
            from rtp_llm.models_py.modules.dsv4._pool_triton import (
                masked_gather_from_pool,
            )

            self.kv_cache = masked_gather_from_pool(
                self._kv_pool_view,
                safe_slot,
                valid,
                out_shape=(bsz, T, D),
                dtype=dtype,
            ).contiguous()

    def _write_current_compressed_to_pool(
        self,
        compressed: Optional[torch.Tensor],
        start_pos,
    ) -> None:
        if (
            compressed is None
            or self._kv_pool_view is None
            or self._kv_block_table is None
            or self._kv_eb <= 0
            or compressed.numel() == 0
        ):
            return
        cp_ctx = self._cp_ctx
        if cp_ctx is not None and cp_ctx.cp_size > 1:
            sp = int(cp_ctx.prefix_length)
        elif isinstance(start_pos, torch.Tensor):
            if start_pos.numel() != 1:
                return
            sp = int(start_pos.item())
        else:
            sp = int(start_pos)
        bsz = int(compressed.shape[0])
        n = int(compressed.shape[1])
        if n <= 0:
            return

        device = compressed.device
        eb = int(self._kv_eb)
        with record_function_range("dsv4.indexer.write_current_slot_meta"):
            write_start = sp // self.compress_ratio
            pos = torch.arange(
                write_start, write_start + n, device=device, dtype=torch.long
            )
            block_in_seq = pos // eb
            in_block = pos % eb
            in_capacity = block_in_seq < int(self._kv_block_table.shape[1])
            safe_block = torch.where(
                in_capacity, block_in_seq, torch.zeros_like(block_in_seq)
            )
            bt_long = self._kv_block_table[:bsz].to(device=device, dtype=torch.long)
            b_idx = torch.arange(bsz, device=device, dtype=torch.long).unsqueeze(1)
            block_id = bt_long[b_idx, safe_block.unsqueeze(0).expand(bsz, -1)]
            valid = in_capacity.unsqueeze(0) & (block_id > 0)
            safe_slot = torch.where(
                valid, block_id * eb + in_block.unsqueeze(0), torch.zeros_like(block_id)
            )
            slot_mapping = torch.where(
                valid, safe_slot, torch.full_like(safe_slot, -1)
            ).reshape(-1)
        from rtp_llm.models_py.modules.dsv4.decode.kv_write_decode_op import (
            write_kv_to_pool,
        )

        with record_function_range("dsv4.indexer.write_current_to_pool"):
            write_kv_to_pool(
                compressed[:bsz].reshape(bsz * n, -1),
                slot_mapping,
                self._kv_pool_view,
                mask_negative=True,
            )

    def _overlay_current_compressed(
        self,
        compressed: Optional[torch.Tensor],
        start_pos,
    ) -> None:
        if compressed is None or self.kv_cache is None or compressed.numel() == 0:
            return
        cp_ctx = self._cp_ctx
        if cp_ctx is not None and cp_ctx.cp_size > 1:
            sp = int(cp_ctx.prefix_length)
        elif isinstance(start_pos, torch.Tensor):
            if start_pos.numel() != 1:
                return
            sp = int(start_pos.item())
        else:
            sp = int(start_pos)
        write_start = sp // self.compress_ratio
        n = int(compressed.shape[1])
        write_end = min(write_start + n, int(self.kv_cache.shape[1]))
        if write_end <= write_start:
            return
        self.kv_cache[: compressed.shape[0], write_start:write_end] = compressed[
            :, : write_end - write_start
        ].to(self.kv_cache.dtype)

    def set_cp_ctx(self, cp_ctx: Optional[CPContext]) -> None:
        """Bind CP context.  When active, rank-local Q applies RoPE at
        GLOBAL positions (not rank-local row indices); the causal mask
        over the compressed-KV axis uses global positions; and the score
        einsum reads the nested compressor's ``kv_cache[:, :seq_len_full
        // ratio]`` which was just populated with the full compressed KV
        by ``Compressor.forward``'s all-gather path.

        The outer ``offset`` passed by ``Attention`` is the number of
        sliding-window KV slots that precede the compressed-KV block in
        the concatenated ``[sliding | compressed]`` layout — under CP
        the sliding slots equal ``seq_len_full``, not ``chunk_length``.
        ``Attention`` passes the already-computed offset; this method
        only needs to expose the context so Indexer can position the
        causal mask / topk-mask relative to global Q positions."""
        self._cp_ctx = cp_ctx

    def forward_decode(
        self,
        x: torch.Tensor,  # [B, 1, dim] bf16 — single decode token per req
        qr: torch.Tensor,  # [B, 1, q_lora_rank] bf16 — q_a output
        start_pos: torch.Tensor,  # [B] int32 — abs pos per request
        out_topk_buffer: torch.Tensor,  # [B, 1, K=index_topk] int32 — pre-allocated by metadata builder
    ) -> torch.Tensor:
        """Batched decode-time indexer.

        Per request r: runs the small Compressor step (already
        ``self.compressor.forward_decode``-friendly), computes per-token
        index score against ``self.kv_cache[r, :compressed_len[r]]``,
        and writes top-K indices into ``out_topk_buffer[r]``.

        For requests with ``compressed_len < K``, the unused slots are
        filled with -1 (downstream sparse_attn masks them).

        Decode-only — does NOT touch the prefill ``forward`` arm.
        """
        assert x.shape[1] == 1, "decode-only: q_len must be 1"
        bsz = x.size(0)
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        K = self.index_topk

        self._propagate_pool_to_nested()
        try:
            # Compressor decode step gathers its own state + kv_cache from the
            # INDEXER_KV / INDEXER_STATE pool, writes the new compressed-K
            # slot, and scatters back on exit (pool-only #50).
            self.compressor.forward_decode(x, start_pos)

            # qr -> wq_b -> [B, 1, n_heads * head_dim] -> unflatten
            if qr.dim() > 2:
                shape = qr.shape
                q = self.wq_b(qr.reshape(-1, shape[-1])).view(
                    *shape[:-1],
                    self.n_heads * self.head_dim,
                )
            else:
                q = self.wq_b(qr)
            q = q.unflatten(-1, (self.n_heads, self.head_dim))  # [B, 1, H_idx, D_idx]

            # Per-request RoPE on q_pe (each request has its own start_pos).
            freqs_cis_per_req = self.freqs_cis[start_pos.long()]  # [B, freqs_dim]
            apply_rotary_emb_batched(q[..., -rd:], freqs_cis_per_req)

            # weights = weights_proj(x) * scale
            weights = F.linear(x, self.weights_proj) * (
                self.softmax_scale * self.n_heads**-0.5
            )  # [B, 1, H_idx]

            # Gather the up-to-date INDEXER_KV pool (now includes the slot
            # the nested compressor just scattered) for the scoring step.
            self._bind_kv_cache_from_pool(
                bsz, is_fresh_prefill=False, device=x.device, dtype=torch.bfloat16
            )

            # score against per-request compressed-K cache slice + topk.
            # compressed_len[r] = (start_pos[r] + 1) // ratio (post-step length).
            compressed_len = ((start_pos + 1) // ratio).to(torch.int64)  # [B]
            out_topk_buffer.fill_(-1)
            for r in range(bsz):
                T_r = int(compressed_len[r].item())
                if T_r <= 0:
                    continue
                kv_r = self.kv_cache[r : r + 1, :T_r].contiguous()  # [1, T_r, D_idx]
                q_r = q[r : r + 1].contiguous()  # [1, 1, H_idx, D_idx]
                w_r = weights[r : r + 1]  # [1, 1, H_idx]
                from rtp_llm.models_py.modules.dsv4._indexer_score_triton import (
                    v4_indexer_score,
                )

                score = v4_indexer_score(
                    q_r, kv_r, w_r, q_pos=None, compress_ratio=1
                )  # [1, 1, T_r]
                k_r = min(K, T_r)
                topk_r = select_indexer_topk(
                    score,
                    k_r,
                    lengths=compressed_len[r : r + 1].to(torch.int32),
                )
                out_topk_buffer[r : r + 1, :, :k_r] = topk_r
            return out_topk_buffer
        finally:
            self._clear_nested_pool()
            self.kv_cache = None

    def forward_decode_vectorized(
        self,
        x: torch.Tensor,  # [B, 1, dim] bf16
        qr: torch.Tensor,  # [B, 1, q_lora_rank] bf16
        start_pos: torch.Tensor,  # [B] int32
        out_topk_buffer: torch.Tensor,  # [B, 1, K]
    ) -> torch.Tensor:
        """Stage 3B vectorized variant of :meth:`forward_decode`.

        No Python loops over B, no ``.item()`` calls. The compressor step
        uses the vectorized variant; per-request RoPE uses
        ``apply_rotary_emb_batched``; the score / topk is computed
        batched across B with a length mask (positions beyond
        ``compressed_len[r]`` are set to ``-inf`` so ``topk`` returns
        valid indices for the leading prefix and arbitrary indices for
        the masked tail). The caller masks via ``compressed_lens`` from
        the metadata (already pre-computed).

        Result shape: ``out_topk_buffer`` modified in place; returns
        the same tensor for caller convenience.
        """
        assert x.shape[1] == 1, "decode-only: q_len must be 1"

        bsz = x.size(0)
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        K = self.index_topk

        self._propagate_pool_to_nested()
        try:
            # Compressor decode (vectorized) self-binds from INDEXER_KV +
            # INDEXER_STATE pool, writes the new compressed-K slot, scatters
            # back, and clears its kv_cache / state to None.
            self.compressor.forward_decode_vectorized(x, start_pos)

            # qr -> wq_b -> [B, 1, n_heads * head_dim] -> [B, 1, H_idx, D_idx]
            if qr.dim() > 2:
                shape = qr.shape
                q = self.wq_b(qr.reshape(-1, shape[-1])).view(
                    *shape[:-1],
                    self.n_heads * self.head_dim,
                )
            else:
                q = self.wq_b(qr)
            q = q.unflatten(-1, (self.n_heads, self.head_dim))

            # Per-request batched RoPE on q_pe.
            freqs_per_b = self.freqs_cis[start_pos.long()]  # [B, freqs_dim]
            apply_rotary_emb_batched(q[..., -rd:], freqs_per_b)

            weights = F.linear(x, self.weights_proj) * (
                self.softmax_scale * self.n_heads**-0.5
            )  # [B, 1, H_idx]

            # Fresh pool read reflects the compressor's just-scattered slot.
            self._bind_kv_cache_from_pool(
                bsz, is_fresh_prefill=False, device=x.device, dtype=torch.bfloat16
            )

            # Batched score against the FULL kv_cache prefix (length
            # max_seq_len // ratio). Mask invalid positions to -inf so topk
            # only returns valid leading indices.
            compressed_len = ((start_pos + 1) // ratio).to(torch.int64).view(bsz, 1, 1)
            T_max = self.kv_cache.shape[1]
            from rtp_llm.models_py.modules.dsv4._indexer_score_triton import (
                v4_indexer_score,
            )

            score = v4_indexer_score(
                q.contiguous(),
                self.kv_cache[:bsz].contiguous(),
                weights,
                q_pos=None,
                compress_ratio=1,
            )

            K_eff = min(K, T_max)
            out_topk_buffer.fill_(-1)
            if K_eff > 0:
                topk_idxs = select_indexer_topk(
                    score,
                    K_eff,
                    lengths=compressed_len.view(-1),
                )
                out_topk_buffer[:, :, :K_eff].copy_(topk_idxs)

            return out_topk_buffer
        finally:
            self._clear_nested_pool()
            self.kv_cache = None

    def _forward_batched_prefill(
        self,
        x: torch.Tensor,  # [B, max_S, dim]
        qr: torch.Tensor,  # [B, max_S, q_lora_rank]
        start_pos: torch.Tensor,  # [B] int
        offset: int,
        sequence_lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Per-row batched prefill dispatch for the Indexer.

        #50 pool-only: each row's ``self.forward(x_b, qr_b, sp_b)`` call
        handles its own bind/scatter against a narrowed [1, max_blocks]
        INDEXER_KV / INDEXER_STATE block table.  Pool views are kept
        global (addressed by absolute slot id); only the block tables are
        narrowed per row.
        """
        bsz = int(x.size(0))
        max_S = int(x.size(1))
        device = x.device

        if sequence_lengths is None:
            seq_t = torch.full((bsz,), max_S, device=device, dtype=torch.long)
        else:
            seq_t = sequence_lengths.to(device=device, dtype=torch.long)
        sp_t = start_pos.to(device=device, dtype=torch.long)

        end_pos_t = sp_t + seq_t
        k_per_row = torch.minimum(
            torch.full_like(end_pos_t, self.index_topk),
            end_pos_t // self.compress_ratio,
        )
        K_max = int(k_per_row.max().item()) if bsz > 0 else 0
        if K_max <= 0:
            return torch.full((bsz, max_S, 0), -1, dtype=torch.long, device=device)
        out = torch.full((bsz, max_S, K_max), -1, dtype=torch.long, device=device)

        saved_kv_bt = self._kv_block_table
        saved_state_bt = self._state_block_table
        saved_dbg = self._dbg_prefix
        try:
            self._dbg_prefix = None
            for b in range(bsz):
                seq_b = int(seq_t[b].item())
                if seq_b == 0:
                    continue
                sp_b = int(sp_t[b].item())
                x_b = x[b : b + 1, :seq_b]
                qr_b = qr[b : b + 1, :seq_b]
                self._kv_block_table = (
                    saved_kv_bt[b : b + 1] if saved_kv_bt is not None else None
                )
                self._state_block_table = (
                    saved_state_bt[b : b + 1] if saved_state_bt is not None else None
                )
                topk_b = self.forward(x_b, qr_b, sp_b, offset)  # [1, seq_b, K_b]
                k_b = int(topk_b.size(-1))
                if k_b > 0:
                    out[b : b + 1, :seq_b, :k_b] = topk_b
        finally:
            self._kv_block_table = saved_kv_bt
            self._state_block_table = saved_state_bt
            self._dbg_prefix = saved_dbg

        return out

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos,
        offset: int,
        sequence_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt

        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        cp_ctx = self._cp_ctx
        is_batched = isinstance(start_pos, torch.Tensor) and start_pos.numel() > 1
        is_prefill = seqlen > 1

        # Batched prefill dispatch (seqlen > 1 with [B] start_pos).  The
        # scalar body below is called per-row from _forward_batched_prefill.
        if is_batched and seqlen > 1:
            return self._forward_batched_prefill(
                x, qr, start_pos, offset, sequence_lengths
            )

        cp_on = (
            cp_ctx is not None and cp_ctx.cp_size > 1 and not is_batched and is_prefill
        )
        # Master switch: gated by both MOEDBG (process-wide) and per-instance
        # _dbg_prefix (set externally; e.g. attention layer-0..2 wiring).
        _dbg = self._dbg_prefix if _rt.ENABLED else None

        if cp_on:
            freqs_cis = cp_freqs_cis_local(self.freqs_cis, cp_ctx)
            end_pos = cp_ctx.seq_len_total
        elif is_batched:
            positions = start_pos.long()
            freqs_cis = self.freqs_cis[positions].unsqueeze(1)  # [B, 1, rope_dim//2]
            end_pos = (start_pos.max() + seqlen).item()
        else:
            sp = int(start_pos) if isinstance(start_pos, torch.Tensor) else start_pos
            freqs_cis = self.freqs_cis[sp : sp + seqlen]
            end_pos = sp + seqlen

        if self.compressor.freqs_cis is None:
            self.compressor.freqs_cis = self.freqs_cis

        self._propagate_pool_to_nested()
        try:
            if qr.dim() > 2:
                shape = qr.shape
                q = self.wq_b(qr.reshape(-1, shape[-1])).view(
                    *shape[:-1],
                    self.n_heads * self.head_dim,
                )
            else:
                q = self.wq_b(qr)
            q = q.unflatten(-1, (self.n_heads, self.head_dim))
            if _dbg is not None:
                _rt.record_if_level(2, f"{_dbg}_q_pre_rope", q)
                _rt.record_if_level(2, f"{_dbg}_freqs_cis", freqs_cis)
            q_rope = q[..., -rd:]
            if q_rope.is_cuda and _use_rope_only_kernel():
                from rtp_llm.models_py.modules.dsv4._rope_only_triton import (
                    rope_only_inplace,
                )

                rope_only_inplace(q_rope, freqs_cis)
            else:
                apply_rotary_emb(q_rope, freqs_cis)
            if _dbg is not None:
                _rt.record_if_level(2, f"{_dbg}_q_post_rope", q)

            # Nested compressor: reads its own _cp_ctx set by V4Transformer,
            # all-gathers rank-local kv/score → writes full compressed KV
            # into the INDEXER_KV pool via its own bind/scatter lifecycle.
            if _dbg is not None:
                self.compressor._dbg_prefix = f"{_dbg}_cmp"
            with record_function_range("dsv4.indexer.nested_compressor"):
                current_compressed = self.compressor(x, start_pos)
            self._write_current_compressed_to_pool(current_compressed, start_pos)
            if _dbg is not None:
                self.compressor._dbg_prefix = None

            # Fresh read of INDEXER_KV pool picks up the slots the nested
            # compressor just scattered — gives us the prefix compressed-K
            # rows needed for scoring below.
            self._bind_kv_cache_from_pool(
                bsz, is_fresh_prefill=False, device=x.device, dtype=torch.bfloat16
            )
            self._overlay_current_compressed(current_compressed, start_pos)
            if _dbg is not None:
                _rt.record_if_level(
                    2,
                    f"{_dbg}_compressor_kv_cache",
                    self.kv_cache[:bsz, : end_pos // ratio],
                )
            with record_function_range("dsv4.indexer.weights_proj_full"):
                weights = F.linear(x, self.weights_proj)
            weights = weights * (self.softmax_scale * self.n_heads**-0.5)
            if _dbg is not None:
                _rt.record_if_level(2, f"{_dbg}_weights", weights)

            kv = self.kv_cache[:bsz, : end_pos // ratio]
            # Causal mask: each prefill Q token at absolute position g can
            # only read compressed KV blocks [0, (g+1)//ratio).  This applies
            # to both fresh and continuation prefill because the suffix
            # compressor writes all current blocks before index scoring.
            is_prefill = (not is_batched) and seqlen > 1
            if is_prefill:
                if cp_on:
                    q_pos = cp_ctx.global_positions.to(torch.int32).unsqueeze(0)
                else:
                    sp = (
                        int(start_pos.item())
                        if isinstance(start_pos, torch.Tensor)
                        else int(start_pos)
                    )
                    q_pos = (
                        sp + torch.arange(seqlen, device=x.device, dtype=torch.int32)
                    ).unsqueeze(0)
                # Kernel computes thr=(q_pos+1)//ratio, matching the prior
                # post-add mask (kv_col >= (q_pos+1)//ratio → -inf).
                with record_function_range("dsv4.indexer.score"):
                    from rtp_llm.models_py.modules.dsv4._indexer_score_triton import (
                        v4_indexer_score,
                    )

                    index_score = v4_indexer_score(
                        q.contiguous(),
                        kv.contiguous(),
                        weights,
                        q_pos=q_pos.expand(bsz, seqlen).contiguous(),
                        compress_ratio=ratio,
                    )
            else:
                with record_function_range("dsv4.indexer.score"):
                    from rtp_llm.models_py.modules.dsv4._indexer_score_triton import (
                        v4_indexer_score,
                    )

                    index_score = v4_indexer_score(
                        q.contiguous(),
                        kv.contiguous(),
                        weights,
                        q_pos=None,
                        compress_ratio=1,
                    )
            if _dbg is not None:
                _rt.record_if_level(2, f"{_dbg}_score_post_mask", index_score)

            with record_function_range("dsv4.indexer.topk"):
                k_eff = min(self.index_topk, end_pos // ratio)
                topk_lengths = None
                topk_offset: int | torch.Tensor = 0
                if is_prefill:
                    if cp_on:
                        q_pos_1 = (cp_ctx.global_positions + 1).to(torch.int32)
                    else:
                        sp = (
                            int(start_pos.item())
                            if isinstance(start_pos, torch.Tensor)
                            else int(start_pos)
                        )
                        q_pos_1 = (
                            sp
                            + torch.arange(seqlen, device=x.device, dtype=torch.int32)
                            + 1
                        )
                    topk_lengths = (
                        (q_pos_1 // ratio).unsqueeze(0).expand(bsz, seqlen).reshape(-1)
                    )
                topk_idxs = select_indexer_topk(
                    index_score,
                    k_eff,
                    lengths=topk_lengths,
                    offset=topk_offset,
                )
            if _dbg is not None:
                _rt.record_if_level(2, f"{_dbg}_topk_pre_offset", topk_idxs)
            with record_function_range("dsv4.indexer.topk_postprocess"):
                if is_prefill:
                    topk_idxs = torch.where(
                        topk_idxs >= 0,
                        topk_idxs + offset,
                        topk_idxs,
                    )
                else:
                    topk_idxs = topk_idxs + offset
            if _dbg is not None:
                _rt.record_if_level(2, f"{_dbg}_topk_final", topk_idxs)
            return topk_idxs.long()
        finally:
            self._clear_nested_pool()
            self.kv_cache = None
