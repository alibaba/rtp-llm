"""DeepSeek-V4 Transformer Block: mHC + Attention + MoE.

Mirrors `inference/model.py:Block`. Each call applies hc_pre/F/hc_post
twice — once for Attention and once for MoE FFN.
"""

import logging
import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.modules import RMSNorm
from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8
from rtp_llm.models_py.modules.dsv4.hc import build_hc_unit
from rtp_llm.models_py.modules.dsv4.moe import MoE


class Block(nn.Module):
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
        moe_inter_dim: int,
        n_routed_experts: int,
        n_activated_experts: int,
        n_shared_experts: int,
        score_func: str,
        route_scale: float,
        swiglu_limit: float,
        n_hash_layers: int,
        vocab_size: int,
        hc_mult: int,
        hc_sinkhorn_iters: int,
        hc_eps: float,
        norm_eps: float = 1e-6,
        layer_weights: Optional[Dict[str, torch.Tensor]] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        ep_size: int = 1,
        ep_rank: int = 0,
        max_tokens_per_rank: int = 8192,
        is_decode_role: bool = False,
        fp8_kv_cache: bool = False,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.fp8_kv_cache = fp8_kv_cache

        attn_cls = AttentionFP8
        self.attn = attn_cls(
            layer_id=layer_id,
            dim=dim,
            n_heads=n_heads,
            q_lora_rank=q_lora_rank,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            o_lora_rank=o_lora_rank,
            o_groups=o_groups,
            window_size=window_size,
            compress_ratio=compress_ratio,
            compress_rope_theta=compress_rope_theta,
            rope_theta=rope_theta,
            rope_factor=rope_factor,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            original_seq_len=original_seq_len,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            norm_eps=norm_eps,
            layer_weights=layer_weights,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        self._cp_sync_after_attn_done = False
        self.ffn = MoE(
            layer_id=layer_id,
            dim=dim,
            moe_inter_dim=moe_inter_dim,
            n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts,
            n_shared_experts=n_shared_experts,
            score_func=score_func,
            route_scale=route_scale,
            swiglu_limit=swiglu_limit,
            n_hash_layers=n_hash_layers,
            vocab_size=vocab_size,
            layer_weights=layer_weights,
            ep_size=ep_size,
            ep_rank=ep_rank,
            max_tokens_per_rank=max_tokens_per_rank,
            is_decode_role=is_decode_role,
        )
        # Framework loader already casts norms to bf16 (compute_dtype) and
        # hc_* tensors to fp32 (descriptor data_type); pass refs straight
        # into ``RMSNorm`` at construction time.  Norms here see 2D inputs
        # ``[T, dim]`` from the hc_pre output, so framework ``RMSNorm``
        # (which expects 2D) drops in directly.
        from rtp_llm.utils.model_weight import W

        self.attn_norm = RMSNorm(layer_weights[W.v4_attn_norm], norm_eps)
        self.ffn_norm = RMSNorm(layer_weights[W.v4_ffn_norm], norm_eps)

        self.attn_hc = build_hc_unit(
            layer_weights[W.v4_hc_attn_fn],
            layer_weights[W.v4_hc_attn_base],
            layer_weights[W.v4_hc_attn_scale],
            dim=dim,
            hc_mult=hc_mult,
            hc_sinkhorn_iters=hc_sinkhorn_iters,
            norm_eps=norm_eps,
            hc_eps=hc_eps,
            layer_id=layer_id,
            name="attn",
        )
        self.ffn_hc = build_hc_unit(
            layer_weights[W.v4_hc_ffn_fn],
            layer_weights[W.v4_hc_ffn_base],
            layer_weights[W.v4_hc_ffn_scale],
            dim=dim,
            hc_mult=hc_mult,
            hc_sinkhorn_iters=hc_sinkhorn_iters,
            norm_eps=norm_eps,
            hc_eps=hc_eps,
            layer_id=layer_id,
            name="ffn",
        )

    def _sync_after_first_cp_prefill_attention(self) -> None:
        if self._cp_sync_after_attn_done:
            return
        if os.environ.get("DSV4_CP_SYNC_AFTER_ATTN_ONCE", "1") == "0":
            return
        if getattr(getattr(self.ffn, "_strategy", None), "name", "") not in (
            "mega",
            "mega_fused",
        ):
            return
        if getattr(self.attn, "_cp_ctx", None) is None:
            return

        import torch.distributed as dist

        if not (
            dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        ):
            return

        torch.cuda.synchronize()
        try:
            dist.barrier(device_ids=[torch.cuda.current_device()])
        except TypeError:
            dist.barrier()
        self._cp_sync_after_attn_done = True
        logging.info(
            "[DeepSeekV4Block] CP first-prefill sync after attention done layer=%d",
            self.layer_id,
        )

    def forward_decode(
        self,
        x: torch.Tensor,  # [B, 1, hc, dim]
        attn_metadata: "DSv4DecodeAttnMetadata",  # type: ignore[name-defined]
        input_ids: torch.Tensor,  # [B, 1]
        kv_cache=None,
    ) -> torch.Tensor:
        """Decode-only block forward — mirrors prefill ``forward`` but
        delegates attention to ``Attention.forward_decode``. Prefill
        ``forward`` is byte-identical for PD-disagg cleanliness.
        """
        from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt

        _dbg_layer = _rt.should_record_layer(self.layer_id)
        # Attention path
        residual = x
        x_pre, post, comb = self.attn_hc.pre(
            x,
            dbg_tag=f"L{self.layer_id:02d}_decode_attn_hc_pre" if _dbg_layer else None,
        )
        # Framework RMSNorm wants 2D — collapse [B, q_len, dim] → [B*q_len, dim]
        # and view back; attention.forward_decode wants the original 3D shape.
        bsz, q_len, dim_ = x_pre.shape
        x_pre = self.attn_norm(x_pre.reshape(bsz * q_len, dim_)).view(bsz, q_len, dim_)
        if _dbg_layer:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_decode_attn_in", x_pre)
        attn_out = self.attn.forward_decode(x_pre, attn_metadata, kv_cache=kv_cache)
        if _dbg_layer:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_decode_attn_out", attn_out)
        x = self.attn_hc.post(attn_out, residual, post, comb)
        if _dbg_layer:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_decode_attn_residual", x)

        # FFN path — MoE has no per-step state, reuse existing forward
        residual = x
        x_pre, post, comb = self.ffn_hc.pre(
            x,
            dbg_tag=f"L{self.layer_id:02d}_decode_ffn_hc_pre" if _dbg_layer else None,
        )
        bsz, q_len, dim_ = x_pre.shape
        x_pre = self.ffn_norm(x_pre.reshape(bsz * q_len, dim_)).view(bsz, q_len, dim_)
        if _dbg_layer:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_decode_ffn_in", x_pre)
        ffn_out = self.ffn(x_pre, input_ids)
        if _dbg_layer:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_decode_ffn_out", ffn_out)
        x = self.ffn_hc.post(ffn_out, residual, post, comb)
        return x

    def forward(
        self,
        x: torch.Tensor,  # [T, hc, dim]
        input_ids: Optional[torch.Tensor],  # [T]
        positions: torch.Tensor,  # [T] int64
        cu_seqlens: torch.Tensor,  # [B+1] int64
        kv_cache=None,
        block_tables_by_type=None,
    ) -> torch.Tensor:
        """Flat per-block forward — accepts ``[T, hc, dim]`` hidden and 1D
        ``input_ids`` / ``positions`` / ``cu_seqlens``, matching the vLLM
        DSV4 layer layout.  HC pre/post are flat-native;
        ``self.ffn`` (MoE) reshapes to 2D internally so any shape flows.

        Attention is called with a padded ``[B, max_S, dim]`` tensor and
        per-row ``start_pos [B]`` / ``sequence_lengths [B]`` derived from
        ``cu_seqlens`` + ``positions``.  For B==1 the scatter/gather
        collapses to an identity copy and Attention's internal scalar
        path kicks in (start_pos tensor with ``numel()==1`` falls through
        to ``int(start_pos)``), so the single-request case is bit-equal
        to the pre-batched behavior.
        """
        from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt

        # Master switch: when MOEDBG=0 the AND short-circuits so neither the
        # layer_id compare nor any record_if_level call site below runs.
        # By default keep the narrow trace from _record_tensor; use
        # MOEDBG_LAYER/MOEDBG_ALL_LAYERS for wider bisection.
        _dbg_layer = _rt.should_record_layer(self.layer_id)
        dbg_pos = getattr(_rt, "_DBG_GLOBAL_POS", -1)
        dbg_pos_mask = None
        dbg_pos_name = None
        dbg_positions = positions
        cp_ctx = getattr(self.attn, "_cp_ctx", None)
        if cp_ctx is not None and getattr(cp_ctx, "global_positions", None) is not None:
            cp_positions = cp_ctx.global_positions
            if cp_positions.numel() == positions.numel():
                dbg_positions = cp_positions
        if _dbg_layer and dbg_pos >= 0:
            dbg_pos_mask = dbg_positions.to(torch.long) == int(dbg_pos)
            dbg_pos_name = f"pos{dbg_pos}"
        # Attention path
        residual = x
        x_pre, post, comb = self.attn_hc.pre(
            x,
            dbg_tag=f"L{self.layer_id:02d}_attn_hc_pre" if _dbg_layer else None,
        )  # [T, dim], [T, hc, 1], [T, hc, hc]
        x_pre = self.attn_norm(x_pre)  # [T, dim]
        if _dbg_layer:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_attn_in", x_pre)
            if dbg_pos_mask is not None:
                _rt.record_if_level(
                    2,
                    f"L{self.layer_id:02d}_attn_in_{dbg_pos_name}",
                    x_pre[dbg_pos_mask].contiguous(),
                )

        # AttentionFP8 is flat-native: takes [T, dim] + per-token positions,
        # no padding / per-request scalars. The runtime kv-cache dtype flag can
        # be false while this branch still constructs AttentionFP8, so dispatch
        # on the module type instead of only the cache config.
        if isinstance(self.attn, AttentionFP8):
            attn_out = self.attn(
                x_pre,  # [T, dim]
                positions,  # [T] int64 absolute positions
                kv_cache=kv_cache,
                block_tables_by_type=block_tables_by_type,
            )  # [T, dim]
        else:
            # Present flat [T, dim] as [B, S, dim] for Attention.  The common
            # prefill layouts (single request, or dense equal-length batch) are
            # already request-major, so a view is enough.  Ragged batches keep
            # the padded scatter/gather fallback.
            cu_long = cu_seqlens.to(torch.long)
            batch_size = int(cu_long.numel() - 1)
            seqlens = cu_long[1:] - cu_long[:-1]  # [B]
            T = int(x_pre.size(0))
            D = int(x_pre.size(-1))
            device = x_pre.device
            max_S = (
                int(seqlens.max().item())
                if batch_size > 0 and seqlens.device.type == "cpu"
                else T
            )
            # Per-row start_pos = absolute position of each request's first
            # token, pulled directly off the flat positions tensor.
            start_pos_per_req = positions[cu_long[:-1]].to(torch.long)  # [B]

            equal_len_dense = (
                T == batch_size * max_S
                and seqlens.device.type == "cpu"
                and bool((seqlens == max_S).all().item())
            )
            dense_layout = batch_size == 1 or (batch_size > 1 and equal_len_dense)
            if dense_layout:
                x_padded = x_pre.view(batch_size, max_S, D)
                b_idx = None
                s_idx = None
            else:
                t_idx = torch.arange(T, device=device, dtype=torch.long)
                # right=True so tokens at cu_seqlens[b] land in request b
                # (cu_long[1:] are exclusive ends).
                b_idx = torch.searchsorted(cu_long[1:].contiguous(), t_idx, right=True)
                s_idx = t_idx - cu_long[:-1][b_idx]
                x_padded = torch.zeros(
                    batch_size, max_S, D, dtype=x_pre.dtype, device=device
                )
                x_padded[b_idx, s_idx] = x_pre

            attn_out_padded = self.attn(
                x_padded,  # [B, max_S, dim]
                start_pos_per_req,
                sequence_lengths=seqlens,
                kv_cache=kv_cache,
                block_tables_by_type=block_tables_by_type,
            )  # [B, max_S, dim]
            if dense_layout:
                attn_out = attn_out_padded.reshape(T, D)
            else:
                attn_out = attn_out_padded[b_idx, s_idx]  # [T, dim]
        if _dbg_layer:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_attn_out", attn_out)
            if dbg_pos_mask is not None:
                _rt.record_if_level(
                    2,
                    f"L{self.layer_id:02d}_attn_out_{dbg_pos_name}",
                    attn_out[dbg_pos_mask].contiguous(),
                )
        x = self.attn_hc.post(attn_out, residual, post, comb)  # [T, hc, dim]
        self._sync_after_first_cp_prefill_attention()
        if _dbg_layer:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_attn_residual", x)
            if dbg_pos_mask is not None:
                _rt.record_if_level(
                    2,
                    f"L{self.layer_id:02d}_attn_residual_{dbg_pos_name}",
                    x[dbg_pos_mask].contiguous(),
                )

        # FFN path
        residual = x
        x_pre, post, comb = self.ffn_hc.pre(
            x,
            dbg_tag=f"L{self.layer_id:02d}_ffn_hc_pre" if _dbg_layer else None,
        )  # [T, dim], ...
        x_pre = self.ffn_norm(x_pre)  # [T, dim]
        if _dbg_layer:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_ffn_in", x_pre)
            if dbg_pos_mask is not None:
                _rt.record_if_level(
                    2,
                    f"L{self.layer_id:02d}_ffn_in_{dbg_pos_name}",
                    x_pre[dbg_pos_mask].contiguous(),
                )
        ffn_in_ids = (
            input_ids
            if input_ids is not None
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )
        if _dbg_layer and dbg_pos_mask is not None:
            setattr(self.ffn, "_dbg_positions", dbg_positions)
        try:
            ffn_out = self.ffn(x_pre, ffn_in_ids)  # [T, dim]
        finally:
            if _dbg_layer and hasattr(self.ffn, "_dbg_positions"):
                setattr(self.ffn, "_dbg_positions", None)
        if _dbg_layer:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_ffn_out", ffn_out)
            if dbg_pos_mask is not None:
                _rt.record_if_level(
                    2,
                    f"L{self.layer_id:02d}_ffn_out_{dbg_pos_name}",
                    ffn_out[dbg_pos_mask].contiguous(),
                )
        x = self.ffn_hc.post(ffn_out, residual, post, comb)  # [T, hc, dim]
        if _dbg_layer and dbg_pos_mask is not None:
            _rt.record_if_level(
                2,
                f"L{self.layer_id:02d}_ffn_residual_{dbg_pos_name}",
                x[dbg_pos_mask].contiguous(),
            )
        return x  # [T, hc, dim]
