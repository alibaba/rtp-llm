"""DeepSeek-V4 Transformer Block: mHC + Attention + MoE.

Mirrors `inference/model.py:Block`. Each call applies hc_pre/F/hc_post
twice — once for Attention and once for MoE FFN.
"""

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.models_py.modules import RMSNorm
from rtp_llm.models_py.modules.dsv4.attention import Attention
from rtp_llm.models_py.modules.dsv4.mhc import hc_split_sinkhorn
from rtp_llm.models_py.modules.dsv4.moe import MoE

# P-sinkhorn (plan_0427.md): single-Triton-kernel replacement for
# mhc.py:hc_split_sinkhorn (~135 launches → 1).  32× per-call speedup
# (1.340 ms → 0.041 ms at B=1 T=16384 HC=4 iters=20), max abs diff
# < 5e-7 vs eager (essentially fp32 round-off).  Default ON; gate via
# ``DSV4_SINKHORN_FUSED=0`` to disable.
try:
    from rtp_llm.models_py.modules.dsv4._sinkhorn_triton import fused_hc_split_sinkhorn

    _SINKHORN_FUSED_OK = True
except Exception:  # pragma: no cover
    fused_hc_split_sinkhorn = None
    _SINKHORN_FUSED_OK = False


def _use_fused_sinkhorn(mixes: torch.Tensor, hc_mult: int) -> bool:
    if not _SINKHORN_FUSED_OK or fused_hc_split_sinkhorn is None:
        return False
    if os.environ.get("DSV4_SINKHORN_FUSED", "1") == "0":
        return False
    if hc_mult != 4:
        return False
    if mixes.dtype != torch.float32:
        return False
    if mixes.numel() == 0:
        return False
    return True


# Vendored TileKernels (DeepSeek) fused mHC entry points.  See
# rtp_llm/models_py/3rdparty/tile_kernels and the _mhc_tilelang adapter.
# Each TK call returns None to signal "fall back to REF" (env-disabled,
# wrong dtype/mult, autograd, or per-op JIT-fail sticky verdict).
from rtp_llm.models_py.modules.dsv4._mhc_tilelang import tk_mhc_post as _tk_mhc_post
from rtp_llm.models_py.modules.dsv4._mhc_tilelang import tk_mhc_pre as _tk_mhc_pre


def _maybe_squeeze_hc_1d(t: torch.Tensor) -> torch.Tensor:
    """Reverse ``weight_module.py:357``'s "1D scale → 2D unsqueeze" heuristic.

    The framework auto-promotes any 1D float scale tensor to ``[N, 1]`` so
    per-row UE8M0 quant scales broadcast cleanly. mHC scales (hc_*_scale,
    hc_head_scale) are also 1D but are NOT quant scales — they're plain
    learnable factors. Squeeze trailing 1-dim back to keep the forward
    shape contract with the legacy load path."""
    if t.dim() == 2 and t.shape[-1] == 1:
        return t.squeeze(-1)
    return t


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
        kv_cache_dtype: Any = None,
    ):
        """``layer_weights`` is the framework's per-layer dict
        (``ModelWeights.weights[layer_id]``) keyed by ``W.v4_*`` enum.
        Block reads ``W.v4_attn_norm`` / ``W.v4_ffn_norm`` / ``W.v4_hc_*``
        and forwards the dict to ``Attention`` and ``MoE``."""
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters

        self.attn = Attention(
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
            kv_cache_dtype=kv_cache_dtype,
        )
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
        )
        # Framework loader already casts norms to bf16 (compute_dtype) and
        # hc_* tensors to fp32 (descriptor data_type); pass refs straight
        # into ``RMSNorm`` at construction time.  Norms here see 2D inputs
        # ``[T, dim]`` from the hc_pre output, so framework ``RMSNorm``
        # (which expects 2D) drops in directly.
        from rtp_llm.utils.model_weight import W

        self.attn_norm = RMSNorm(layer_weights[W.v4_attn_norm], norm_eps)
        self.ffn_norm = RMSNorm(layer_weights[W.v4_ffn_norm], norm_eps)

        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * dim
        self.hc_attn_fn = layer_weights[W.v4_hc_attn_fn]
        self.hc_ffn_fn = layer_weights[W.v4_hc_ffn_fn]
        self.hc_attn_base = layer_weights[W.v4_hc_attn_base]
        self.hc_ffn_base = layer_weights[W.v4_hc_ffn_base]
        # Scales are 1D mHC factors; the framework's
        # ``weight_module.py:357`` "1D scale → 2D unsqueeze" heuristic
        # (intended for per-row UE8M0 quant scales) over-promotes them
        # to ``[N, 1]`` — squeeze them back to 1D so the forward sees
        # the same shape as the legacy load path.
        self.hc_attn_scale = _maybe_squeeze_hc_1d(layer_weights[W.v4_hc_attn_scale])
        self.hc_ffn_scale = _maybe_squeeze_hc_1d(layer_weights[W.v4_hc_ffn_scale])

    def _hc_fn_bf16(self, hc_fn: torch.Tensor) -> torch.Tensor:
        """Lazy-cached bf16 view of an FP32 hc_fn parameter.

        The FP32 → BF16 cast is ~30 µs on a [24, 16384] tensor; we cache by
        ``id(hc_fn)`` so attn vs ffn fns get separate slots.  Memory: ~768 KB
        per fn × 2 fns / layer × 43 layers ≈ 64 MB total — acceptable for
        cutting per-call cast latency to zero.
        """
        cache = getattr(self, "_bf16_cache", None)
        if cache is None:
            cache = {}
            self._bf16_cache = cache
        key = id(hc_fn)
        bf = cache.get(key)
        if bf is None or bf.shape != hc_fn.shape or bf.device != hc_fn.device:
            bf = hc_fn.to(torch.bfloat16)
            cache[key] = bf
        return bf

    def _hc_linear_mixes(
        self, x_flat: torch.Tensor, hc_fn_bf16: torch.Tensor, rsqrt: torch.Tensor
    ) -> torch.Tensor:
        positions = getattr(self, "_hc_positions", None)
        if (
            x_flat.dim() != 2
            or positions is None
            or positions.numel() != x_flat.shape[0]
        ):
            return (F.linear(x_flat, hc_fn_bf16) * rsqrt).float()

        pos = positions.to(device=x_flat.device, dtype=torch.long)
        block = pos // 256
        T = int(x_flat.shape[0])
        out = torch.empty(
            T, hc_fn_bf16.shape[0], dtype=x_flat.dtype, device=x_flat.device
        )
        start = 0
        while start < T:
            cur = block[start]
            end = start + 1
            while end < T and bool((block[end] == cur).item()):
                end += 1
            out[start:end] = F.linear(x_flat[start:end], hc_fn_bf16)
            start = end
        return (out * rsqrt).float()

    def _hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        dbg_tag: Optional[str] = None,
    ):
        """Accepts ``[B, S, hc, d]`` or the flat ``[T, hc, d]`` layout.
        Returns ``y``, ``post [..., hc, 1]``, ``comb [..., hc, hc]`` with
        leading shape matching the input.  REF path uses ``dim=-2`` /
        ``flatten(-2)`` so 3D and 4D share code; TK is 4D-only, so 3D
        input is wrapped with unsqueeze/squeeze around the TK call."""
        # Vendored TK fast path. Falls back to the REF rewrite below on
        # JIT-fail / unsupported shape (sticky per-op).
        tk_x = x.unsqueeze(0) if x.dim() == 3 else x
        tk_out = _tk_mhc_pre(
            tk_x,
            hc_fn,
            hc_scale,
            hc_base,
            norm_eps=self.norm_eps,
            pre_eps=self.hc_eps,
            sinkhorn_eps=self.hc_eps,
            sinkhorn_iters=self.hc_sinkhorn_iters,
            hc_mult=self.hc_mult,
        )
        if tk_out is not None:
            self._dbg_record_hc_pre_path(dbg_tag, x, "tk", use_fused=None)
            if x.dim() == 3:
                y_tk, post_tk, comb_tk = tk_out
                return y_tk.squeeze(0), post_tk.squeeze(0), comb_tk.squeeze(0)
            return tk_out
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(-2)  # [..., hc*d] bf16
        # FP32 squared mean → rsqrt, single cast.  Kept in fp32 because the
        # accumulation magnitude here is what determines numeric stability of
        # the layer norm; a bf16 reduction on hc*d=16384 would lose ~7 bits.
        x_flat_f32 = x_flat.float()
        rsqrt = torch.rsqrt(
            x_flat_f32.square().mean(-1, keepdim=True) + self.norm_eps
        ).to(dtype)
        mixes = self._hc_linear_mixes(x_flat, self._hc_fn_bf16(hc_fn), rsqrt)
        mixes = mixes.contiguous()
        use_fused = _use_fused_sinkhorn(mixes, self.hc_mult)
        self._dbg_record_hc_pre_path(dbg_tag, x, "fallback", use_fused=use_fused)
        if use_fused:
            pre, post, comb = fused_hc_split_sinkhorn(
                mixes,
                hc_scale.contiguous(),
                hc_base.contiguous(),
                hc_mult=self.hc_mult,
                sinkhorn_iters=self.hc_sinkhorn_iters,
                eps=self.hc_eps,
            )
        else:
            pre, post, comb = hc_split_sinkhorn(
                mixes,
                hc_scale,
                hc_base,
                hc_mult=self.hc_mult,
                sinkhorn_iters=self.hc_sinkhorn_iters,
                eps=self.hc_eps,
            )
        y = torch.sum(pre.to(dtype).unsqueeze(-1) * x.view(*shape), dim=-2)
        # post in the [..., hc, 1] convention so _hc_post can call TK mhc_post.
        return y.to(dtype), post.unsqueeze(-1), comb

    def _dbg_record_hc_pre_path(
        self,
        dbg_tag: Optional[str],
        x: torch.Tensor,
        path: str,
        use_fused: Optional[bool],
    ) -> None:
        if dbg_tag is None:
            return
        from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt

        if not _rt.ENABLED:
            return
        path_code = 1.0 if path == "tk" else 0.0
        fused_code = -1.0 if use_fused is None else float(use_fused)
        _rt.record_if_level(
            2,
            f"{dbg_tag}_path_codes",
            torch.tensor([path_code, fused_code], device=x.device, dtype=torch.float32),
        )
        try:
            import torch.distributed as dist

            rank = dist.get_rank() if dist.is_initialized() else 0
        except Exception:
            rank = 0
        print(
            "MOEDBG_HC_PRE_PATH "
            f"rank={rank} layer={self.layer_id} tag={dbg_tag} "
            f"path={path} use_fused_sinkhorn={use_fused}",
            flush=True,
        )

    def _hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        """Combine sublayer output ``x`` with per-stream residual via post/comb.

        Three paths, fastest-first:
          1. TileKernels ``mhc_post`` — single fused kernel, ~35× the REF.
          2. BF16 BMM rewrite — ``comb.T @ residual`` runs as a tensor-core BMM.
          3. Original 5D broadcast+sum — last resort for non-bf16 / odd shapes.

        ``post`` arrives shaped ``[..., hc, 1]`` (``_hc_pre`` unsqueezes).
        Path 1 consumes that shape directly; paths 2/3 squeeze back to ``[..., hc]``.
        Accepts either ``[B, T, d]`` or flat ``[T, d]`` layout; TK is 4D-only
        and wraps 3D inputs with unsqueeze/squeeze.  The slow fallback reduces
        along ``dim=-3`` so both shapes share code.
        """
        if post.dim() == residual.dim() and x.dim() != 2:
            tk_x = x.unsqueeze(0) if x.dim() == 2 else x
            tk_res = residual.unsqueeze(0) if residual.dim() == 3 else residual
            tk_post = post.unsqueeze(0) if post.dim() == 3 else post
            tk_comb = comb.unsqueeze(0) if comb.dim() == 3 else comb
            tk_out = _tk_mhc_post(tk_x, tk_res, tk_post, tk_comb, hc_mult=self.hc_mult)
            if tk_out is not None:
                return tk_out.squeeze(0) if x.dim() == 2 else tk_out

        if post.dim() == residual.dim():
            post_b = post.squeeze(-1)
        else:
            post_b = post

        if x.dtype == torch.bfloat16 and residual.dtype == torch.bfloat16:
            post_bf16 = post_b.to(x.dtype)
            first = post_bf16.unsqueeze(-1) * x.unsqueeze(-2)
            comb_bf16 = comb.to(x.dtype)
            second = torch.sum(comb_bf16.unsqueeze(-1) * residual.unsqueeze(-2), dim=-3)
            return (first + second).to(x.dtype)

        y = post_b.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=-3
        )
        return y.type_as(x)

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
        x_pre, post, comb = self._hc_pre(
            x,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
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
        x = self._hc_post(attn_out, residual, post, comb)
        if _dbg_layer:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_decode_attn_residual", x)

        # FFN path — MoE has no per-step state, reuse existing forward
        residual = x
        x_pre, post, comb = self._hc_pre(
            x,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            dbg_tag=f"L{self.layer_id:02d}_decode_ffn_hc_pre" if _dbg_layer else None,
        )
        bsz, q_len, dim_ = x_pre.shape
        x_pre = self.ffn_norm(x_pre.reshape(bsz * q_len, dim_)).view(bsz, q_len, dim_)
        if _dbg_layer:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_decode_ffn_in", x_pre)
        ffn_out = self.ffn(x_pre, input_ids)
        if _dbg_layer:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_decode_ffn_out", ffn_out)
        x = self._hc_post(ffn_out, residual, post, comb)
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
        DSV4 layer layout.  ``_hc_pre`` / ``_hc_post`` are flat-native;
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

        self._hc_positions = positions
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
        x_pre, post, comb = self._hc_pre(
            x,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
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

        # Scatter flat [T, dim] into padded [B, max_S, dim] so that
        # Attention._forward_body's [B, S, dim] body processes every
        # request in a single layer call.
        cu_long = cu_seqlens.to(torch.long)
        batch_size = int(cu_long.numel() - 1)
        seqlens = cu_long[1:] - cu_long[:-1]  # [B]
        T = int(x_pre.size(0))
        D = int(x_pre.size(-1))
        device = x_pre.device
        max_S = int(seqlens.max().item()) if batch_size > 0 else 0
        t_idx = torch.arange(T, device=device, dtype=torch.long)
        # right=True so tokens at cu_seqlens[b] land in request b
        # (cu_long[1:] are exclusive ends).
        b_idx = torch.searchsorted(cu_long[1:].contiguous(), t_idx, right=True)
        s_idx = t_idx - cu_long[:-1][b_idx]
        x_padded = torch.zeros(batch_size, max_S, D, dtype=x_pre.dtype, device=device)
        x_padded[b_idx, s_idx] = x_pre
        # Per-row start_pos = absolute position of each request's first
        # token, pulled directly off the flat positions tensor.
        start_pos_per_req = positions[cu_long[:-1]].to(torch.long)  # [B]

        attn_out_padded = self.attn(
            x_padded,  # [B, max_S, dim]
            start_pos_per_req,
            sequence_lengths=seqlens,
            kv_cache=kv_cache,
            block_tables_by_type=block_tables_by_type,
        )  # [B, max_S, dim]
        attn_out = attn_out_padded[b_idx, s_idx]  # [T, dim]
        if _dbg_layer:
            _rt.record_if_level(2, f"L{self.layer_id:02d}_attn_out", attn_out)
            if dbg_pos_mask is not None:
                _rt.record_if_level(
                    2,
                    f"L{self.layer_id:02d}_attn_out_{dbg_pos_name}",
                    attn_out[dbg_pos_mask].contiguous(),
                )
        x = self._hc_post(attn_out, residual, post, comb)  # [T, hc, dim]
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
        x_pre, post, comb = self._hc_pre(
            x,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
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
        x = self._hc_post(ffn_out, residual, post, comb)  # [T, hc, dim]
        if _dbg_layer and dbg_pos_mask is not None:
            _rt.record_if_level(
                2,
                f"L{self.layer_id:02d}_ffn_residual_{dbg_pos_name}",
                x[dbg_pos_mask].contiguous(),
            )
        self._hc_positions = None
        return x  # [T, hc, dim]


class MTPBlock(Block):
    """Multi-Token Prediction draft block — V4 speculative decode head.

    Given the last-layer hidden ``x [B, S, hc, dim]`` (from the main model)
    and the NEXT-step ``input_ids`` (shifted one position), this block
    fuses the shifted embed with the hidden state via ``e_proj + h_proj +
    enorm/hnorm``, runs the fused tensor through the standard V4 Block
    (attention + MoE-FFN + mHC), then produces per-position logits via
    its own ``hc_head_*`` reduce and the shared LM head.

    Ckpt keys live under ``mtp.{i}.*`` and mirror the regular block keys
    plus:
      - ``e_proj.{weight,scale}`` — FP8 e4m3fn + UE8M0 block-128 scale
      - ``h_proj.{weight,scale}`` — same format
      - ``enorm.weight``, ``hnorm.weight``, ``norm.weight`` — BF16 (cast
        to FP32 on load, matching regular block norm params)
      - ``hc_head_{fn,base,scale}`` — FP32

    Mirrors ``inference/model.py:MTPBlock``; the speculative-decoding
    driver (prefill + draft-step sampler) is a framework-side concern and
    lives outside this class.
    """

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
        weights: Optional[Dict[str, torch.Tensor]] = None,
        prefix: str = "",
        tp_size: int = 1,
        tp_rank: int = 0,
        ep_size: int = 1,
        ep_rank: int = 0,
        max_tokens_per_rank: int = 8192,
        kv_cache_dtype: Any = None,
    ):
        super().__init__(
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
            moe_inter_dim=moe_inter_dim,
            n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts,
            n_shared_experts=n_shared_experts,
            score_func=score_func,
            route_scale=route_scale,
            swiglu_limit=swiglu_limit,
            n_hash_layers=n_hash_layers,
            vocab_size=vocab_size,
            hc_mult=hc_mult,
            hc_sinkhorn_iters=hc_sinkhorn_iters,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
            # MTP descriptor weight tags not declared yet; pass None so
            # Block stays in the legacy unit-test path until W.v4_mtp_*
            # lands.  See note below.
            layer_weights=None,
            tp_size=tp_size,
            tp_rank=tp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            max_tokens_per_rank=max_tokens_per_rank,
            kv_cache_dtype=kv_cache_dtype,
        )
        from rtp_llm.models_py.modules.dsv4.qlinear import QuantizedLinear

        self.dim = dim
        # MTP weight tags (W.v4_mtp_*) not declared in the descriptor yet;
        # MTPBlock still consumes the legacy ckpt-key ``weights`` dict + the
        # ``mtp.{i}.*`` prefix.  Once the descriptor lands, migrate the body
        # to ``layer_weights[W.v4_mtp_*]`` and drop ``weights/prefix``.
        # ``transformer.py`` asserts ``n_mtp_layers == 0`` so this branch is
        # never hit in production.
        _mtp_factory = weights is not None
        if _mtp_factory:
            from rtp_llm.models_py.modules.dsv4.attention import (
                _v4_fp8_linear_from_dict,
            )

            self.e_proj = _v4_fp8_linear_from_dict(
                weights,
                f"{prefix}.e_proj.weight",
                f"{prefix}.e_proj.scale",
            )
            self.h_proj = _v4_fp8_linear_from_dict(
                weights,
                f"{prefix}.h_proj.weight",
                f"{prefix}.h_proj.scale",
            )
            self._h_proj_is_factory = True
        else:
            self.e_proj = QuantizedLinear(dim, dim, storage="fp8")
            self.h_proj = QuantizedLinear(dim, dim, storage="fp8")
            self._h_proj_is_factory = False

        # MTPBlock norms see N-D inputs ([B, S, hc, dim] for hnorm, etc.);
        # framework ``RMSNorm`` expects 2D, so callers must reshape.
        # MTPBlock itself is not instantiated yet (n_mtp_layers==0 assert
        # in transformer.py); this construction is only reached once
        # W.v4_mtp_* tags land and the body is migrated.
        assert _mtp_factory, "MTPBlock requires weights; unit-test None path retired"
        self.enorm = RMSNorm(weights[f"{prefix}.enorm.weight"], norm_eps)
        self.hnorm = RMSNorm(weights[f"{prefix}.hnorm.weight"], norm_eps)
        self.norm = RMSNorm(weights[f"{prefix}.norm.weight"], norm_eps)

        hc_dim = hc_mult * dim
        if _mtp_factory:
            self.hc_head_fn = weights[f"{prefix}.hc_head_fn"]
            self.hc_head_base = weights[f"{prefix}.hc_head_base"]
            self.hc_head_scale = weights[f"{prefix}.hc_head_scale"]
        else:
            # Unit-test path — caller binds tensors externally.
            self.hc_head_fn = None
            self.hc_head_base = None
            self.hc_head_scale = None

    def _apply_proj(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Factory FP8 linears want 2D input; QuantizedLinear accepts N-D."""
        if self._h_proj_is_factory and x.dim() > 2:
            shape = x.shape
            return layer(x.reshape(-1, shape[-1])).view(*shape[:-1], -1)
        return layer(x)

    def forward_draft(
        self,
        x: torch.Tensor,  # [B, S, hc, dim]
        start_pos: int,
        input_ids: torch.Tensor,  # [B, S] shifted one step
        embed: nn.Module,  # shared V4Transformer.embed
        lm_head_weight: torch.Tensor,  # shared [vocab, dim] FP32
    ) -> torch.Tensor:
        """Draft-token logits. See class docstring.

        MTP draft is B==1-only today — the super().forward() call is bridged
        to the flat Block.forward signature by squeezing the unit-B axis off
        the fused tensor, synthesising positions/cu_seqlens, then restoring
        the [B, S, hc, dim] shape for the mHC head reduce.
        """
        e = embed(input_ids)  # [B, S, dim]
        e = self.enorm(e)
        x_norm = self.hnorm(x)  # [B, S, hc, dim] (norm over last dim)
        e_proj_out = self._apply_proj(self.e_proj, e).unsqueeze(2)  # [B, S, 1, dim]
        h_proj_out = self._apply_proj(self.h_proj, x_norm)  # [B, S, hc, dim]
        x_fused = e_proj_out + h_proj_out  # [B, S, hc, dim]

        # Bridge to flat Block.forward: drop the unit-B axis, synthesise
        # per-token positions and cu_seqlens, then reconstruct [B, S, ...]
        # for the hc_head_reduce that follows.
        B, S, hc, dim = x_fused.shape
        assert B == 1, f"MTPBlock.forward_draft expects B==1; got B={B}"
        x_flat = x_fused.squeeze(0)  # [S, hc, dim]
        input_ids_flat = input_ids.squeeze(0)  # [S]
        positions = start_pos + torch.arange(
            S, device=x_fused.device, dtype=torch.int64
        )  # [S]
        cu_seqlens = torch.tensor(
            [0, S], dtype=torch.int64, device=x_fused.device
        )  # [2]
        x_after_flat = super().forward(
            x_flat,
            input_ids_flat,
            positions,
            cu_seqlens,
        )  # [S, hc, dim]
        x_after = x_after_flat.unsqueeze(0)  # [B=1, S, hc, dim]

        # hc_head_reduce (same math as V4Transformer._hc_head_reduce)
        shape, dtype = x_after.size(), x_after.dtype
        x_flat = x_after.flatten(2).float()  # [B, S, hc*dim]
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_flat, self.hc_head_fn) * rsqrt
        pre = (
            torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        )
        h = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=2)  # [B, S, dim]
        h = self.norm(h.to(dtype))  # [B, S, dim]
        return F.linear(h.float(), lm_head_weight)  # [B, S, vocab]
