"""Immutable TP1 DSV4 HCA megakernel weight layouts (Pro & Flash geometries)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch

from .mega_csa_weights import (
    HC,
    HC_MIX,
    HEAD_DIM,
    MAX_BATCH,
    PRO_GEOMETRY,
    CSAGeometry,
    _cat_rows,
    _normalize_weight_scale,
    _require_dtype,
    _require_shape,
)

HCA_COMPRESS_RATIO = 128
HCA_APE_ROWS = HCA_COMPRESS_RATIO
# HCA_STATE pool rows interleave ``kv(512) | gate(512)`` fp32.
HCA_STATE_WIDTH = HEAD_DIM


@dataclass(frozen=True)
class MegaHCAWeights:
    """Fused extension layouts built once from checkpoint tensors."""

    front_fp8: torch.Tensor
    front_sf: torch.Tensor
    front_bf16: torch.Tensor
    wq_b_fp8: torch.Tensor
    wq_b_sf: torch.Tensor
    q_norm: torch.Tensor
    window_norm: torch.Tensor
    compressor_norm: torch.Tensor
    compressor_ape: torch.Tensor
    hc_fn: torch.Tensor
    hc_base: torch.Tensor
    hc_scale: torch.Tensor
    attn_norm: torch.Tensor
    geometry: CSAGeometry = field(default=PRO_GEOMETRY)

    @classmethod
    def from_layer_weights(
        cls,
        layer_weights: Dict[str, torch.Tensor],
        geometry: CSAGeometry = PRO_GEOMETRY,
    ) -> "MegaHCAWeights":
        """Pack one HCA layer without changing checkpoint FP8 bits or scales."""
        from rtp_llm.utils.model_weight import W

        def get(tag: str) -> torch.Tensor:
            try:
                return layer_weights[tag]
            except KeyError as exc:
                raise KeyError(f"missing DSV4 mega weight {tag!r}") from exc

        return cls._pack(
            geometry,
            wq_a=get(W.v4_attn_wq_a_w),
            wkv=get(W.v4_attn_wkv_w),
            wq_b=get(W.v4_attn_wq_b_w),
            wq_a_scale=get(W.v4_attn_wq_a_s),
            wkv_scale=get(W.v4_attn_wkv_s),
            wq_b_scale=get(W.v4_attn_wq_b_s),
            comp_wkv=get(W.v4_compressor_wkv),
            comp_wgate=get(W.v4_compressor_wgate),
            q_norm=get(W.v4_attn_q_norm),
            kv_norm=get(W.v4_attn_kv_norm),
            comp_norm=get(W.v4_compressor_norm),
            ape=get(W.v4_compressor_ape),
            hc_fn=get(W.v4_hc_attn_fn),
            hc_base=get(W.v4_hc_attn_base),
            hc_scale=get(W.v4_hc_attn_scale),
            attn_norm=get(W.v4_attn_norm),
        )

    @classmethod
    def from_owned_block(cls, block, geometry: CSAGeometry) -> "MegaHCAWeights":
        """Pack the post-load module parameters; never access a loader weight map."""
        attn = block.attn
        comp = attn.compressor
        return cls._pack(
            geometry,
            wq_a=attn.wq_a.weight,
            wkv=attn.wkv.weight,
            wq_b=attn.wq_b.weight,
            wq_a_scale=attn.wq_a.scale,
            wkv_scale=attn.wkv.scale,
            wq_b_scale=attn.wq_b.scale,
            comp_wkv=comp.wkv.weight,
            comp_wgate=comp.wgate.weight,
            q_norm=attn.q_norm,
            kv_norm=attn.kv_norm,
            comp_norm=comp.norm.weight,
            ape=comp.ape,
            hc_fn=block.attn_hc.fn,
            hc_base=block.attn_hc.base,
            hc_scale=block.attn_hc.scale,
            attn_norm=block.attn_norm.weight,
        )

    @classmethod
    def _pack(
        cls,
        g: CSAGeometry,
        *,
        wq_a: torch.Tensor,
        wkv: torch.Tensor,
        wq_b: torch.Tensor,
        wq_a_scale: torch.Tensor,
        wkv_scale: torch.Tensor,
        wq_b_scale: torch.Tensor,
        comp_wkv: torch.Tensor,
        comp_wgate: torch.Tensor,
        q_norm: torch.Tensor,
        kv_norm: torch.Tensor,
        comp_norm: torch.Tensor,
        ape: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_base: torch.Tensor,
        hc_scale: torch.Tensor,
        attn_norm: torch.Tensor,
    ) -> "MegaHCAWeights":
        for name, tensor, shape in (
            ("wq_a", wq_a, (g.q_lora_rank, g.dim)),
            ("wkv", wkv, (512, g.dim)),
            ("wq_b", wq_b, (g.n_main, g.q_lora_rank)),
        ):
            _require_shape(name, tensor, shape)
            _require_dtype(name, tensor, (torch.float8_e4m3fn,))

        wq_a_sf = _normalize_weight_scale("wq_a_sf", wq_a_scale, tuple(wq_a.shape))
        wkv_sf = _normalize_weight_scale("wkv_sf", wkv_scale, tuple(wkv.shape))
        wq_b_sf = _normalize_weight_scale("wq_b_sf", wq_b_scale, tuple(wq_b.shape))

        for name, tensor in (
            ("hca_compressor_wkv", comp_wkv),
            ("hca_compressor_wgate", comp_wgate),
        ):
            _require_shape(name, tensor, (HCA_STATE_WIDTH, g.dim))
            _require_dtype(name, tensor, (torch.bfloat16,))

        def fp32(tag: str, value: torch.Tensor, shape) -> torch.Tensor:
            tensor = value.float().contiguous()
            _require_shape(tag, tensor, shape)
            return tensor

        def fp32_flat(tag: str, value: torch.Tensor, elements: int) -> torch.Tensor:
            tensor = value.float().reshape(-1).contiguous()
            _require_shape(tag, tensor, (elements,))
            return tensor

        attn_norm = attn_norm.contiguous()
        _require_shape("attn_norm", attn_norm, (g.dim,))
        _require_dtype("attn_norm", attn_norm, (torch.bfloat16,))

        return cls(
            front_fp8=_cat_rows("front_fp8", (wq_a, wkv), (g.front_fp8_rows, g.dim)),
            front_sf=_cat_rows(
                "front_sf", (wq_a_sf, wkv_sf), (g.front_fp8_rows // 128, g.sf_k)
            ),
            front_bf16=_cat_rows(
                "front_bf16", (comp_wkv, comp_wgate), (2 * HCA_STATE_WIDTH, g.dim)
            ),
            wq_b_fp8=wq_b.contiguous(),
            wq_b_sf=wq_b_sf.contiguous(),
            q_norm=fp32("q_norm", q_norm, (g.q_lora_rank,)),
            window_norm=fp32("kv_norm", kv_norm, (HEAD_DIM,)),
            compressor_norm=fp32("comp_norm", comp_norm, (HEAD_DIM,)),
            compressor_ape=fp32("ape", ape, (HCA_APE_ROWS, HCA_STATE_WIDTH)),
            hc_fn=fp32("hc_fn", hc_fn, (HC_MIX, HC * g.dim)),
            hc_base=fp32("hc_base", hc_base, (HC_MIX,)),
            hc_scale=fp32_flat("hc_scale", hc_scale, 3),
            attn_norm=attn_norm,
            geometry=g,
        )


__all__ = [
    "HCA_APE_ROWS",
    "HCA_COMPRESS_RATIO",
    "HCA_STATE_WIDTH",
    "MAX_BATCH",
    "MegaHCAWeights",
]
