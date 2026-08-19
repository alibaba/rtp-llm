"""Immutable TP1 DSV4 HCA megakernel weight layouts (Pro & Flash geometries)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch

from .mega_csa_weights import (
    DIM,
    HC,
    HC_MIX,
    HEAD_DIM,
    MAX_BATCH,
    PRO_GEOMETRY,
    Q_LORA_RANK,
    CSAGeometry,
    _cat_rows,
    _require_dtype,
    _require_shape,
)

HCA_COMPRESS_RATIO = 128
# opA publishes ``[wq_a(1536) | window_kv(512)]`` columns; opB reads the first
# ``Q_LORA_RANK`` for the projection and the final ``HEAD_DIM`` for SWA writes.
HCA_FRONT_OUT_DIM = Q_LORA_RANK + HEAD_DIM
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

        g = geometry

        def get(tag: str) -> torch.Tensor:
            try:
                return layer_weights[tag]
            except KeyError as exc:
                raise KeyError(f"missing DSV4 mega weight {tag!r}") from exc

        wq_a = get(W.v4_attn_wq_a_w)
        wkv = get(W.v4_attn_wkv_w)
        wq_b = get(W.v4_attn_wq_b_w)
        for name, tensor, shape in (
            ("wq_a", wq_a, (g.q_lora_rank, g.dim)),
            ("wkv", wkv, (512, g.dim)),
            ("wq_b", wq_b, (g.n_main, g.q_lora_rank)),
        ):
            _require_shape(name, tensor, shape)
            _require_dtype(name, tensor, (torch.float8_e4m3fn,))

        wq_a_sf = get(W.v4_attn_wq_a_s)
        wkv_sf = get(W.v4_attn_wkv_s)
        wq_b_sf = get(W.v4_attn_wq_b_s)
        scale_dtypes = (torch.float8_e8m0fnu, torch.uint8)
        for name, tensor, shape in (
            ("wq_a_sf", wq_a_sf, (g.sf_q, g.sf_k)),
            ("wkv_sf", wkv_sf, (4, g.sf_k)),
            ("wq_b_sf", wq_b_sf, (g.n_main // 128, g.sf_q)),
        ):
            _require_shape(name, tensor, shape)
            _require_dtype(name, tensor, scale_dtypes)

        comp_wkv = get(W.v4_compressor_wkv)
        comp_wgate = get(W.v4_compressor_wgate)
        for name, tensor in (
            ("hca_compressor_wkv", comp_wkv),
            ("hca_compressor_wgate", comp_wgate),
        ):
            _require_shape(name, tensor, (HCA_STATE_WIDTH, g.dim))
            _require_dtype(name, tensor, (torch.bfloat16,))

        def fp32(tag: str, shape) -> torch.Tensor:
            tensor = get(tag).float().contiguous()
            _require_shape(tag, tensor, shape)
            return tensor

        def fp32_flat(tag: str, elements: int) -> torch.Tensor:
            tensor = get(tag).float().reshape(-1).contiguous()
            _require_shape(tag, tensor, (elements,))
            return tensor

        attn_norm = get(W.v4_attn_norm).contiguous()
        _require_shape(W.v4_attn_norm, attn_norm, (g.dim,))
        _require_dtype(W.v4_attn_norm, attn_norm, (torch.bfloat16,))

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
            q_norm=fp32(W.v4_attn_q_norm, (g.q_lora_rank,)),
            window_norm=fp32(W.v4_attn_kv_norm, (HEAD_DIM,)),
            compressor_norm=fp32(W.v4_compressor_norm, (HEAD_DIM,)),
            compressor_ape=fp32(W.v4_compressor_ape, (HCA_APE_ROWS, HCA_STATE_WIDTH)),
            hc_fn=fp32(W.v4_hc_attn_fn, (HC_MIX, HC * g.dim)),
            hc_base=fp32(W.v4_hc_attn_base, (HC_MIX,)),
            hc_scale=fp32_flat(W.v4_hc_attn_scale, 3),
            attn_norm=attn_norm,
            geometry=g,
        )


__all__ = [
    "HCA_APE_ROWS",
    "HCA_COMPRESS_RATIO",
    "HCA_FRONT_OUT_DIM",
    "HCA_STATE_WIDTH",
    "MAX_BATCH",
    "MegaHCAWeights",
]
