"""Immutable TP1 DSV4-Pro CSA megakernel weight layouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

DIM = 7168
HC = 4
HC_MIX = 24
Q_LORA_RANK = 1536
HEAD_DIM = 512
ROPE_DIM = 64
MAIN_HEADS = 128
O_GROUPS = 16
O_LORA_RANK = 1024
INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
COMPRESS_RATIO = 4
MAX_BATCH = 128
FRONT_OUT_DIM = 4672
MQA_SPLIT_KV = 256


def _require_shape(name: str, tensor: torch.Tensor, shape: Tuple[int, ...]) -> None:
    actual = tuple(int(value) for value in tensor.shape)
    if actual != shape:
        raise ValueError(f"{name} must have shape {shape}, got {actual}")


def _require_dtype(
    name: str, tensor: torch.Tensor, allowed: Tuple[torch.dtype, ...]
) -> None:
    if tensor.dtype not in allowed:
        choices = ", ".join(str(dtype) for dtype in allowed)
        raise TypeError(f"{name} must have dtype in ({choices}), got {tensor.dtype}")


def _cat_rows(
    name: str, tensors: Tuple[torch.Tensor, ...], shape: Tuple[int, int]
) -> torch.Tensor:
    result = torch.cat(tensors, dim=0).contiguous()
    _require_shape(name, result, shape)
    return result


@dataclass(frozen=True)
class MegaCSAWeights:
    """Fused extension layouts built once from checkpoint tensors."""

    front_fp8: torch.Tensor
    front_sf: torch.Tensor
    front_bf16: torch.Tensor
    wq_b_fp8: torch.Tensor
    wq_b_sf: torch.Tensor
    q_norm: torch.Tensor
    window_norm: torch.Tensor
    indexer_norm: torch.Tensor
    main_compressor_norm: torch.Tensor
    main_ape: torch.Tensor
    indexer_ape: torch.Tensor
    hc_fn: torch.Tensor
    hc_base: torch.Tensor
    hc_scale: torch.Tensor
    attn_norm: torch.Tensor

    @classmethod
    def from_layer_weights(
        cls, layer_weights: Dict[str, torch.Tensor]
    ) -> "MegaCSAWeights":
        """Pack one CSA layer without changing checkpoint FP8 bits or scales."""
        from rtp_llm.utils.model_weight import W

        def get(tag: str) -> torch.Tensor:
            try:
                return layer_weights[tag]
            except KeyError as exc:
                raise KeyError(f"missing DSV4 mega weight {tag!r}") from exc

        wq_a = get(W.v4_attn_wq_a_w)
        wkv = get(W.v4_attn_wkv_w)
        main_wq_b = get(W.v4_attn_wq_b_w)
        index_wq_b = get(W.v4_indexer_wq_b_w)
        for name, tensor, shape in (
            ("wq_a", wq_a, (1536, DIM)),
            ("wkv", wkv, (512, DIM)),
            ("main_wq_b", main_wq_b, (65536, Q_LORA_RANK)),
            ("index_wq_b", index_wq_b, (8192, Q_LORA_RANK)),
        ):
            _require_shape(name, tensor, shape)
            _require_dtype(name, tensor, (torch.float8_e4m3fn,))

        wq_a_sf = get(W.v4_attn_wq_a_s)
        wkv_sf = get(W.v4_attn_wkv_s)
        main_wq_b_sf = get(W.v4_attn_wq_b_s)
        index_wq_b_sf = get(W.v4_indexer_wq_b_s)
        scale_dtypes = (torch.float8_e8m0fnu, torch.uint8)
        for name, tensor, shape in (
            ("wq_a_sf", wq_a_sf, (12, 56)),
            ("wkv_sf", wkv_sf, (4, 56)),
            ("main_wq_b_sf", main_wq_b_sf, (512, 12)),
            ("index_wq_b_sf", index_wq_b_sf, (64, 12)),
        ):
            _require_shape(name, tensor, shape)
            _require_dtype(name, tensor, scale_dtypes)

        main_wkv = get(W.v4_compressor_wkv)
        main_wgate = get(W.v4_compressor_wgate)
        index_wkv = get(W.v4_indexer_compressor_wkv)
        index_wgate = get(W.v4_indexer_compressor_wgate)
        weights_proj = get(W.v4_indexer_weights_proj_w)
        for name, tensor, shape in (
            ("main_compressor_wkv", main_wkv, (1024, DIM)),
            ("main_compressor_wgate", main_wgate, (1024, DIM)),
            ("indexer_compressor_wkv", index_wkv, (256, DIM)),
            ("indexer_compressor_wgate", index_wgate, (256, DIM)),
            ("indexer_weights_proj", weights_proj, (INDEX_HEADS, DIM)),
        ):
            _require_shape(name, tensor, shape)
            _require_dtype(name, tensor, (torch.bfloat16,))

        # Match IndexerFP8: fold both score normalization factors into w64.
        weights_proj = (
            weights_proj * (INDEX_HEAD_DIM**-0.5 * INDEX_HEADS**-0.5)
        ).contiguous()

        def fp32(tag: str, shape: Tuple[int, ...]) -> torch.Tensor:
            tensor = get(tag).float().contiguous()
            _require_shape(tag, tensor, shape)
            return tensor

        def fp32_flat(tag: str, elements: int) -> torch.Tensor:
            tensor = get(tag).float().reshape(-1).contiguous()
            _require_shape(tag, tensor, (elements,))
            return tensor

        attn_norm = get(W.v4_attn_norm).contiguous()
        _require_shape(W.v4_attn_norm, attn_norm, (DIM,))
        _require_dtype(W.v4_attn_norm, attn_norm, (torch.bfloat16,))

        return cls(
            front_fp8=_cat_rows("front_fp8", (wq_a, wkv), (2048, DIM)),
            front_sf=_cat_rows("front_sf", (wq_a_sf, wkv_sf), (16, 56)),
            front_bf16=_cat_rows(
                "front_bf16",
                (main_wkv, main_wgate, index_wkv, index_wgate, weights_proj),
                (2624, DIM),
            ),
            # The fused WQ_B kernel requires indexer rows before main-Q rows.
            wq_b_fp8=_cat_rows(
                "wq_b_fp8", (index_wq_b, main_wq_b), (73728, Q_LORA_RANK)
            ),
            wq_b_sf=_cat_rows("wq_b_sf", (index_wq_b_sf, main_wq_b_sf), (576, 12)),
            q_norm=fp32(W.v4_attn_q_norm, (Q_LORA_RANK,)),
            window_norm=fp32(W.v4_attn_kv_norm, (HEAD_DIM,)),
            indexer_norm=fp32(W.v4_indexer_compressor_norm, (INDEX_HEAD_DIM,)),
            main_compressor_norm=fp32(W.v4_compressor_norm, (HEAD_DIM,)),
            main_ape=fp32(W.v4_compressor_ape, (COMPRESS_RATIO, 1024)),
            indexer_ape=fp32(W.v4_indexer_compressor_ape, (COMPRESS_RATIO, 256)),
            hc_fn=fp32(W.v4_hc_attn_fn, (HC_MIX, HC * DIM)),
            hc_base=fp32(W.v4_hc_attn_base, (HC_MIX,)),
            hc_scale=fp32_flat(W.v4_hc_attn_scale, 3),
            attn_norm=attn_norm,
        )


__all__ = ["MegaCSAWeights"]
