"""Immutable TP1 DSV4 CSA megakernel weight layouts (Pro & Flash geometries)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch

HC = 4
HC_MIX = 24
HEAD_DIM = 512
ROPE_DIM = 64
O_LORA_RANK = 1024
INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
COMPRESS_RATIO = 4
MAX_BATCH = 128
MQA_SPLIT_KV = 256


@dataclass(frozen=True)
class CSAGeometry:
    """Width-class parameters that differ between DeepSeek-V4 Pro and Flash.

    Everything else (HEAD_DIM, ROPE_DIM, indexer geometry, compress ratio,
    window) is shared and stays module-level. The extension compiles both
    geometries into one binary and dispatches by tensor shape, so these
    values only steer packing shapes and workspace sizes.
    """

    dim: int
    q_lora_rank: int
    main_heads: int
    o_groups: int

    @property
    def n_main(self) -> int:
        return self.main_heads * HEAD_DIM

    @property
    def n_index(self) -> int:
        return INDEX_HEADS * INDEX_HEAD_DIM

    @property
    def n_merged(self) -> int:
        return self.n_main + self.n_index

    @property
    def front_fp8_rows(self) -> int:
        # wq_a rows ++ window-KV rows
        return self.q_lora_rank + HEAD_DIM

    @property
    def front_bf16_rows(self) -> int:
        # compressor kv/gate (1024+1024) + indexer kv/gate (256+256) + w64.
        # Geometry-invariant: both models share HEAD_DIM and the indexer.
        return 2624

    @property
    def front_out_dim(self) -> int:
        return self.front_fp8_rows + self.front_bf16_rows

    @property
    def sf_k(self) -> int:
        return self.dim // 128

    @property
    def sf_q(self) -> int:
        return self.q_lora_rank // 128


PRO_GEOMETRY = CSAGeometry(dim=7168, q_lora_rank=1536, main_heads=128, o_groups=16)
FLASH_GEOMETRY = CSAGeometry(dim=4096, q_lora_rank=1024, main_heads=64, o_groups=8)
GEOMETRY_BY_DIM = {g.dim: g for g in (PRO_GEOMETRY, FLASH_GEOMETRY)}

# Pro aliases kept for existing imports; new code should read the geometry.
DIM = PRO_GEOMETRY.dim
Q_LORA_RANK = PRO_GEOMETRY.q_lora_rank
MAIN_HEADS = PRO_GEOMETRY.main_heads
O_GROUPS = PRO_GEOMETRY.o_groups
FRONT_OUT_DIM = PRO_GEOMETRY.front_out_dim
assert FRONT_OUT_DIM == 4672 and FLASH_GEOMETRY.front_out_dim == 4160


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
    geometry: CSAGeometry = field(default=PRO_GEOMETRY)

    @classmethod
    def from_layer_weights(
        cls,
        layer_weights: Dict[str, torch.Tensor],
        geometry: CSAGeometry = PRO_GEOMETRY,
    ) -> "MegaCSAWeights":
        """Pack one CSA layer without changing checkpoint FP8 bits or scales."""
        from rtp_llm.utils.model_weight import W

        g = geometry

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
            ("wq_a", wq_a, (g.q_lora_rank, g.dim)),
            ("wkv", wkv, (512, g.dim)),
            ("main_wq_b", main_wq_b, (g.n_main, g.q_lora_rank)),
            ("index_wq_b", index_wq_b, (g.n_index, g.q_lora_rank)),
        ):
            _require_shape(name, tensor, shape)
            _require_dtype(name, tensor, (torch.float8_e4m3fn,))

        wq_a_sf = get(W.v4_attn_wq_a_s)
        wkv_sf = get(W.v4_attn_wkv_s)
        main_wq_b_sf = get(W.v4_attn_wq_b_s)
        index_wq_b_sf = get(W.v4_indexer_wq_b_s)
        scale_dtypes = (torch.float8_e8m0fnu, torch.uint8)
        for name, tensor, shape in (
            ("wq_a_sf", wq_a_sf, (g.sf_q, g.sf_k)),
            ("wkv_sf", wkv_sf, (4, g.sf_k)),
            ("main_wq_b_sf", main_wq_b_sf, (g.n_main // 128, g.sf_q)),
            ("index_wq_b_sf", index_wq_b_sf, (g.n_index // 128, g.sf_q)),
        ):
            _require_shape(name, tensor, shape)
            _require_dtype(name, tensor, scale_dtypes)

        main_wkv = get(W.v4_compressor_wkv)
        main_wgate = get(W.v4_compressor_wgate)
        index_wkv = get(W.v4_indexer_compressor_wkv)
        index_wgate = get(W.v4_indexer_compressor_wgate)
        weights_proj = get(W.v4_indexer_weights_proj_w)
        for name, tensor, shape in (
            ("main_compressor_wkv", main_wkv, (1024, g.dim)),
            ("main_compressor_wgate", main_wgate, (1024, g.dim)),
            ("indexer_compressor_wkv", index_wkv, (256, g.dim)),
            ("indexer_compressor_wgate", index_wgate, (256, g.dim)),
            ("indexer_weights_proj", weights_proj, (INDEX_HEADS, g.dim)),
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
        _require_shape(W.v4_attn_norm, attn_norm, (g.dim,))
        _require_dtype(W.v4_attn_norm, attn_norm, (torch.bfloat16,))

        return cls(
            front_fp8=_cat_rows("front_fp8", (wq_a, wkv), (g.front_fp8_rows, g.dim)),
            front_sf=_cat_rows(
                "front_sf", (wq_a_sf, wkv_sf), (g.front_fp8_rows // 128, g.sf_k)
            ),
            front_bf16=_cat_rows(
                "front_bf16",
                (main_wkv, main_wgate, index_wkv, index_wgate, weights_proj),
                (g.front_bf16_rows, g.dim),
            ),
            # The fused WQ_B kernel requires indexer rows before main-Q rows.
            wq_b_fp8=_cat_rows(
                "wq_b_fp8", (index_wq_b, main_wq_b), (g.n_merged, g.q_lora_rank)
            ),
            wq_b_sf=_cat_rows(
                "wq_b_sf",
                (index_wq_b_sf, main_wq_b_sf),
                (g.n_merged // 128, g.sf_q),
            ),
            q_norm=fp32(W.v4_attn_q_norm, (g.q_lora_rank,)),
            window_norm=fp32(W.v4_attn_kv_norm, (HEAD_DIM,)),
            indexer_norm=fp32(W.v4_indexer_compressor_norm, (INDEX_HEAD_DIM,)),
            main_compressor_norm=fp32(W.v4_compressor_norm, (HEAD_DIM,)),
            main_ape=fp32(W.v4_compressor_ape, (COMPRESS_RATIO, 1024)),
            indexer_ape=fp32(W.v4_indexer_compressor_ape, (COMPRESS_RATIO, 256)),
            hc_fn=fp32(W.v4_hc_attn_fn, (HC_MIX, HC * g.dim)),
            hc_base=fp32(W.v4_hc_attn_base, (HC_MIX,)),
            hc_scale=fp32_flat(W.v4_hc_attn_scale, 3),
            attn_norm=attn_norm,
            geometry=g,
        )


__all__ = [
    "CSAGeometry",
    "FLASH_GEOMETRY",
    "GEOMETRY_BY_DIM",
    "MegaCSAWeights",
    "PRO_GEOMETRY",
]
