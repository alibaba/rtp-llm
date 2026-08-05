"""B12X NVFP4 weight preparation owned by the CUDA loading boundary."""

import logging
from typing import Optional

import torch

from rtp_llm.config.moe_config import (
    B12X_ZEROED_ENERGY_LIMIT_ENV,
    validate_b12x_zeroed_energy_limit,
)
from rtp_llm.config.quant_config import NVFP4_BLOCK_SIZE
from rtp_llm.device.flashinfer_b12x_adapter import convert_b12x_blockscale_to_mma_layout

logger = logging.getLogger(__name__)

_E4M3_MIN_NORMAL = 2.0**-6


def validate_b12x_checkpoint_input_scale(
    name: str,
    input_scale: Optional[torch.Tensor],
    expected_device: torch.device,
) -> None:
    """Validate calibration metadata that B12X intentionally does not consume."""
    if input_scale is None:
        return
    if input_scale.device != expected_device:
        raise ValueError(
            f"b12x FP4 {name} input_scale must be on {expected_device}, "
            f"got {input_scale.device}"
        )
    if not bool(torch.isfinite(input_scale).all()) or not bool((input_scale > 0).all()):
        raise ValueError(
            f"b12x FP4 {name} input_scale must contain finite, strictly "
            "positive calibration values"
        )


def validate_folded_b12x_blockscale(
    name: str,
    product: torch.Tensor,
    folded: torch.Tensor,
    zeroed_energy_limit: float,
) -> tuple[torch.Tensor, float, float]:
    """Validate the e4m3 fold and return statistics used for diagnostics."""
    folded_f32 = folded.to(torch.float32)
    if not bool(torch.isfinite(folded_f32).all()):
        raise ValueError(
            f"b12x FP4: {name} blockscale overflowed e4m3 while folding "
            "weight_scale_2; the checkpoint's scales are out of range"
        )

    sf_nonzero = product != 0
    zeroed = (folded_f32 == 0) & sf_nonzero
    total_energy = (product**2).sum().item()
    if total_energy == 0:
        raise ValueError(
            f"b12x FP4: {name} blockscales have zero total scale energy after "
            "folding weight_scale_2; the checkpoint scale is missing, zero, "
            "or paired with the wrong weight tensor"
        )
    lost_energy = (product[zeroed] ** 2).sum().item() / total_energy
    if lost_energy > zeroed_energy_limit:
        raise ValueError(
            f"b12x FP4: folding weight_scale_2 underflowed "
            f"{int(zeroed.sum())}/{zeroed.numel()} {name} blockscales to "
            f"zero, dropping {lost_energy:.2%} of the total scale energy from "
            f"the GEMM (configured limit: {zeroed_energy_limit:.2%}). SM12X "
            "has no alternative single-GPU FP4 MoE backend; use non-FP4 MoE "
            f"weights or temporarily raise {B12X_ZEROED_ENERGY_LIMIT_ENV}."
        )

    subnormal_frac = (
        ((folded_f32.abs() < _E4M3_MIN_NORMAL) & sf_nonzero & ~zeroed)
        .float()
        .mean()
        .item()
    )
    return zeroed, lost_energy, subnormal_frac


def prepare_b12x_blockscale(
    name: str,
    kernel: torch.Tensor,
    blockscale: torch.Tensor,
    scale_2: Optional[torch.Tensor],
    input_scale: Optional[torch.Tensor],
    zeroed_energy_limit: float,
) -> torch.Tensor:
    """Fold ModelOpt's global scale and return the vendor's 6D MMA view."""
    zeroed_energy_limit = validate_b12x_zeroed_energy_limit(zeroed_energy_limit)
    if kernel.dtype is not torch.uint8 or kernel.ndim != 3:
        raise ValueError(
            f"b12x FP4 {name} must be a rank-3 packed uint8 tensor, got "
            f"shape={tuple(kernel.shape)}, dtype={kernel.dtype}"
        )
    if blockscale.dtype is not torch.float8_e4m3fn or blockscale.ndim != 3:
        raise ValueError(
            f"b12x FP4 {name} blockscale must be rank-3 torch.float8_e4m3fn, "
            f"got shape={tuple(blockscale.shape)}, dtype={blockscale.dtype}"
        )
    if scale_2 is None:
        raise ValueError(f"b12x FP4 {name} requires weight_scale_2")

    num_experts, rows, packed_k = kernel.shape
    expected_scale_shape = (
        num_experts,
        rows,
        packed_k * 2 // NVFP4_BLOCK_SIZE,
    )
    if tuple(blockscale.shape) != expected_scale_shape:
        raise ValueError(
            f"b12x FP4 {name} blockscale shape must be {expected_scale_shape}, "
            f"got {tuple(blockscale.shape)}"
        )
    if scale_2.dtype is not torch.float32 or (
        scale_2.ndim == 0
        or scale_2.shape[0] != num_experts
        or scale_2.numel() != num_experts
    ):
        raise ValueError(
            f"b12x FP4 {name} weight_scale_2 must contain one float32 scalar "
            f"per expert ({num_experts} values), got shape={tuple(scale_2.shape)}, "
            f"dtype={scale_2.dtype}"
        )
    expected_device = kernel.device
    if blockscale.device != expected_device or scale_2.device != expected_device:
        raise ValueError(
            f"b12x FP4 {name}, blockscale, and weight_scale_2 must share one "
            f"device, got {expected_device}, {blockscale.device}, {scale_2.device}"
        )
    if not bool(torch.isfinite(scale_2).all()) or not bool((scale_2 > 0).all()):
        raise ValueError(
            f"b12x FP4 {name} weight_scale_2 must contain finite, strictly "
            "positive values for every expert"
        )
    validate_b12x_checkpoint_input_scale(name, input_scale, expected_device)

    product = blockscale.to(torch.float32) * scale_2.reshape(num_experts, 1, 1).to(
        torch.float32
    )
    folded = product.to(torch.float8_e4m3fn)
    zeroed, lost_energy, subnormal_frac = validate_folded_b12x_blockscale(
        name, product, folded, zeroed_energy_limit
    )
    if bool(zeroed.any()):
        logger.info(
            "b12x FP4: %d/%d %s blockscale entries underflowed e4m3 to zero "
            "while folding weight_scale_2 (%.4f%% of scale energy; near-zero "
            "blocks, negligible precision impact).",
            int(zeroed.sum()),
            zeroed.numel(),
            name,
            lost_energy * 100,
        )
    if subnormal_frac > 0.5:
        logger.info(
            "b12x FP4: %.1f%% of %s blockscales are e4m3-subnormal after "
            "folding weight_scale_2 (benign, but scale mantissa precision is "
            "reduced for those blocks).",
            subnormal_frac * 100,
            name,
        )

    return convert_b12x_blockscale_to_mma_layout(
        folded.reshape(-1).contiguous(),
        m=rows,
        k=packed_k * 2,
        num_groups=num_experts,
    )
