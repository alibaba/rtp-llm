"""PPU overrides; unchanged behavior is inherited from the shared DSV4 implementation."""

from __future__ import annotations
from rtp_llm.models_py.modules.dsv4.utils import (
    _V4_FP8_BLOCK_CFG,
    _sparse_attn,
)
import torch
from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models_py.modules.factory.linear import LinearFactory

def _repack_v4_fp8_scale_to_int32(scale: torch.Tensor) -> torch.Tensor:
    """V4 ckpt UE8M0 ``[N/128, K/128]`` to DeepGEMM int32-packed scale."""
    assert scale.dtype == torch.float8_e8m0fnu, f"unexpected scale dtype {scale.dtype}"
    assert scale.dim() == 2, f"unexpected scale dim {scale.dim()}"

    from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor

    n_blk, _ = scale.shape
    n = n_blk * 128
    idx = torch.arange(n, device=scale.device) // 128
    scale_rep = scale.float().index_select(-2, idx)
    return get_mn_major_tma_aligned_packed_ue8m0_tensor(scale_rep)

def _v4_fp8_linear(w: torch.Tensor, s: torch.Tensor, *, platform_provider=None):
    """Build a CudaFp8DeepGEMMLinear from raw V4 FP8 weight + scale tensors."""
    from rtp_llm.platforms.ppu.models.dsv4.impl.platform_provider import build_dsv4_fp8_linear

    def _default_factory(weight: torch.Tensor, scale: torch.Tensor):
        if scale.dtype == torch.float8_e8m0fnu:
            scale = _repack_v4_fp8_scale_to_int32(scale)
        local = {"_w": weight, "_s": scale}
        return LinearFactory.create_linear_from_weights(
            local,
            "_w",
            "_s",
            quant_config=_V4_FP8_BLOCK_CFG,
        )

    assert s is not None, "expected non-null FP8 scale"
    return build_dsv4_fp8_linear(
        _default_factory, w, s, platform_provider=platform_provider
    )

def _v4_fp8_linear_from_dict(weights: dict, weight_key: str, scale_key: str):
    """Backwards-compat bridge over ``_v4_fp8_linear`` for flat dict callers."""
    w = weights[weight_key]
    s = weights[scale_key]
    if s.dtype == torch.float8_e8m0fnu:
        s = _repack_v4_fp8_scale_to_int32(s)
        weights[scale_key] = s
    return _v4_fp8_linear(w, s)
