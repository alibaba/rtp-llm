"""Fixed projection and RMS reduction for DeepSeek hyper-connections."""

import importlib

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.determinism.quantized_linear import linear
from rtp_llm.models_py.modules.dsv4.hc.tilelang_impl import TileLangHCUnit


@triton.jit
def _normalize_mixes(
    residual,
    mixes,
    width: tl.constexpr,
    mix_width: tl.constexpr,
    eps: tl.constexpr,
    block: tl.constexpr,
    mix_block: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.arange(0, block)
    partial = tl.zeros((block,), tl.float32)
    for begin in range(0, width, block):
        idx = begin + tile
        x = tl.load(residual + row * width + idx, idx < width, other=0).to(tl.float32)
        partial += x * x
    factor = tl.rsqrt(tl.sum(partial, 0) / width + eps)
    col = tl.arange(0, mix_block)
    pointer = mixes + row * mix_width + col
    raw = tl.load(pointer, col < mix_width, other=0)
    tl.store(pointer, raw * factor, col < mix_width)


def prenorm(residual: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply the same FP32 contraction and per-row RMS at every token count."""
    width = weight.size(1)
    x = residual.reshape(-1, width).contiguous()
    mixes = linear(x, weight, out_dtype=torch.float32)
    if x.size(0):
        _normalize_mixes[(x.size(0),)](
            x,
            mixes,
            width,
            weight.size(0),
            eps,
            256,
            triton.next_power_of_2(weight.size(0)),
            num_warps=4,
        )
    return mixes.view(*residual.shape[:-2], weight.size(0))


class BatchInvariantHCUnit(TileLangHCUnit):
    def _pre_impl(self, x: torch.Tensor, dbg_tag=None):
        # Keep the existing per-token mix and sinkhorn kernels. The split-K
        # prenorm GEMM is the part of TileLang mHC whose geometry depends on M.
        from rtp_llm.models_py.modules.dsv4 import tilelang_kernels  # noqa: F401

        ops = importlib.import_module(
            "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.functional"
        )
        residual = x.reshape(1, -1, self.hc_mult, self.dim)
        mixes = prenorm(residual, self.fn, self.norm_eps)
        pre, post, comb = ops.mhc_pre_split_mixes(
            mixes, self.scale, self.base, self.hc_mult, 2.0, self.hc_eps
        )
        comb = ops.sinkhorn_normalize(
            comb, repeat=self.hc_sinkhorn_iters, eps=self.hc_eps
        )
        layer_input = ops.mhc_pre_apply_mix(residual, pre)
        leading = x.shape[:-2]
        return (
            layer_input.reshape(*leading, self.dim),
            post.reshape(*leading, self.hc_mult, 1),
            comb.reshape(*leading, self.hc_mult, self.hc_mult),
        )
