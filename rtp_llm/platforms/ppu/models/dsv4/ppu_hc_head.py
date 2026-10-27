"""Decode HC head with the original FP32 projection, RMS and sigmoid path."""

import torch
from rtp_llm.models_py.modules.dsv4.hc.fallback_impl import (
    FallbackHCHead,
    _tp_linear_mixes,
)


class PpuDecodeHCHead(FallbackHCHead):
    def _head_impl(self, x):
        if not (
            x.is_cuda
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and x.ndim in (3, 4)
            and x.shape[-2:] == (4, 4096)
            and 0 < x.numel() // 16384 <= 512
            and getattr(self, "tp_size", 1) == 1
        ):
            return super()._head_impl(x)
        from ...kernels.ppu_hc_head_sum import hc_head_sum

        mixes = _tp_linear_mixes(self, x.flatten(-2), use_fp32=True)
        pre = torch.sigmoid(mixes * self.scale + self.base)
        return hc_head_sum(x, pre, self.hc_eps)
