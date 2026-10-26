"""PPU HC head with an instance-owned chunk policy and FP32 reference math."""

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4.hc.base import HCHeadBase


class PpuHCHead(HCHeadBase):
    def __init__(self, *args, options, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        try:
            self._chunk_tokens = max(int(options.get("DSV4_HC_FALLBACK_CHUNK_TOKENS", "16384")), 1)
        except (TypeError, ValueError):
            self._chunk_tokens = 16384

    def _head_impl(self, x: torch.Tensor) -> torch.Tensor:
        shape, dtype = x.size(), x.dtype
        # ``x_flat.float()`` materialises a full fp32 copy ``[T, hc*dim]`` —
        # ~17 GiB at 274K tokens × hc=4 × dim=4096.  Chunk along T so the fp32
        # intermediate stays per-chunk-bounded and we never ride above the
        # KV-cache headroom.
        T = shape[0] if x.dim() == 3 else int(torch.tensor(shape[:-2]).prod().item())
        chunk = self._chunk_tokens

        def _head_chunk(_x):
            _x_flat = _x.flatten(-2).float()
            _rsqrt = torch.rsqrt(
                _x_flat.square().mean(-1, keepdim=True) + self.norm_eps
            )
            _mixes = F.linear(_x_flat, self.fn) * _rsqrt
            _pre = torch.sigmoid(_mixes * self.scale + self.base) + self.hc_eps
            return torch.sum(_pre.unsqueeze(-1) * _x_flat.view(_x.shape), dim=-2)

        if x.dim() == 3 and T > chunk:
            y = torch.empty((T, shape[-1]), dtype=torch.float32, device=x.device)
            for s in range(0, T, chunk):
                e = min(s + chunk, T)
                y[s:e] = _head_chunk(x[s:e])
            y = y.view(*shape[:-2], shape[-1])
        else:
            y = _head_chunk(x)
        return y.to(dtype)


class PpuDecodeHCHead(PpuHCHead):
    def __init__(self, *args, options=None, **kwargs):
        super().__init__(*args, options={} if options is None else options, **kwargs)

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

        mixes = F.linear(x.flatten(-2).float(), self.fn.float())
        mixes *= torch.rsqrt(x.flatten(-2).float().square().mean(-1, keepdim=True) + self.norm_eps)
        pre = torch.sigmoid(mixes * self.scale + self.base)
        return hc_head_sum(x, pre, self.hc_eps)
