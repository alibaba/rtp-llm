"""Shared DeepSeek vision encoder and aligner.

Extracted from ca840ddb38a7fafdb6781151ae11465454895de8; image-token layout
belongs to each model adapter.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.models_py.modules.dsv41.vision_residual_add import (
    apply_vision_residual_rmsnorm,
)


@lru_cache(32)
def _vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float, device: str):
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    hpos = torch.arange(n_h, device=device).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w, device=device).unsqueeze(0).expand(n_h, n_w)
    freqs = (
        torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    ).flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Checkpoint parameters remain BF16. FlashInfer accumulates in FP32
        # inside the kernel; no per-forward weight conversion is needed.
        self._rmsnorm_backend = "flashinfer"
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))

    def _uses_flashinfer(self, x):
        return x.is_cuda and not self.training and not torch.is_grad_enabled()

    def _flashinfer_rmsnorm(self, x):
        from flashinfer.norm import rmsnorm

        if x.dtype != torch.bfloat16 or self.weight.dtype != torch.bfloat16:
            raise ValueError("FlashInfer vision RMSNorm requires BF16 input and weight")
        if x.ndim not in (2, 3):
            raise ValueError("FlashInfer vision RMSNorm requires 2D/3D input")
        if x.stride(-1) != 1 or self.weight.device != x.device:
            raise ValueError(
                "FlashInfer vision RMSNorm requires contiguous last dimension and same-device weight"
            )
        # Out-of-place: the residual stream may still own x. Missing optional
        # dependencies or failed launches are errors, not silent native fallback.
        # FlashInfer loads BF16 operands and reduces in FP32 inside the kernel.
        return rmsnorm(x, self.weight, eps=self.eps)

    def forward(self, x: torch.Tensor):
        if self._uses_flashinfer(x):
            return self._flashinfer_rmsnorm(x)
        # CPU and autograd retain the ordinary expression. Do not catch
        # dependency or launch errors after selecting CUDA inference.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)

    def forward_with_residual(self, x, residual):
        """Preserve both outputs needed by the pre-norm residual block."""
        if self._uses_flashinfer(x):
            fused = apply_vision_residual_rmsnorm(
                x, residual, self.weight, self.eps
            )
            if fused is not None:
                return fused
        summed = x + residual
        return summed, self(summed)


class PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(
            3 * config["vision_patch_size"] ** 2, config["vision_dim"]
        )

    def forward(self, x):
        return self.proj(x.flatten(1))


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config["vision_dim"]
        self.n_heads = config["vision_n_heads"]
        self.head_dim = dim // self.n_heads
        self.wqkv = nn.Linear(dim, 3 * dim)
        self.wo = nn.Linear(dim, dim)

    def _apply_rotary(self, q, k, cos, sin):
        return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)

    def _attention(self, q, k, v):
        return (
            F.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                k.transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0),
            )
            .squeeze(0)
            .transpose(0, 1)
        )

    def forward(self, x, cos, sin):
        n = x.size(0)
        q, k, v = (
            t.view(n, self.n_heads, self.head_dim)
            for t in self.wqkv(x).chunk(3, dim=-1)
        )
        q, k = self._apply_rotary(q, k, cos, sin)
        return self.wo(self._attention(q, k, v).reshape(n, -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim, inter = config["vision_dim"], config["vision_inter_dim"]
        self.w1 = nn.Linear(dim, 2 * inter, bias=False)
        self.w2 = nn.Linear(inter, dim, bias=False)

    def forward(self, x):
        from rtp_llm.models_py.modules.dsv41.vision_silu_mul_fused import (
            apply_vision_silu_mul,
        )
        from rtp_llm.models_py.modules.dsv41.vision_swiglu import apply_vision_swiglu

        if not self.training:
            intermediate = apply_vision_swiglu(x, self.w1.weight)
            if intermediate is not None:
                return self.w2(intermediate)

        # Unsupported device/shape/dtype and autograd retain the old path.
        # Supported Quack dependency/launch errors propagate from the helper.
        gate, up = self.w1(x).chunk(2, dim=-1)
        intermediate = apply_vision_silu_mul(gate, up)
        if intermediate is not None:
            return self.w2(intermediate)
        return self.w2(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, config, *, attention_cls=Attention):
        super().__init__()
        dim = config["vision_dim"]
        self.norm1, self.attn = RMSNorm(dim), attention_cls(config)
        self.norm2, self.mlp = RMSNorm(dim), MLP(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    def __init__(self, config, *, attention_cls=Attention):
        super().__init__()
        self.rope_dim = config["vision_dim"] // config["vision_n_heads"] // 2
        self.rope_theta = float(config.get("vision_rope_theta", 10000.0))
        self.patch_embed = PatchEmbed(config)
        self.blocks = nn.ModuleList(
            [
                Block(config, attention_cls=attention_cls)
                for _ in range(config["vision_n_layers"])
            ]
        )
        self.norm = RMSNorm(config["vision_dim"])

    def forward(self, patches, n_h, n_w):
        x = self.patch_embed(patches)
        cos, sin = _vision_cos_sin(
            n_h, n_w, self.rope_dim, self.rope_theta, str(x.device)
        )
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class Aligner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.downsample_ratio = config["vision_downsample_ratio"]
        in_dim = config["vision_dim"] * self.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, config["hidden_size"])
        self.w2 = nn.Linear(config["hidden_size"], config["hidden_size"])

    def forward(self, x, n_h, n_w):
        ratio = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % ratio, 0, -n_h % ratio))
        # Keep the official unfold/transpose strides at the BF16 GEMM boundary.
        x = F.unfold(x.unsqueeze(0), ratio, stride=ratio).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))
