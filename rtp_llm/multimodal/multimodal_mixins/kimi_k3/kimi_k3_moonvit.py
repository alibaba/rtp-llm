"""Kimi-K3 MoonViT implementation."""

import logging
import math
from copy import deepcopy
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.utils.flash_attn_utils import can_use_flash_attn

try:
    from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_rope_triton import (
        maybe_fused_apply_rope,
    )
except (ImportError, OSError, AttributeError) as exc:  # pragma: no cover
    logging.info(f"K3 fused RoPE unavailable ({exc}); falling back to eager RoPE")

    def maybe_fused_apply_rope(
        _xq: torch.Tensor, _xk: torch.Tensor, _freqs_cis: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return None

_FLASH_ATTN_AVAILABLE = False
try:
    if can_use_flash_attn():
        from flash_attn import flash_attn_varlen_func  # type: ignore

        _FLASH_ATTN_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent
    logging.info(f"flash_attn unavailable for MoonViT ({exc}); falling back to SDPA")


# -----------------------------------------------------------------------------
# Position embeddings
# -----------------------------------------------------------------------------


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def _get_1d_sincos_pos_embed(embed_dim: int, t_size: int) -> np.ndarray:
    grid_t = np.arange(t_size, dtype=np.float32)
    return _get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)


class Learnable2DInterpPosEmbDivided_fixed(nn.Module):
    """Learnable 2D (H, W) embedding + sincos time embedding (non-persistent).

    Mirrors the HF reference: `weight` is the only ckpt-side parameter; the
    `time_weight` buffer is reconstructed at init time from a 1D sincos grid
    and excluded from the state dict.
    """

    def __init__(
        self,
        height: int,
        width: int,
        num_frames: int,
        dim: int,
        interpolation_mode: str = "bicubic",
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.dim = dim
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        self.register_buffer(
            "time_weight",
            torch.from_numpy(_get_1d_sincos_pos_embed(dim, num_frames))
            .float()
            .unsqueeze(1),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)

    def _interp(self, target_h: int, target_w: int) -> torch.Tensor:
        # weight: (H, W, dim) -> (1, dim, H, W) -> interpolate -> (target_h*target_w, dim)
        org = self.weight
        x = (
            F.interpolate(
                org.permute(2, 0, 1).unsqueeze(0),
                size=(target_h, target_w),
                mode=self.interpolation_mode,
            )
            .squeeze(0)
            .permute(1, 2, 0)
            .flatten(end_dim=1)
        )
        return x

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for t, h, w in grid_thws.tolist():
            assert (
                t <= self.num_frames
            ), f"t={t} exceeds init_pos_emb_time={self.num_frames}"
            if (h, w) == (self.weight.shape[0], self.weight.shape[1]):
                pos_emb_2d = self.weight.flatten(end_dim=1)
            else:
                pos_emb_2d = self._interp(h, w)

            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = (
                    pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) + self.time_weight[0:t]
                )
            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))

        pos_emb = pos_embs[0] if len(pos_embs) == 1 else torch.cat(pos_embs)
        out = x + pos_emb.to(x.dtype)
        return out


class Rope2DPosEmbRepeated(nn.Module):
    """2D rotary positional embedding with multi-resolution support.

    Lazily caches ``freqs_cis`` (complex64) up to ``(max_height, max_width)``.
    Per-image freqs are sliced from this cache and repeated along the time
    axis when ``t > 1``.
    """

    def __init__(
        self,
        dim: int,
        max_height: int,
        max_width: int,
        theta_base: float = 10000.0,
    ) -> None:
        super().__init__()
        assert dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
        self.dim = dim
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N, dtype=torch.float32, device=device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = torch.arange(0, self.dim, 4, dtype=torch.float32, device=device)[
            : (self.dim // 4)
        ]
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()
        y_freqs = torch.outer(y_pos, freqs).float()
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
        freqs_cis = torch.cat(
            [x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1
        ).reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis(
        self, grid_thws: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        if not hasattr(self, "freqs_cis"):
            self.register_buffer(
                "freqs_cis",
                self._precompute_freqs_cis(device),
                persistent=False,
            )
        elif self.freqs_cis.device != device:
            self.freqs_cis = self.freqs_cis.to(device)

        shapes = grid_thws.tolist()
        per_media_freqs = []
        for t, h, w in shapes:
            freqs = self.freqs_cis[:h, :w].reshape(-1, self.dim // 2)
            if t > 1:
                freqs = freqs.repeat(t, 1)
            per_media_freqs.append(freqs)
        freqs_cis = (
            per_media_freqs[0]
            if len(per_media_freqs) == 1
            else torch.cat(per_media_freqs, dim=0)
        )
        return freqs_cis


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply 2D rotary embedding via complex multiplication.

    Args:
        xq, xk: ``(seq, num_heads, head_dim)`` (head_dim must be even).
        freqs_cis: ``(seq, head_dim/2)`` complex64.
    """
    fused = maybe_fused_apply_rope(xq, xk, freqs_cis)
    if fused is not None:
        return fused

    assert freqs_cis.dtype == torch.complex64, freqs_cis.dtype
    freqs_cis = freqs_cis.unsqueeze(-2)  # (seq, 1, head_dim/2)
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# -----------------------------------------------------------------------------
# Blocks
# -----------------------------------------------------------------------------


class MoonVision3dPatchEmbed(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        ps = config.patch_size
        self.patch_size = (ps, ps)
        self.proj = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.vt_hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=config.patch_embed_proj_bias,
        )
        if config.pos_emb_type != "divided_fixed":
            raise NotImplementedError(
                f"pos_emb_type={config.pos_emb_type} not supported"
            )
        # pos_emb_interpolation_mode is NOT forwarded on purpose: the ckpt says
        # "bilinear" but official modeling never reads it, so upstream is bicubic.
        self.pos_emb = Learnable2DInterpPosEmbDivided_fixed(
            height=config.init_pos_emb_height,
            width=config.init_pos_emb_width,
            num_frames=config.init_pos_emb_time,
            dim=config.vt_hidden_size,
        )

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        # x: (sum_patches, 3, ph, pw)
        x = self.proj(x).view(x.size(0), -1)
        x = self.pos_emb(x, grid_thws)
        return x


class MLP2(nn.Module):
    """Two-layer MLP matching HF/vLLM naming (``fc0``/``fc1``)."""

    def __init__(
        self,
        hidden_dim: int,
        mlp_dim: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.fc0 = nn.Linear(hidden_dim, mlp_dim, bias=bias)
        self.fc1 = nn.Linear(mlp_dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc0(x)
        x = F.gelu(x, approximate="tanh")
        x = self.fc1(x)
        return x


def _make_vision_norm(norm_type: str, hidden_dim: int) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(hidden_dim)
    if norm_type == "rmsnorm":
        return nn.RMSNorm(hidden_dim)
    raise NotImplementedError(f"norm_type={norm_type} not supported")


def _make_vision_mlp(
    mlp_type: str, hidden_dim: int, mlp_dim: int, bias: bool
) -> nn.Module:
    if mlp_type == "mlp2":
        return MLP2(hidden_dim, mlp_dim, bias=bias)
    raise NotImplementedError(f"mlp_type={mlp_type} not supported")


class MoonViTEncoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        qkv_hidden_size: Optional[int] = None,
        norm_type: str = "layernorm",
        mlp_type: str = "mlp2",
        attn_bias: bool = True,
        linear_bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.qkv_hidden_size = qkv_hidden_size or hidden_dim
        self.head_dim = self.qkv_hidden_size // num_heads
        assert self.head_dim * num_heads == self.qkv_hidden_size

        self.norm0 = _make_vision_norm(norm_type, hidden_dim)
        self.norm1 = _make_vision_norm(norm_type, hidden_dim)
        self.wqkv = nn.Linear(hidden_dim, self.qkv_hidden_size * 3, bias=attn_bias)
        self.wo = nn.Linear(self.qkv_hidden_size, hidden_dim, bias=attn_bias)
        self.mlp = _make_vision_mlp(mlp_type, hidden_dim, mlp_dim, bias=linear_bias)

    def _attention(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor,
        max_seqlen: int,
        segment_offsets: Optional[List[int]] = None,
    ) -> torch.Tensor:
        seq_length = x.size(0)
        xqkv = self.wqkv(x)
        xqkv = xqkv.view(seq_length, 3, self.num_heads, self.head_dim)
        xq, xk, xv = torch.unbind(xqkv, dim=1)
        xq, xk = apply_rope(xq, xk, rope_freqs_cis)

        if _FLASH_ATTN_AVAILABLE and xq.is_cuda:
            attn_out = flash_attn_varlen_func(
                xq,
                xk,
                xv,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=int(max_seqlen),
                max_seqlen_k=int(max_seqlen),
            )
            attn_out = attn_out.reshape(seq_length, self.num_heads * self.head_dim)
        else:
            # Block-diagonal mask via SDPA fallback. Build an attention mask
            # only when there is more than one segment to keep the common
            # single-image path cheap.
            n_seg = (
                len(segment_offsets) - 1 if segment_offsets else cu_seqlens.numel() - 1
            )
            if n_seg <= 1:
                q = xq.transpose(0, 1).unsqueeze(0)  # (1, H, S, D)
                k = xk.transpose(0, 1).unsqueeze(0)
                v = xv.transpose(0, 1).unsqueeze(0)
                attn_out = F.scaled_dot_product_attention(q, k, v)
                attn_out = (
                    attn_out.squeeze(0)
                    .transpose(0, 1)
                    .reshape(seq_length, self.num_heads * self.head_dim)
                )
            else:
                # Avoid a quadratic packed block-diagonal mask in the fallback.
                # Each media segment is independent, so running SDPA per segment
                # is equivalent and keeps peak memory bounded by the largest one.
                cu = segment_offsets or cu_seqlens.tolist()
                segment_outputs = []
                for i in range(n_seg):
                    s, e = cu[i], cu[i + 1]
                    q = xq[s:e].transpose(0, 1).unsqueeze(0)
                    k = xk[s:e].transpose(0, 1).unsqueeze(0)
                    v = xv[s:e].transpose(0, 1).unsqueeze(0)
                    segment_outputs.append(
                        F.scaled_dot_product_attention(q, k, v)
                        .squeeze(0)
                        .transpose(0, 1)
                    )
                attn_out = torch.cat(segment_outputs, dim=0).reshape(
                    seq_length, self.num_heads * self.head_dim
                )

        return self.wo(attn_out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor,
        max_seqlen: int,
        segment_offsets: Optional[List[int]] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)
        hidden_states = self._attention(
            hidden_states,
            cu_seqlens,
            rope_freqs_cis,
            max_seqlen,
            segment_offsets,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class MoonViT3dEncoder(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        assert (
            config.video_attn_type == "spatial_temporal"
        ), f"only spatial_temporal video_attn_type supported, got {config.video_attn_type}"
        qkv_hidden_size = config.qkv_hidden_size or config.vt_hidden_size
        head_dim = qkv_hidden_size // config.vt_num_attention_heads
        self.rope_2d = Rope2DPosEmbRepeated(
            head_dim,
            max_height=config.max_pos_emb_height,
            max_width=config.max_pos_emb_width,
            theta_base=config.rope_theta,
        )
        self.blocks = nn.ModuleList(
            [
                MoonViTEncoderLayer(
                    num_heads=config.vt_num_attention_heads,
                    hidden_dim=config.vt_hidden_size,
                    mlp_dim=config.vt_intermediate_size,
                    qkv_hidden_size=config.qkv_hidden_size,
                    norm_type=config.norm_type,
                    mlp_type=config.mlp_type,
                    attn_bias=config.attn_bias,
                    linear_bias=config.linear_bias,
                )
                for _ in range(config.vt_num_hidden_layers)
            ]
        )
        self.final_layernorm = _make_vision_norm(
            config.norm_type, config.vt_hidden_size
        )

    def forward(
        self, hidden_states: torch.Tensor, grid_thws: torch.Tensor
    ) -> torch.Tensor:
        rope_freqs_cis = self.rope_2d.get_freqs_cis(
            grid_thws=grid_thws, device=hidden_states.device
        )
        lengths = [t * h * w for t, h, w in grid_thws.tolist()]
        segment_offsets = [0]
        for length in lengths:
            segment_offsets.append(segment_offsets[-1] + length)
        cu_seqlens = torch.tensor(segment_offsets, dtype=torch.int32)
        if _FLASH_ATTN_AVAILABLE and hidden_states.is_cuda:
            cu_seqlens = cu_seqlens.pin_memory().to(
                hidden_states.device, non_blocking=True
            )
        max_seqlen = max(lengths)
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens,
                rope_freqs_cis,
                max_seqlen,
                segment_offsets,
            )
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states


def tpool_patch_merger(
    x: torch.Tensor,
    grid_thws: torch.Tensor,
    merge_kernel_size: Tuple[int, int] = (2, 2),
) -> List[torch.Tensor]:
    """Temporal pooling + 2x2 spatial merge.

    Splits the packed (sum(t*h*w), d) tensor by per-media lengths, time-pools
    each segment, then rearranges to ``(nh*nw, kh*kw, d)`` ready for the
    multimodal projector.
    """
    kh, kw = merge_kernel_size
    lengths = (grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2]).tolist()
    seqs = x.split(lengths, dim=0)

    outputs: List[torch.Tensor] = []
    for seq, (t, h, w) in zip(seqs, grid_thws.tolist()):
        nh, nw = h // kh, w // kw
        v = seq.view(t, nh, kh, nw, kw, -1).mean(dim=0)  # (nh, kh, nw, kw, d)
        v = v.permute(0, 2, 1, 3, 4).contiguous().reshape(nh * nw, kh * kw, -1)
        outputs.append(v)
    return outputs


class MoonViT3dPretrainedModel(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = deepcopy(config)
        self.merge_kernel_size = tuple(config.merge_kernel_size)
        self.patch_size = config.patch_size
        self.merge_type = config.merge_type
        self.patch_embed = MoonVision3dPatchEmbed(config)
        self.encoder = MoonViT3dEncoder(config)

    def forward(
        self, pixel_values: torch.Tensor, grid_thws: torch.Tensor
    ) -> List[torch.Tensor]:
        hidden_states = self.patch_embed(pixel_values, grid_thws)
        hidden_states = self.encoder(hidden_states, grid_thws)
        if self.merge_type != "sd2_tpool":
            raise NotImplementedError(f"merge_type={self.merge_type} not supported")
        return tpool_patch_merger(
            hidden_states, grid_thws, merge_kernel_size=self.merge_kernel_size
        )
