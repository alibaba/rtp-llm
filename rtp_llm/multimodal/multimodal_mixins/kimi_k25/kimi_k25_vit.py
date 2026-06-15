"""Kimi-K2.5 vision tower (MoonViT 3D) + multimodal projector.

Architecture mirrors the HF reference and vLLM/SGLang implementations:

    KimiK25ImageEmbedding
    ├── vision_tower (MoonViT3dPretrainedModel)
    │   ├── patch_embed (MoonVision3dPatchEmbed)
    │   │   ├── proj    : Conv2d(3, 1152, k=14, s=14)
    │   │   └── pos_emb (Learnable2DInterpPosEmbDivided_fixed)
    │   │         ├── weight       : Parameter(64, 64, 1152)
    │   │         └── time_weight  : non-persistent buffer
    │   └── encoder (MoonViT3dEncoder)
    │       ├── rope_2d (Rope2DPosEmbRepeated)  — no parameters
    │       ├── blocks.{0..N-1} (MoonViTEncoderLayer)
    │       │   ├── norm0/norm1 : LayerNorm
    │       │   ├── wqkv / wo   : Linear(bias=True)
    │       │   └── mlp (MLP2)  : fc0 / fc1
    │       └── final_layernorm : LayerNorm
    └── mm_projector (KimiK25MultiModalProjector)
        ├── pre_norm : LayerNorm
        ├── linear_1 / linear_2 : Linear(bias=True)
        └── act      : GELU(tanh approximation)

The ViT runs in BF16; cutlass W4A8 path only covers the LLM MoE — the
vision tower is independent.
"""

import logging
import math
from copy import deepcopy
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers.configuration_utils import PretrainedConfig

from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    ImageEmbeddingInterface,
)
from rtp_llm.utils.flash_attn_utils import can_use_flash_attn

_FLASH_ATTN_AVAILABLE = False
try:
    if can_use_flash_attn():
        from flash_attn import flash_attn_varlen_func  # type: ignore

        _FLASH_ATTN_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent
    logging.info(
        f"flash_attn unavailable for kimi_k25 vit ({exc}); falling back to SDPA"
    )


class KimiK25VisionConfig(PretrainedConfig):
    """Vision config preserving HF's `vt_*` naming used by Kimi-K2.5 ckpt."""

    model_type: str = "kimi_k25_vision"

    def __init__(
        self,
        vt_hidden_size: int = 1152,
        vt_intermediate_size: int = 4304,
        vt_num_hidden_layers: int = 27,
        vt_num_attention_heads: int = 16,
        patch_size: int = 14,
        num_channels: int = 3,
        merge_kernel_size=(2, 2),
        merge_type: str = "sd2_tpool",
        mm_projector_type: str = "patchmerger",
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        init_pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        video_attn_type: str = "spatial_temporal",
        text_hidden_size: int = 7168,
        mm_hidden_size: Optional[int] = None,
        projector_ln_eps: float = 1e-5,
        projector_hidden_act: str = "gelu",
        rope_theta: float = 10000.0,
        max_pos_emb_height: int = 512,
        max_pos_emb_width: int = 512,
        **kwargs: Any,
    ) -> None:
        self.vt_hidden_size = vt_hidden_size
        self.vt_intermediate_size = vt_intermediate_size
        self.vt_num_hidden_layers = vt_num_hidden_layers
        self.vt_num_attention_heads = vt_num_attention_heads
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.merge_kernel_size = (
            list(merge_kernel_size)
            if not isinstance(merge_kernel_size, int)
            else [merge_kernel_size, merge_kernel_size]
        )
        self.merge_type = merge_type
        self.mm_projector_type = mm_projector_type
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.init_pos_emb_time = init_pos_emb_time
        self.pos_emb_type = pos_emb_type
        self.video_attn_type = video_attn_type
        self.text_hidden_size = text_hidden_size
        # When `mm_hidden_size` is omitted in the HF config it defaults to
        # `text_hidden_size`. The HF Kimi-K2.5 ckpt sets it to 1152 (==
        # `vt_hidden_size`) which is the *projector input* size, not the
        # output. Keep the original value for serialization but expose
        # `text_hidden_size` as the canonical projector output dim.
        self.mm_hidden_size = (
            mm_hidden_size if mm_hidden_size is not None else text_hidden_size
        )
        self.projector_ln_eps = projector_ln_eps
        self.projector_hidden_act = projector_hidden_act
        self.rope_theta = rope_theta
        self.max_pos_emb_height = max_pos_emb_height
        self.max_pos_emb_width = max_pos_emb_width
        super().__init__(**kwargs)


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

        out = x + torch.cat(pos_embs).to(x.dtype)
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
        freqs_cis = torch.cat(
            [
                self.freqs_cis[:h, :w].reshape(-1, self.dim // 2).repeat(t, 1)
                for t, h, w in shapes
            ],
            dim=0,
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
    def __init__(self, config: KimiK25VisionConfig) -> None:
        super().__init__()
        ps = config.patch_size
        self.patch_size = (ps, ps)
        self.proj = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.vt_hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        if config.pos_emb_type != "divided_fixed":
            raise NotImplementedError(
                f"pos_emb_type={config.pos_emb_type} not supported"
            )
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


class MoonViTEncoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        attn_bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim

        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.wqkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=attn_bias)
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=attn_bias)
        self.mlp = MLP2(hidden_dim, mlp_dim, bias=True)

    def _attention(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        seq_length = x.size(0)
        xqkv = self.wqkv(x)
        xqkv = xqkv.view(seq_length, 3, self.num_heads, self.head_dim)
        xq, xk, xv = torch.unbind(xqkv, dim=1)
        xq, xk = apply_rope(xq, xk, rope_freqs_cis)

        if _FLASH_ATTN_AVAILABLE and xq.is_cuda:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
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
            n_seg = cu_seqlens.numel() - 1
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
                mask = torch.zeros(
                    seq_length, seq_length, device=x.device, dtype=torch.bool
                )
                cu = cu_seqlens.tolist()
                for i in range(n_seg):
                    s, e = cu[i], cu[i + 1]
                    mask[s:e, s:e] = True
                q = xq.transpose(0, 1).unsqueeze(0)
                k = xk.transpose(0, 1).unsqueeze(0)
                v = xv.transpose(0, 1).unsqueeze(0)
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                attn_out = (
                    attn_out.squeeze(0)
                    .transpose(0, 1)
                    .reshape(seq_length, self.num_heads * self.head_dim)
                )

        return self.wo(attn_out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)
        hidden_states = self._attention(hidden_states, cu_seqlens, rope_freqs_cis)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class MoonViT3dEncoder(nn.Module):
    def __init__(self, config: KimiK25VisionConfig) -> None:
        super().__init__()
        assert (
            config.video_attn_type == "spatial_temporal"
        ), f"only spatial_temporal video_attn_type supported, got {config.video_attn_type}"
        head_dim = config.vt_hidden_size // config.vt_num_attention_heads
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
                    attn_bias=True,
                )
                for _ in range(config.vt_num_hidden_layers)
            ]
        )
        self.final_layernorm = nn.LayerNorm(config.vt_hidden_size)

    def forward(
        self, hidden_states: torch.Tensor, grid_thws: torch.Tensor
    ) -> torch.Tensor:
        rope_freqs_cis = self.rope_2d.get_freqs_cis(
            grid_thws=grid_thws, device=hidden_states.device
        )
        lengths = torch.cat(
            (
                torch.zeros(1, dtype=grid_thws.dtype, device=grid_thws.device),
                grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
            )
        )
        cu_seqlens = lengths.cumsum(dim=0, dtype=torch.int32).to(hidden_states.device)
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, rope_freqs_cis)
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
    def __init__(self, config: KimiK25VisionConfig) -> None:
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


# -----------------------------------------------------------------------------
# Multimodal projector
# -----------------------------------------------------------------------------


class KimiK25MultiModalProjector(nn.Module):
    """patchmerger projector matching HF ckpt layout (`pre_norm` +
    `proj.{0,2}`).

    The two ``Linear`` layers live inside an ``nn.Sequential`` so the HF
    indices ``proj.0`` (first linear), ``proj.1`` (GELU, no params) and
    ``proj.2`` (second linear) align with the on-disk weight names.
    """

    def __init__(self, config: KimiK25VisionConfig) -> None:
        super().__init__()
        merge_h, merge_w = config.merge_kernel_size
        self.input_dim = config.vt_hidden_size * merge_h * merge_w
        self.pre_norm = nn.LayerNorm(config.vt_hidden_size, eps=config.projector_ln_eps)
        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.input_dim, config.text_hidden_size, bias=True),
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        # image_features: (n_tokens_after_merge, kh*kw, vt_hidden)
        hidden = self.pre_norm(image_features).view(-1, self.input_dim)
        return self.proj(hidden)


@torch.inference_mode()
def mm_projector_forward(
    mm_projector: KimiK25MultiModalProjector, vt_output: List[torch.Tensor]
) -> List[torch.Tensor]:
    num_embedding_list = [x.shape[0] for x in vt_output]
    batched = torch.cat(vt_output, dim=0)
    proj_out = mm_projector(batched)
    proj_out = proj_out.reshape(-1, proj_out.shape[-1])
    return list(torch.split(proj_out, num_embedding_list))


# -----------------------------------------------------------------------------
# Top-level wrapper for RTP-LLM
# -----------------------------------------------------------------------------


_DEFAULT_MEDIA_PROC_CFG = {
    "in_patch_limit": 16384,
    "patch_size": 14,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "merge_kernel_size": 2,
    "fixed_output_tokens": None,
    "patch_limit_on_one_side": 512,
    "in_patch_limit_each_frame": 4096,
    "in_patch_limit_video": None,
    "sample_fps": 2.0,
    "max_num_frames_each_video": None,
    "temporal_merge_kernel_size": 4,
    "timestamp_mode": "hh:mm:ss.fff",
}


class KimiK25ImageEmbedding(ImageEmbeddingInterface):
    """Top-level ViT + projector wrapped as MultiModalEmbeddingInterface."""

    def __init__(self, mm_related_params) -> None:
        cfg_dict = mm_related_params.config or {}
        vision_cfg_raw = cfg_dict.get("vision_config", {}) or {}
        self.vision_config = KimiK25VisionConfig(**vision_cfg_raw)
        # Attribute names match HF ckpt prefixes (`vision_tower.*`,
        # `mm_projector.*`) so weight loading via BaseVitWeights produces
        # correct ckpt key strings.
        self.vision_tower = MoonViT3dPretrainedModel(self.vision_config)
        self.mm_projector = KimiK25MultiModalProjector(self.vision_config)

        ckpt_path = cfg_dict.get("ckpt_path")
        self.image_processor = self._build_image_processor(ckpt_path)
        self.media_token_id = int(
            mm_related_params.special_token_ids.get("image_token_index", 163605)
        )

    @staticmethod
    def _build_image_processor(ckpt_path: Optional[str]):
        """Load the HF-side image processor, falling back to a built-in
        instance when ``trust_remote_code`` cannot resolve the ckpt path."""

        if ckpt_path is None:
            from rtp_llm.multimodal.multimodal_mixins.kimi_k25.kimi_k25_image_processor import (
                KimiK25VisionProcessor,
            )

            return KimiK25VisionProcessor(media_proc_cfg=_DEFAULT_MEDIA_PROC_CFG)

        try:
            from transformers import AutoImageProcessor

            return AutoImageProcessor.from_pretrained(ckpt_path, trust_remote_code=True)
        except Exception as exc:  # pragma: no cover - depends on HF cache
            logging.warning(
                f"AutoImageProcessor.from_pretrained({ckpt_path}) failed ({exc}); "
                f"falling back to local KimiK25VisionProcessor"
            )
            from rtp_llm.multimodal.multimodal_mixins.kimi_k25.kimi_k25_image_processor import (
                KimiK25VisionProcessor,
            )

            return KimiK25VisionProcessor(media_proc_cfg=_DEFAULT_MEDIA_PROC_CFG)

    @property
    def _device(self):
        return self.vision_tower.patch_embed.proj.weight.device

    @property
    def _data_type(self):
        # Source of truth is the vision tower's first conv weight dtype.
        # Used by image_embedding (line ~664) to cast pixel_values.
        return self.vision_tower.patch_embed.proj.weight.dtype

    def media_tokens_for(self, image: Image.Image) -> int:
        """Compute the number of mm_projector output tokens for an image
        without running the ViT forward."""
        media = {"type": "image", "image": image}
        return int(self.image_processor.media_tokens_calculator(media))

    @torch.inference_mode()
    def image_embedding(self, images: List[Image.Image]) -> List[torch.Tensor]:
        # Returns a list of per-image (tokens, hidden) tensors so the base
        # `ImageEmbeddingInterface.mm_process` (`image_embedding([img])[0]`)
        # yields the full token sequence for that image rather than its
        # first row.
        device = self._device
        dtype = self._data_type

        medias = [{"type": "image", "image": img} for img in images]
        proc = self.image_processor.preprocess(medias, return_tensors="pt")
        pixel_values = proc["pixel_values"].to(device).to(dtype)
        grid_thws = proc["grid_thws"].to(device)

        vt_outputs = self.vision_tower(pixel_values, grid_thws)
        return mm_projector_forward(self.mm_projector, vt_outputs)

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        # MultiModalEmbeddingInterface contract: return (image_features, pos_ids).
        # Kimi K25 is not an mrope model so pos_ids is None. preprocess_input
        # delivers a single PIL.Image (see base ImageEmbeddingInterface.preprocess_input),
        # so we wrap into a list and unpack the [0] result from image_embedding.
        return self.image_embedding([data])[0], None
