# SPDX-License-Identifier: Apache-2.0
# MiniMax-M3 VL vision tower (CLIP-style ViT with 3D RoPE) + multimodal
# projector + patch merger. Pure-torch port of the reference sglang
# implementation (sglang/srt/models/minimax_vl_common.py). No sglang or
# tensor-parallel utilities; runs on a single device in BF16.
#
# Naming conventions (must match the published HF checkpoint exactly):
#   * vision_tower.vision_model.embeddings.patch_embedding.weight  (Conv3d)
#   * vision_tower.vision_model.pre_layrnorm.{weight,bias}          (typo!)
#   * vision_tower.vision_model.encoder.layers.{i}.layer_norm{1,2}.{weight,bias}
#   * vision_tower.vision_model.encoder.layers.{i}.self_attn.{q,k,v}_proj.{weight,bias}
#   * vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.{weight,bias}
#   * vision_tower.vision_model.encoder.layers.{i}.mlp.fc{1,2}.{weight,bias}
#   * vision_tower.vision_model.post_layernorm.{weight,bias}
#   * multi_modal_projector.linear_{1,2}.{weight,bias}
#   * patch_merge_mlp.linear_{1,2}.{weight,bias}
#
# Runtime naming differs from the checkpoint only for QKV: the three published
# q/k/v tensors are concatenated into `qkv_proj` by the multimodal weight loader.
# `out_proj` keeps its checkpoint name.

import logging
import os
import sys
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_FA4_VARLEN_FUNC = None
try:
    from flash_attn.cute import flash_attn_varlen_func as _FA4_VARLEN_FUNC
except Exception:
    # FA4 is packaged only for CUDA 13 x86. Other builds use FA2/FA3 or SDPA.
    pass

_FLASH_ATTN_VARLEN_FUNC = None
try:
    from flash_attn import flash_attn_varlen_func as _FLASH_ATTN_VARLEN_FUNC
except Exception:
    pass

_FLASHINFER_RAGGED_WRAPPER = None
_FLASHINFER_IMPORT_ERROR = None
try:
    # Bazel does not process the wheel's .pth file for subprocesses.
    import nvidia_cutlass_dsl

    for package_dir in nvidia_cutlass_dsl.__path__:
        python_packages_dir = os.path.join(package_dir, "python_packages")
        if os.path.isdir(python_packages_dir) and python_packages_dir not in sys.path:
            sys.path.insert(0, python_packages_dir)
    from flashinfer.prefill import (
        BatchPrefillWithRaggedKVCacheWrapper as _FLASHINFER_RAGGED_WRAPPER,
    )
except Exception as error:
    _FLASHINFER_IMPORT_ERROR = f"{type(error).__name__}: {error}"

try:
    from rtp_llm.multimodal.multimodal_mixins.minimax_m3_vl.minimax_m3_vl_rope import (
        fused_qkv_rope,
    )
except Exception:
    fused_qkv_rope = None


def get_fused_qkv_checkpoint_names(
    parameter_name: str,
) -> Optional[Tuple[str, str, str]]:
    """Map a live fused-QKV parameter to the three published checkpoint keys."""
    marker = "qkv_proj."
    if marker not in parameter_name:
        return None
    prefix, suffix = parameter_name.rsplit(marker, 1)
    if suffix not in ("weight", "bias"):
        return None
    return tuple(
        f"{prefix}{projection}_proj.{suffix}" for projection in ("q", "k", "v")
    )


def _select_attention_backend(tensor: torch.Tensor) -> str:
    if not tensor.is_cuda:
        return "sdpa"
    capability = torch.cuda.get_device_capability(tensor.device)
    if _FA4_VARLEN_FUNC is not None and capability in ((9, 0), (10, 0), (11, 0)):
        return "fa4"
    if _FLASH_ATTN_VARLEN_FUNC is not None and capability[0] in (8, 9):
        return "flash_attn"
    if _FLASHINFER_RAGGED_WRAPPER is not None:
        return "flashinfer"
    return "sdpa"


@dataclass
class PackedAttentionContext:
    backend: str
    flashinfer_wrapper: Optional[Any] = None
    fallback_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class VisionConfig:
    """Vision-tower hyperparameters. Mirrors sglang's CLIPVisionConfig.

    Defaults match the MiniMax-M3 preview checkpoint's ``vision_config`` block
    (see ``config.json``); any field can be overridden via ``from_dict``.
    """

    hidden_size: int = 1280
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    intermediate_size: int = 5120
    patch_size: int = 14
    image_size: int = 2016
    num_channels: int = 3
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    position_embedding_type: str = "rope"
    rope_mode: str = "3d"
    rope_theta: float = 10000.0
    vision_segment_max_frames: Optional[int] = 4
    img_token_compression_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "image_token_compression_method": "patch_merge",
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        }
    )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VisionConfig":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ---------------------------------------------------------------------------
# RoPE helpers (3D)
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dim halves: (a, b) -> (-b, a)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embedding to leading ``rot_dim`` channels of q/k.

    Shapes:
        q, k : [seq, num_heads, head_dim]
        cos, sin : [seq, 1, rot_dim]   (rot_dim may be < head_dim)
    """
    rot_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rot_dim].float(), q[..., rot_dim:]
    k_rot, k_pass = k[..., :rot_dim].float(), k[..., rot_dim:]

    q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)

    q = torch.cat((q_rot.to(q_pass.dtype), q_pass), dim=-1)
    k = torch.cat((k_rot.to(k_pass.dtype), k_pass), dim=-1)
    return q, k


def _prepare_qkv(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if fused_qkv_rope is not None:
        fused = fused_qkv_rope(qkv, cos, sin)
        if fused is not None:
            return fused

    q, k, v = qkv.unbind(dim=1)
    q, k = _apply_rope(q, k, cos, sin)
    return q, k, v.contiguous()


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class CLIPVisionEmbeddings(nn.Module):
    """Conv3d patch embedder. Input is the *flattened* patch tensor.

    Input  : [N_total_patches, num_channels * temporal_patch_size * patch_size * patch_size]
    Output : [N_total_patches, hidden_size]
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.input_num_channels = config.num_channels
        self.temporal_patch_size = config.img_token_compression_config.get(
            "temporal_patch_size", 2
        )

        self.patch_embedding = nn.Conv3d(
            in_channels=self.input_num_channels,
            out_channels=self.embed_dim,
            kernel_size=(self.temporal_patch_size, self.patch_size, self.patch_size),
            stride=(self.temporal_patch_size, self.patch_size, self.patch_size),
            bias=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        assert (
            pixel_values.dim() == 2
        ), f"pixel_values must be 2D, got {pixel_values.dim()}D"

        # Cast Conv3d weights to match input dtype (typically bf16).
        if self.patch_embedding.weight.dtype != pixel_values.dtype:
            self.patch_embedding = self.patch_embedding.to(pixel_values.dtype)

        x = pixel_values.reshape(
            pixel_values.shape[0],
            self.input_num_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        x = self.patch_embedding(x)  # [N, hidden_size, 1, 1, 1]
        return x.reshape(x.shape[0], -1)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class CLIPAttention(nn.Module):
    """Fused-QKV packed vision attention with a segmented SDPA fallback."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=True)
        # Keep the HF checkpoint name `out_proj` (sglang renames to `proj`
        # only because of its TP RowParallelLinear wrapper).
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.last_backend = "uninitialized"
        self.last_backend_error: Optional[str] = None

    @staticmethod
    def _segmented_sdpa(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        segment_offsets: Sequence[int],
    ) -> torch.Tensor:
        out = torch.empty_like(q)
        for start, end in zip(segment_offsets[:-1], segment_offsets[1:]):
            if end == start:
                continue
            qs = q[start:end].transpose(0, 1).unsqueeze(0)
            ks = k[start:end].transpose(0, 1).unsqueeze(0)
            vs = v[start:end].transpose(0, 1).unsqueeze(0)
            attn = F.scaled_dot_product_attention(
                qs,
                ks,
                vs,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            out[start:end] = attn.squeeze(0).transpose(0, 1)
        return out

    def _packed_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        segment_offsets: Sequence[int],
        attention_context: PackedAttentionContext,
    ) -> torch.Tensor:
        backend = attention_context.backend
        self.last_backend = backend
        self.last_backend_error = attention_context.fallback_reason
        if backend == "fa4":
            out = _FA4_VARLEN_FUNC(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=False,
                softmax_scale=self.scaling,
            )
            return out[0] if isinstance(out, tuple) else out
        if backend == "flashinfer":
            wrapper = attention_context.flashinfer_wrapper
            if wrapper is not None:
                try:
                    out = wrapper.run(q, k, v)
                    return out[0] if isinstance(out, tuple) else out
                except (AssertionError, RuntimeError, ValueError) as error:
                    if "out of memory" in str(error).lower():
                        raise
                    logger.warning(
                        "FlashInfer vision attention unavailable; "
                        "falling back to segmented SDPA: %s",
                        error,
                    )
                    attention_context.fallback_reason = (
                        f"FlashInfer run failed: {type(error).__name__}: {error}"
                    )
            attention_context.backend = "sdpa"
            self.last_backend = "sdpa"
            self.last_backend_error = attention_context.fallback_reason
        if backend == "flash_attn":
            out = _FLASH_ATTN_VARLEN_FUNC(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=0.0,
                softmax_scale=self.scaling,
                causal=False,
            )
            return out[0] if isinstance(out, tuple) else out
        return self._segmented_sdpa(q, k, v, segment_offsets)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [seq, hidden]
        cu_seqlens: torch.Tensor,  # [num_segments + 1] int32
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # (cos, sin)
        max_seqlen: int,
        segment_offsets: Sequence[int],
        attention_context: Optional[PackedAttentionContext] = None,
    ) -> torch.Tensor:
        seq_len, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states).view(
            seq_len, 3, self.num_heads, self.head_dim
        )
        cos, sin = position_embeddings  # [seq, 1, rot_dim]
        q, k, v = _prepare_qkv(qkv, cos, sin)

        if attention_context is None:
            attention_context = PackedAttentionContext(
                backend=_select_attention_backend(q)
            )
        out = self._packed_attention(
            q,
            k,
            v,
            cu_seqlens,
            max_seqlen,
            segment_offsets,
            attention_context,
        )
        out = out.reshape(seq_len, self.embed_dim)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# MLP / Encoder layer / Encoder
# ---------------------------------------------------------------------------


class CLIPMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        assert (
            config.hidden_act == "gelu"
        ), f"Only gelu activation is supported, got {config.hidden_act}"
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = CLIPAttention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        max_seqlen: int,
        segment_offsets: Sequence[int],
        attention_context: PackedAttentionContext,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            cu_seqlens,
            position_embeddings,
            max_seqlen,
            segment_offsets,
            attention_context,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class CLIPEncoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        max_seqlen: int,
        segment_offsets: Sequence[int],
        attention_context: PackedAttentionContext,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cu_seqlens,
                position_embeddings,
                max_seqlen,
                segment_offsets,
                attention_context,
            )
        return hidden_states


# ---------------------------------------------------------------------------
# Vision model (embeddings + pre_layrnorm + encoder)
# ---------------------------------------------------------------------------


class MiniMaxM3VLVisionModel(nn.Module):
    """The `vision_model` submodule: patch embed + pre-norm + transformer.

    Forward signature:
        pixel_values : bf16 [N_total_patches, C * T * P * P]
        grid_thw     : long [num_images, 3]  (one (t, h, w) per image/video segment)
    Returns:
        hidden_states : [N_total_patches, hidden_size]
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.temporal_patch_size = config.img_token_compression_config.get(
            "temporal_patch_size", 2
        )
        self.spatial_merge_size = config.img_token_compression_config.get(
            "spatial_merge_size", 2
        )

        self.embeddings = CLIPVisionEmbeddings(config)
        # NOTE: typo `pre_layrnorm` preserved to match HF ckpt key.
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self._flashinfer_workspace: Optional[torch.Tensor] = None
        self._flashinfer_wrapper: Optional[Any] = None

        assert (
            config.position_embedding_type == "rope"
        ), "Only rope position embedding is supported"
        assert config.rope_mode == "3d", "Only 3D RoPE is supported"
        self.vision_segment_max_frames = config.vision_segment_max_frames

        head_dim = embed_dim // config.num_attention_heads
        rope_dims = 2 * (head_dim // 2)
        # Split rope dims evenly across t / h / w (each forced even).
        # For head_dim=80: rope_dims=80, t=h=w=26, passthrough=2 channels.
        self.t_dim = int(2 * ((rope_dims // 3) // 2))
        self.h_dim = int(2 * ((rope_dims // 3) // 2))
        self.w_dim = int(2 * ((rope_dims // 3) // 2))

        inv_freq_t = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, self.t_dim, 2, dtype=torch.float32) / self.t_dim)
        )
        inv_freq_h = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, self.h_dim, 2, dtype=torch.float32) / self.h_dim)
        )
        inv_freq_w = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, self.w_dim, 2, dtype=torch.float32) / self.w_dim)
        )
        self.register_buffer("inv_freq_t", inv_freq_t, persistent=False)
        self.register_buffer("inv_freq_h", inv_freq_h, persistent=False)
        self.register_buffer("inv_freq_w", inv_freq_w, persistent=False)

    # ------------------------------------------------------------------ rope

    def _get_3d_rope_embed(
        self, grid_t: int, grid_h: int, grid_w: int, merge: int
    ) -> torch.Tensor:
        """Per-token (t, h, w) position ids, with spatial axes permuted into
        merge-block order so that the patch_merger's reshape groups
        spatially-adjacent tokens.
        """
        device = self.inv_freq_t.device
        tokens_per_frame = grid_h * grid_w

        tpos_ids = (
            torch.arange(grid_t, device=device)
            .unsqueeze(1)
            .expand(-1, tokens_per_frame)
            .flatten()
        )

        hpos_ids = torch.arange(grid_h, device=device).unsqueeze(1).expand(-1, grid_w)
        hpos_ids = hpos_ids.reshape(
            grid_h // merge, merge, grid_w // merge, merge
        ).permute(0, 2, 1, 3)
        hpos_ids = hpos_ids.unsqueeze(0).expand(grid_t, -1, -1, -1, -1).flatten()

        wpos_ids = torch.arange(grid_w, device=device).unsqueeze(0).expand(grid_h, -1)
        wpos_ids = wpos_ids.reshape(
            grid_h // merge, merge, grid_w // merge, merge
        ).permute(0, 2, 1, 3)
        wpos_ids = wpos_ids.unsqueeze(0).expand(grid_t, -1, -1, -1, -1).flatten()

        max_t = max(grid_t, 1)
        max_hw = max(grid_h, grid_w)
        seq_t = torch.arange(max_t, device=device, dtype=self.inv_freq_t.dtype)
        seq_hw = torch.arange(max_hw, device=device, dtype=self.inv_freq_h.dtype)

        freqs_t = torch.outer(seq_t, self.inv_freq_t)  # [max_t, t_dim/2]
        freqs_h = torch.outer(seq_hw, self.inv_freq_h)  # [max_hw, h_dim/2]
        freqs_w = torch.outer(seq_hw, self.inv_freq_w)  # [max_hw, w_dim/2]

        emb_t = freqs_t[tpos_ids]
        emb_h = freqs_h[hpos_ids]
        emb_w = freqs_w[wpos_ids]
        return torch.cat([emb_t, emb_h, emb_w], dim=-1)  # [N, rope_dim/2]

    def _get_rope_embed_3d(self, grid_thw: List[List[int]], merge: int) -> torch.Tensor:
        return torch.cat(
            [self._get_3d_rope_embed(t, h, w, merge) for t, h, w in grid_thw], dim=0
        )

    @staticmethod
    def _prepare_cos_sin(
        freqs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """[seq, rope_dim/2] -> (cos, sin) each [seq, 1, rope_dim] in float32."""
        cos = freqs.cos().repeat(1, 2).unsqueeze(-2).float()
        sin = freqs.sin().repeat(1, 2).unsqueeze(-2).float()
        return cos, sin

    # ------------------------------------------------------------------ segmentation

    def _apply_max_frames_limit(
        self, origin_grid_thw: List[List[int]]
    ) -> List[List[int]]:
        """Split any segment with grid_t > vision_segment_max_frames into
        consecutive chunks of at most that many frames."""
        if self.vision_segment_max_frames is None:
            return [list(g) for g in origin_grid_thw]
        max_frames = self.vision_segment_max_frames
        out: List[List[int]] = []
        for grid_t, grid_h, grid_w in origin_grid_thw:
            if grid_t <= max_frames:
                out.append([grid_t, grid_h, grid_w])
            else:
                for i in range(0, grid_t, max_frames):
                    sub_t = min(max_frames, grid_t - i)
                    out.append([sub_t, grid_h, grid_w])
        return out

    def _compute_attention_metadata(
        self, grid_thw: List[List[int]], device: torch.device
    ) -> Tuple[torch.Tensor, int, Tuple[int, ...]]:
        """Build varlen metadata once and reuse it in every encoder layer."""
        segment_lengths = [t * h * w for t, h, w in grid_thw]
        segment_offsets = [0]
        for length in segment_lengths:
            segment_offsets.append(segment_offsets[-1] + length)
        return (
            torch.tensor(segment_offsets, device=device, dtype=torch.int32),
            max(segment_lengths, default=0),
            tuple(segment_offsets),
        )

    def _prepare_attention_context(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> PackedAttentionContext:
        backend = _select_attention_backend(hidden_states)
        if backend != "flashinfer":
            fallback_reason = None
            if (
                backend == "sdpa"
                and hidden_states.is_cuda
                and _FLASHINFER_IMPORT_ERROR is not None
            ):
                fallback_reason = (
                    f"FlashInfer import failed: {_FLASHINFER_IMPORT_ERROR}"
                )
            return PackedAttentionContext(
                backend=backend,
                fallback_reason=fallback_reason,
            )

        try:
            if (
                self._flashinfer_workspace is None
                or self._flashinfer_workspace.device != hidden_states.device
            ):
                self._flashinfer_workspace = torch.empty(
                    128 * 1024 * 1024,
                    dtype=torch.uint8,
                    device=hidden_states.device,
                )
                self._flashinfer_wrapper = _FLASHINFER_RAGGED_WRAPPER(
                    self._flashinfer_workspace,
                    kv_layout="NHD",
                    backend="cute-dsl",
                )

            assert self._flashinfer_wrapper is not None
            head_dim = self.config.hidden_size // self.config.num_attention_heads
            self._flashinfer_wrapper.plan(
                cu_seqlens,
                cu_seqlens,
                self.config.num_attention_heads,
                self.config.num_attention_heads,
                head_dim,
                head_dim,
                causal=False,
                sm_scale=head_dim**-0.5,
                q_data_type=hidden_states.dtype,
                kv_data_type=hidden_states.dtype,
                o_data_type=hidden_states.dtype,
            )
            return PackedAttentionContext(
                backend=backend,
                flashinfer_wrapper=self._flashinfer_wrapper,
            )
        except (AssertionError, RuntimeError, ValueError) as error:
            if "out of memory" in str(error).lower():
                raise
            logger.warning(
                "Failed to plan FlashInfer vision attention; "
                "falling back to segmented SDPA: %s",
                error,
            )
            return PackedAttentionContext(
                backend="sdpa",
                fallback_reason=(
                    f"FlashInfer plan failed: {type(error).__name__}: {error}"
                ),
            )

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        assert pixel_values.dtype == torch.bfloat16, "pixel_values must be bfloat16"

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        # Accept either a long tensor [N, 3] or a python list of lists.
        if isinstance(grid_thw, torch.Tensor):
            grid_thw_list: List[List[int]] = grid_thw.tolist()
        else:
            grid_thw_list = [list(g) for g in grid_thw]
        grid_thw_list = self._apply_max_frames_limit(grid_thw_list)

        cu_seqlens, max_seqlen, segment_offsets = self._compute_attention_metadata(
            grid_thw_list, hidden_states.device
        )
        if segment_offsets[-1] != hidden_states.shape[0]:
            raise ValueError(
                "grid_thw token count does not match pixel_values: "
                f"{segment_offsets[-1]} != {hidden_states.shape[0]}"
            )
        rotary_freqs = self._get_rope_embed_3d(grid_thw_list, self.spatial_merge_size)
        assert rotary_freqs.device == hidden_states.device
        position_embeddings = self._prepare_cos_sin(rotary_freqs)
        attention_context = self._prepare_attention_context(hidden_states, cu_seqlens)

        hidden_states = self.encoder(
            hidden_states,
            cu_seqlens,
            position_embeddings,
            max_seqlen,
            segment_offsets,
            attention_context,
        )
        logger.debug(
            "MiniMax M3VL vision attention backend=%s segments=%d max_seqlen=%d",
            self.encoder.layers[0].self_attn.last_backend,
            len(segment_offsets) - 1,
            max_seqlen,
        )
        return hidden_states


# ---------------------------------------------------------------------------
# Projector + patch merger
# ---------------------------------------------------------------------------


class MiniMaxVLMultiModalProjector(nn.Module):
    """vision_hidden -> mid -> text_hidden, with GELU between."""

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        projector_hidden_act: str = "gelu",
        multimodal_projector_bias: bool = True,
        projector_hidden_size: Optional[int] = None,
    ):
        super().__init__()
        assert (
            projector_hidden_act == "gelu"
        ), f"Only gelu activation is supported, got {projector_hidden_act}"
        mid = (
            projector_hidden_size
            if projector_hidden_size is not None
            else text_hidden_size
        )
        self.linear_1 = nn.Linear(
            vision_hidden_size, mid, bias=multimodal_projector_bias
        )
        self.linear_2 = nn.Linear(mid, text_hidden_size, bias=multimodal_projector_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.gelu(self.linear_1(x)))


class MiniMaxVLPatchMerger(nn.Module):
    """Merge (spatial_merge_size**2) adjacent tokens by reshape, then MLP."""

    def __init__(
        self,
        spatial_merge_size: int,
        text_hidden_size: int,
        projector_hidden_act: str = "gelu",
        patch_merge_bias: bool = True,
        projector_hidden_size: Optional[int] = None,
    ):
        super().__init__()
        assert (
            projector_hidden_act == "gelu"
        ), f"Only gelu activation is supported, got {projector_hidden_act}"
        self.spatial_merge_size = spatial_merge_size
        mid = (
            projector_hidden_size
            if projector_hidden_size is not None
            else text_hidden_size
        )
        in_dim = text_hidden_size * spatial_merge_size**2
        self.linear_1 = nn.Linear(in_dim, mid, bias=patch_merge_bias)
        self.linear_2 = nn.Linear(mid, text_hidden_size, bias=patch_merge_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0] // (self.spatial_merge_size**2), -1)
        return self.linear_2(F.gelu(self.linear_1(x)))


# ---------------------------------------------------------------------------
# Composite vision tower
# ---------------------------------------------------------------------------


class _VisionTowerWrapper(nn.Module):
    """Thin wrapper that exposes ``vision_model`` as an attribute named
    ``vision_tower.vision_model`` in the module hierarchy.

    Exists purely to make the live PyTorch tree match the on-disk HF
    checkpoint hierarchy (top-level ``vision_tower.vision_model.*`` keys),
    so the rtp-llm weight loader can map on-disk names straight onto live
    tensors via getattr walks without any prefix translation.
    """

    def __init__(self, vision_config: VisionConfig):
        super().__init__()
        self.vision_model = MiniMaxM3VLVisionModel(vision_config)


class MiniMaxM3VLVisionTower(nn.Module):
    """Composite module: ``vision_model`` + ``multi_modal_projector`` + ``patch_merge_mlp``.

    Constructor accepts the full HF top-level config (dict-like or object); it
    pulls out the ``vision_config`` block plus the top-level multimodal
    projector / patch merger knobs.
    """

    def __init__(self, config: Any):
        super().__init__()

        # Allow either an object with attributes (HF PretrainedConfig style)
        # or a plain dict.
        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        vision_raw = _get(config, "vision_config", None)
        assert vision_raw is not None, "vision_config is required"
        if hasattr(vision_raw, "to_dict"):
            vision_dict = vision_raw.to_dict()
        elif isinstance(vision_raw, dict):
            vision_dict = vision_raw
        else:
            # PretrainedConfig-ish: dump attributes.
            vision_dict = {
                k: v for k, v in vars(vision_raw).items() if not k.startswith("_")
            }
        vision_config = VisionConfig.from_dict(vision_dict)
        self.vision_config = vision_config

        text_config = _get(config, "text_config", None)
        text_hidden_size = (
            _get(text_config, "hidden_size", None) if text_config is not None else None
        )
        if text_hidden_size is None:
            text_hidden_size = _get(config, "hidden_size", None)
        assert text_hidden_size is not None, "text_hidden_size is required"

        projector_hidden_size = _get(config, "projector_hidden_size", None)
        projector_hidden_act = _get(config, "projector_hidden_act", "gelu")
        multimodal_projector_bias = _get(config, "multimodal_projector_bias", True)
        patch_merge_bias = _get(config, "patch_merge_bias", True)

        # NOTE: the HF checkpoint top-level keys are
        #   vision_tower.vision_model.*  /  multi_modal_projector.*  /  patch_merge_mlp.*
        # We mirror that hierarchy literally — `vision_tower` is a thin wrapper
        # module whose sole child is `vision_model`. Keeping the live tree
        # structurally isomorphic to the on-disk tree lets BaseVitWeights emit
        # weight_names that round-trip back to live tensors via plain getattr
        # walks, with no prefix gymnastics (see qwen3_5_moe_mixin.py for the
        # same pattern).
        self.vision_tower = _VisionTowerWrapper(vision_config)
        self.multi_modal_projector = MiniMaxVLMultiModalProjector(
            vision_hidden_size=vision_config.hidden_size,
            text_hidden_size=text_hidden_size,
            projector_hidden_act=projector_hidden_act,
            multimodal_projector_bias=multimodal_projector_bias,
            projector_hidden_size=projector_hidden_size,
        )
        spatial_merge_size = vision_config.img_token_compression_config.get(
            "spatial_merge_size", 2
        )
        self.spatial_merge_size = spatial_merge_size
        self.patch_merge_mlp = MiniMaxVLPatchMerger(
            spatial_merge_size=spatial_merge_size,
            text_hidden_size=text_hidden_size,
            projector_hidden_act=projector_hidden_act,
            patch_merge_bias=patch_merge_bias,
            projector_hidden_size=projector_hidden_size,
        )

        self.out_hidden_size = text_hidden_size
        self.dtype = (
            self.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.vision_tower.vision_model(
            pixel_values=pixel_values, grid_thw=grid_thw
        )
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.squeeze(0)
        hidden_states = self.multi_modal_projector(hidden_states)
        hidden_states = self.patch_merge_mlp(hidden_states)
        return hidden_states
