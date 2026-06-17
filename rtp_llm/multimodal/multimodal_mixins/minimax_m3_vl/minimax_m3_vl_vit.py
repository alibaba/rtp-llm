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
# Naming choice (q/k/v + out_proj vs fused qkv_proj + proj):
#   We keep the *HF checkpoint* layout — separate q_proj / k_proj / v_proj and
#   `out_proj` — so the weight loader is a straight `state_dict.update`. The
#   sglang loader fuses these into `qkv_proj` and renames `out_proj` -> `proj`,
#   but that is only required for the tensor-parallel `QKVParallelLinear` it
#   uses. In this single-GPU pure-torch port, no fusion or rename happens.

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """Per-segment SDPA with 3D RoPE applied to leading channels of q/k.

    Receives a flat [seq, hidden] tensor plus per-segment ``cu_seqlens`` and
    runs attention independently on each segment slice. This is simple and
    correct; performance optimization (flashinfer / pseudo-batched attention)
    is intentionally out of scope.
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        # Keep the HF checkpoint name `out_proj` (sglang renames to `proj`
        # only because of its TP RowParallelLinear wrapper).
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [seq, hidden]
        cu_seqlens: torch.Tensor,  # [num_segments + 1] int32
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # (cos, sin)
    ) -> torch.Tensor:
        seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(seq_len, self.num_heads, self.head_dim)

        cos, sin = position_embeddings  # [seq, 1, rot_dim]
        q, k = _apply_rope(q, k, cos, sin)

        # Per-segment SDPA.
        out = torch.empty_like(q)
        starts = cu_seqlens[:-1].tolist()
        ends = cu_seqlens[1:].tolist()
        for s, e in zip(starts, ends):
            if e == s:
                continue
            # [1, num_heads, len, head_dim]
            qs = q[s:e].transpose(0, 1).unsqueeze(0)
            ks = k[s:e].transpose(0, 1).unsqueeze(0)
            vs = v[s:e].transpose(0, 1).unsqueeze(0)
            attn = F.scaled_dot_product_attention(
                qs, ks, vs, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            # back to [len, num_heads, head_dim]
            out[s:e] = attn.squeeze(0).transpose(0, 1)

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
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, cu_seqlens, position_embeddings)
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
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, cu_seqlens, position_embeddings)
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

    def _compute_cu_seq_len(
        self, grid_thw: List[List[int]], device: torch.device
    ) -> torch.Tensor:
        """Cumulative sum of per-segment token counts (= t*h*w). Length
        num_segments + 1, leading 0."""
        seg_lens = [0] + [t * h * w for t, h, w in grid_thw]
        return (
            torch.tensor(seg_lens, device=device, dtype=torch.int32)
            .cumsum(0)
            .to(torch.int32)
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

        cu_seqlens = self._compute_cu_seq_len(grid_thw_list, hidden_states.device)
        rotary_freqs = self._get_rope_embed_3d(grid_thw_list, self.spatial_merge_size)
        assert rotary_freqs.device == hidden_states.device
        position_embeddings = self._prepare_cos_sin(rotary_freqs)

        return self.encoder(hidden_states, cu_seqlens, position_embeddings)


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
