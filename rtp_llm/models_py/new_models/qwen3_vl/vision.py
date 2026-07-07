"""Qwen3-VL vision tower (new-loader / RtpModule implementation).

Structure mirrors HF ``Qwen3VLVisionModel`` so every ckpt tensor under
``model.visual.*`` lands on a leaf with an identical shape:

  patch_embed.proj          Conv3d(3, 1024, (2,16,16), bias=True)
  pos_embed                 Embedding(2304, 1024)          # learned abs pos
  blocks.{i}.norm1/norm2    LayerNorm(1024)
  blocks.{i}.attn.qkv       Linear(1024, 3072, bias=True)
  blocks.{i}.attn.proj      Linear(1024, 1024, bias=True)
  blocks.{i}.mlp.linear_fc1 Linear(1024, 4096, bias=True)  # plain MLP (not gated)
  blocks.{i}.mlp.linear_fc2 Linear(4096, 1024, bias=True)
  merger.norm               LayerNorm(1024)                # pre-shuffle norm
  merger.linear_fc1         Linear(4096, 4096, bias=True)
  merger.linear_fc2         Linear(4096, 2560, bias=True)
  deepstack_merger_list.{j}.norm        LayerNorm(4096)    # post-shuffle norm
  deepstack_merger_list.{j}.linear_fc1  Linear(4096, 4096, bias=True)
  deepstack_merger_list.{j}.linear_fc2  Linear(4096, 2560, bias=True)

Leaves are plain ``nn.*`` modules; the ``RtpModule`` container loader copies
each tensor in via ``_assign_weight``. TP is not applied to the vision tower
(it is small and runs once per image), matching how the rest of the new-loader
vision code keeps the ViT replicated.

NOTE: the forward pass (rotary, pos-embed interpolation, DeepStack feature
collection) follows the HF reference but has NOT been numerically validated on
device — verify image outputs against HF before trusting generations.
"""

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.models_py.module_base import RtpModule


class Qwen3VLVisionPatchEmbed(RtpModule):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        patch_size: int,
        temporal_patch_size: int,
        params_dtype: torch.dtype,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size),
            bias=True,
            dtype=params_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [num_patches, in_channels * temporal_patch_size * patch_size**2]
        x = x.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        x = self.proj(x.to(self.proj.weight.dtype))
        return x.view(-1, self.hidden_size)


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def process_weights_after_loading(self):
        dim = self.inv_freq.numel() * 2
        theta = 10000.0
        device = self.inv_freq.device
        dtype = self.inv_freq.dtype
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        self.inv_freq.data.copy_(inv_freq.to(dtype))

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        return torch.outer(seq, self.inv_freq)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q.to(orig_q_dtype), k.to(orig_k_dtype)


class Qwen3VLVisionAttention(RtpModule):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        params_dtype: torch.dtype,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(
            hidden_size, 3 * hidden_size, bias=True, dtype=params_dtype
        )
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True, dtype=params_dtype)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_len = x.shape[0]
        qkv = self.qkv(x).reshape(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(1, 0, 2, 3).unbind(0)  # each [seq, heads, head_dim]

        if rotary_pos_emb is not None:
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos, sin = emb.cos(), emb.sin()
            q, k = _apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Block-diagonal full attention per image/frame via cu_seqlens.
        attn_mask = torch.zeros(
            (1, seq_len, seq_len), dtype=torch.bool, device=x.device
        )
        for i in range(1, len(cu_seqlens)):
            s, e = int(cu_seqlens[i - 1]), int(cu_seqlens[i])
            attn_mask[..., s:e, s:e] = True

        q = q.transpose(0, 1).unsqueeze(0)  # [1, heads, seq, head_dim]
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.squeeze(0).transpose(0, 1).reshape(seq_len, -1)
        return self.proj(out)


class Qwen3VLVisionMLP(RtpModule):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
    ):
        super().__init__()
        self.linear_fc1 = nn.Linear(
            hidden_size, intermediate_size, bias=True, dtype=params_dtype
        )
        self.linear_fc2 = nn.Linear(
            intermediate_size, hidden_size, bias=True, dtype=params_dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_fc1(x)
        x = F.gelu(x, approximate="tanh")
        return self.linear_fc2(x)


class Qwen3VLVisionBlock(RtpModule):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, dtype=params_dtype)
        self.norm2 = nn.LayerNorm(hidden_size, dtype=params_dtype)
        self.attn = Qwen3VLVisionAttention(hidden_size, num_heads, params_dtype)
        self.mlp = Qwen3VLVisionMLP(hidden_size, intermediate_size, params_dtype)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cu_seqlens, rotary_pos_emb)
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen3VLVisionPatchMerger(RtpModule):
    """Merges every ``spatial_merge_size**2`` neighbouring tokens.

    ``use_postshuffle_norm`` distinguishes the two ckpt variants seen in
    Qwen3-VL: the main ``merger`` norms BEFORE the spatial concat (norm dim =
    context_dim = 1024), the DeepStack mergers norm AFTER the concat (norm dim =
    context_dim * merge**2 = 4096).
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int,
        use_postshuffle_norm: bool,
        params_dtype: torch.dtype,
    ):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = self.hidden_size if use_postshuffle_norm else context_dim
        self.norm = nn.LayerNorm(norm_dim, dtype=params_dtype)
        self.linear_fc1 = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True, dtype=params_dtype
        )
        self.linear_fc2 = nn.Linear(
            self.hidden_size, dim, bias=True, dtype=params_dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)
        x = self.linear_fc1(x)
        x = F.gelu(x, approximate="tanh")
        return self.linear_fc2(x)


class Qwen3VLVisionTransformer(RtpModule):
    def __init__(self, model_config: Any, load_config: Any):
        super().__init__()
        vit_config = self._get_vit_config(model_config)
        hidden_size = vit_config.get("hidden_size", 1024)
        num_heads = vit_config.get("num_heads", 16)
        depth = vit_config.get("depth", 24)
        intermediate_size = vit_config.get("intermediate_size", 4096)
        patch_size = vit_config.get("patch_size", 16)
        temporal_patch_size = vit_config.get("temporal_patch_size", 2)
        in_channels = vit_config.get("in_channels", 3)
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.spatial_merge_size = vit_config.get("spatial_merge_size", 2)
        out_hidden_size = vit_config.get("out_hidden_size", 2560)
        self.num_position_embeddings = vit_config.get("num_position_embeddings", 2304)
        self.deepstack_visual_indexes = vit_config.get(
            "deepstack_visual_indexes", [5, 11, 17]
        )
        self.hidden_size = hidden_size

        params_dtype = (
            getattr(load_config, "compute_dtype", torch.bfloat16) or torch.bfloat16
        )

        self.patch_embed = Qwen3VLVisionPatchEmbed(
            in_channels=in_channels,
            hidden_size=hidden_size,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            params_dtype=params_dtype,
        )
        self.pos_embed = nn.Embedding(
            self.num_position_embeddings, hidden_size, dtype=params_dtype
        )
        head_dim = hidden_size // num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [
                Qwen3VLVisionBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                    params_dtype=params_dtype,
                )
                for _ in range(depth)
            ]
        )
        self.merger = Qwen3VLVisionPatchMerger(
            dim=out_hidden_size,
            context_dim=hidden_size,
            spatial_merge_size=self.spatial_merge_size,
            use_postshuffle_norm=False,
            params_dtype=params_dtype,
        )
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(
                    dim=out_hidden_size,
                    context_dim=hidden_size,
                    spatial_merge_size=self.spatial_merge_size,
                    use_postshuffle_norm=True,
                    params_dtype=params_dtype,
                )
                for _ in range(len(self.deepstack_visual_indexes))
            ]
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def get_dtype(self) -> torch.dtype:
        return self.dtype

    def get_device(self) -> torch.device:
        return self.device

    # ---- config helpers -----------------------------------------------------
    @staticmethod
    def _get_vit_config(model_config) -> dict:
        mm = getattr(model_config, "mm_related_params", None)
        if mm is not None and getattr(mm, "config", None):
            return mm.config
        if hasattr(model_config, "vision_config"):
            return model_config.vision_config
        if isinstance(model_config, dict):
            return model_config.get("vision_config", {})
        return {
            "hidden_size": 1024,
            "num_heads": 16,
            "depth": 24,
            "intermediate_size": 4096,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "in_channels": 3,
            "spatial_merge_size": 2,
            "out_hidden_size": 2560,
            "num_position_embeddings": 2304,
            "deepstack_visual_indexes": [5, 11, 17],
        }

    # ---- positional helpers -------------------------------------------------
    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        m = self.spatial_merge_size
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            hpos = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos = hpos.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
            wpos = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos = wpos.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid = int(grid_thw[:, 1:].max())
        freqs = self.rotary_pos_emb(max_grid)
        rotary = freqs[pos_ids].flatten(1)
        return rotary

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Bicubic-interpolate the learned ``num_position_embeddings`` grid to
        each image's (h, w), following HF's Qwen3-VL implementation."""
        num_grid = int(self.num_position_embeddings**0.5)
        pos_embed = self.pos_embed.weight  # [num_pos, hidden]
        outputs = []
        m = self.spatial_merge_size
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            pe = pos_embed.reshape(1, num_grid, num_grid, -1).permute(0, 3, 1, 2)
            pe = F.interpolate(
                pe.float(), size=(h, w), mode="bicubic", align_corners=False
            )
            pe = pe.permute(0, 2, 3, 1).reshape(h, w, -1)
            pe = pe.reshape(h // m, m, w // m, m, -1).permute(0, 2, 1, 3, 4)
            pe = pe.reshape(-1, pe.shape[-1]).repeat(t, 1)
            outputs.append(pe)
        return torch.cat(outputs, dim=0).to(pos_embed.dtype)

    # ---- forward ------------------------------------------------------------
    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        x = self.patch_embed(hidden_states)
        x = x + self.fast_pos_embed_interpolate(grid_thw)
        rotary_pos_emb = self.rot_pos_emb(grid_thw).to(x.device)

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists: List[torch.Tensor] = []
        deepstack_idx_map = {
            layer_idx: merger_idx
            for merger_idx, layer_idx in enumerate(self.deepstack_visual_indexes)
        }
        for layer_idx, block in enumerate(self.blocks):
            x = block(x, cu_seqlens, rotary_pos_emb)
            if layer_idx in deepstack_idx_map:
                merger = self.deepstack_merger_list[deepstack_idx_map[layer_idx]]
                deepstack_feature_lists.append(merger(x))

        x = self.merger(x)
        # DeepStack features are returned alongside the merged output so the
        # caller can inject them into the first len(indexes) LLM layers.
        return x, deepstack_feature_lists
