"""MiniCPM-V-4.7 vision modules ported from transformers v5.18.0 reference
(runs/minicpmv47_h20_20261013/ref_code/minicpmv4_7/modeling_minicpmv4_7.py).

The tower is a SigLIP-style NaViT encoder (packed variable-resolution patches)
with a 2x2 window-attention merger applied inside the encoder at
``insert_layer_id`` (config default 6), followed by a final per-crop 2x2
``Merger`` MLP that projects into the LLM hidden size.  ``16x`` mode applies
both merges (patch->LLM token factor 16); ``4x`` skips the in-tower merger.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

try:
    from rtp_llm.utils.flash_attn_utils import can_use_flash_attn

    if can_use_flash_attn():
        from flash_attn import flash_attn_varlen_func

        _HAS_FLASH_ATTN = True
    else:
        _HAS_FLASH_ATTN = False
except Exception:
    # Standalone import (e.g. offline vision-port verification without the
    # built rtp_llm ops): fall back to a plain flash_attn probe.
    try:
        from flash_attn import flash_attn_varlen_func

        _HAS_FLASH_ATTN = True
    except Exception:
        _HAS_FLASH_ATTN = False


def get_vision_nearest_position_ids(
    target_sizes: torch.Tensor, num_patches_per_side: int
) -> torch.Tensor:
    """Nearest-bucket SigLIP position ids for NaViT packed patches.

    Port of transformers ``get_vision_nearest_position_ids``: fractional grid
    coordinates ``i/h`` bucketize onto the fixed pretraining grid.
    """
    device = target_sizes.device
    boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side, device=device)
    pos_ids_list = []
    for height, width in target_sizes.tolist():
        height, width = int(height), int(width)
        h_coords = torch.arange(height, device=device) / height
        w_coords = torch.arange(width, device=device) / width
        bucket_h = torch.bucketize(h_coords, boundaries, right=True)
        bucket_w = torch.bucketize(w_coords, boundaries, right=True)
        pos_ids_list.append((bucket_h[:, None] * num_patches_per_side + bucket_w).flatten())
    return torch.cat(pos_ids_list)


def get_vision_window_index(
    grid_thw: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Window permutation index for the in-tower 2x2 merger.

    Port of transformers ``get_vision_window_index`` for
    spatial_merge_size=1, patch_size=1: per crop, windows are emitted
    window-major; returns ``(window_index, cu_window_seqlens)`` where
    ``cu_window_seqlens`` are per-window token boundaries (varlen attention
    treats each window as one sequence).
    """
    device = grid_thw.device
    window_index: List[torch.Tensor] = []
    cu_window_seqlens: List[int] = [0]
    window_index_id = 0
    for grid_t, grid_h, grid_w in grid_thw.tolist():
        grid_t, grid_h, grid_w = int(grid_t), int(grid_h), int(grid_w)
        index = torch.arange(grid_t * grid_h * grid_w, device=device).reshape(grid_t, grid_h, grid_w)
        pad_h = window_size - grid_h % window_size
        pad_w = window_size - grid_w % window_size
        num_windows_h = (grid_h + pad_h) // window_size
        num_windows_w = (grid_w + pad_w) // window_size
        index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        index_padded = index_padded.reshape(
            grid_t, num_windows_h, window_size, num_windows_w, window_size
        )
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t, num_windows_h * num_windows_w, window_size, window_size
        )
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        window_index.append(index_new + window_index_id)
        cu_window_seqlens.extend((seqlens.cumsum(0) + cu_window_seqlens[-1]).tolist())
        window_index_id += grid_t * grid_h * grid_w
    window_index = torch.cat(window_index, dim=0)
    cu_window_seqlens = torch.tensor(cu_window_seqlens, device=device, dtype=torch.int32)
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
    return window_index, cu_window_seqlens


@dataclass
class MiniCPMV47VisionConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 980
    patch_size: int = 14
    layer_norm_eps: float = 1e-6
    insert_layer_id: int = 6
    window_kernel_size: Tuple[int, int] = (2, 2)
    downsample_mode: str = "16x"

    @property
    def window_hidden_size(self) -> int:
        return self.hidden_size * self.window_kernel_size[0] * self.window_kernel_size[1]

    @property
    def window_intermediate_size(self) -> int:
        return self.intermediate_size * self.window_kernel_size[0] * self.window_kernel_size[1]

    @classmethod
    def from_checkpoint_config(cls, vision_config: dict, model_config: dict) -> "MiniCPMV47VisionConfig":
        return cls(
            hidden_size=vision_config["hidden_size"],
            intermediate_size=vision_config["intermediate_size"],
            num_hidden_layers=vision_config["num_hidden_layers"],
            num_attention_heads=vision_config["num_attention_heads"],
            num_channels=vision_config.get("num_channels", 3),
            image_size=vision_config["image_size"],
            patch_size=vision_config["patch_size"],
            layer_norm_eps=vision_config.get("layer_norm_eps", 1e-6),
            insert_layer_id=model_config.get("insert_layer_id", 6),
            window_kernel_size=tuple(vision_config.get("window_kernel_size", (2, 2))),
            downsample_mode=model_config.get("downsample_mode", "16x"),
        )


def _nearest_position_ids(target_sizes: torch.Tensor, num_patches_per_side: int) -> torch.Tensor:
    """SigLIP NaViT position ids for packed patches (ported authority above)."""
    return get_vision_nearest_position_ids(target_sizes, num_patches_per_side)


def _window_partition_index(target_sizes: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Window permutation index for the in-tower merger (ported authority above)."""
    grid_thw = F.pad(target_sizes.to(torch.long), (1, 0), value=1)
    return get_vision_window_index(grid_thw, window_size)


class MiniCPMV47VisionAttention(nn.Module):
    def __init__(self, config: MiniCPMV47VisionConfig):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        if self.head_dim * self.num_heads != self.dim:
            raise ValueError(f"embed_dim {self.dim} not divisible by num_heads {self.num_heads}")
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        # hidden_states: [1, seq, dim]; attention is crop-local (varlen, non-causal).
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        if _HAS_FLASH_ATTN and cu_seqlens.numel() > 1:
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
            # varlen layout: [total_seq, heads, head_dim]
            q_flat = q.transpose(1, 2).reshape(-1, self.num_heads, self.head_dim)
            k_flat = k.transpose(1, 2).reshape(-1, self.num_heads, self.head_dim)
            v_flat = v.transpose(1, 2).reshape(-1, self.num_heads, self.head_dim)
            attn_output = flash_attn_varlen_func(
                q_flat,
                k_flat,
                v_flat,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=self.scaling,
                causal=False,
            )
            attn_output = attn_output.view(*input_shape, -1)
        else:
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            if lengths and lengths[0] == q.shape[2] and len(lengths) == 1:
                attn_output = F.scaled_dot_product_attention(q, k, v, scale=self.scaling)
            else:
                outputs = []
                for q_s, k_s, v_s in zip(
                    torch.split(q, lengths, dim=2), torch.split(k, lengths, dim=2), torch.split(v, lengths, dim=2)
                ):
                    outputs.append(F.scaled_dot_product_attention(q_s, k_s, v_s, scale=self.scaling))
                attn_output = torch.cat(outputs, dim=2)
            attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1)
        attn_output = self.out_proj(attn_output)
        return attn_output


class MiniCPMV47ViTWindowAttentionMerger(nn.Module):
    """Window attention + 2x2 merge applied mid-tower (at insert_layer_id)."""

    def __init__(self, config: MiniCPMV47VisionConfig):
        super().__init__()
        self.window_kernel_size = tuple(config.window_kernel_size)
        self.embed_dim = config.hidden_size
        self.self_attn = MiniCPMV47VisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.pre_norm = nn.LayerNorm(config.window_hidden_size, eps=config.layer_norm_eps)
        self.linear_1 = nn.Linear(config.window_hidden_size, config.window_intermediate_size, bias=True)
        self.linear_2 = nn.Linear(config.window_intermediate_size, self.embed_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor, target_sizes: torch.Tensor) -> torch.Tensor:
        # hidden_states: [1, seq, dim] over the packed un-merged patches.
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        device = hidden_states.device
        window_h, window_w = self.window_kernel_size
        window_size = window_h * window_w
        window_index, window_cu_seqlens = _window_partition_index(target_sizes, window_h)
        window_index = window_index.to(device)
        window_cu_seqlens = window_cu_seqlens.to(device)

        hidden_states = hidden_states[:, window_index, :]
        hidden_states = self.self_attn(hidden_states, window_cu_seqlens)
        hidden_states = residual[:, window_index, :] + hidden_states

        if window_cu_seqlens.numel() - 1 != hidden_states.shape[1] // window_size:
            raise ValueError(
                f"patch grids {target_sizes.tolist()} must be divisible by window kernel "
                f"{self.window_kernel_size}"
            )
        embed_dim = hidden_states.shape[-1]
        patch = hidden_states.reshape(-1, window_size, embed_dim)
        flat = patch.flatten(1)
        patch_residual = patch.mean(dim=1)

        hidden_state = self.pre_norm(flat)
        hidden_state = self.linear_1(hidden_state)
        hidden_state = F.gelu(hidden_state, approximate="tanh")
        hidden_state = self.linear_2(hidden_state)
        return (hidden_state + patch_residual).unsqueeze(0)


class MiniCPMV47VisionEmbeddings(nn.Module):
    def __init__(self, config: MiniCPMV47VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )
        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    def forward(self, pixel_values: torch.Tensor, target_sizes: torch.Tensor) -> torch.Tensor:
        # pixel_values: [1, C, patch, packed_len] (NaViT); Conv2d patchifies columns.
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        pos_ids = _nearest_position_ids(target_sizes, self.num_patches_per_side)
        pos_ids = pos_ids.to(self.position_embedding.weight.device)
        position_embeddings = self.position_embedding(pos_ids).unsqueeze(0)
        return embeddings + position_embeddings


class MiniCPMV47VisionMLP(nn.Module):
    def __init__(self, config: MiniCPMV47VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        return self.fc2(hidden_states)


class MiniCPMV47VisionEncoderLayer(nn.Module):
    def __init__(self, config: MiniCPMV47VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = MiniCPMV47VisionAttention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = MiniCPMV47VisionMLP(config)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, cu_seqlens)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class MiniCPMV47VisionEncoder(nn.Module):
    """Keeps the SigLIP ``encoder.layers.N.*`` state-dict naming."""

    def __init__(self, config: MiniCPMV47VisionConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [MiniCPMV47VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )


class MiniCPMV47VisionModel(nn.Module):
    def __init__(self, config: MiniCPMV47VisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = MiniCPMV47VisionEmbeddings(config)
        self.encoder = MiniCPMV47VisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.vit_merger = MiniCPMV47ViTWindowAttentionMerger(config)

    @staticmethod
    def _cu_seqlens(target_sizes: torch.Tensor, device: torch.device) -> torch.Tensor:
        seq = (target_sizes[:, 0] * target_sizes[:, 1]).to(torch.int32)
        return F.pad(torch.cumsum(seq, dim=0, dtype=torch.int32).to(device), (1, 0))

    def forward(
        self,
        pixel_values: torch.Tensor,
        target_sizes: torch.Tensor,
        use_vit_merger: bool = True,
    ) -> torch.Tensor:
        """Returns last_hidden_state [1, seq, hidden]; when use_vit_merger the
        sequence is already 2x2-merged and the caller must halve target_sizes."""
        hidden_states = self.embeddings(pixel_values, target_sizes)
        cu_seqlens = self._cu_seqlens(target_sizes, hidden_states.device)

        insert_layer_id = self.config.insert_layer_id if use_vit_merger else -1
        if use_vit_merger and insert_layer_id >= 0:
            for layer_index, encoder_layer in enumerate(self.encoder.layers):
                hidden_states = encoder_layer(hidden_states, cu_seqlens)
                if layer_index == insert_layer_id:
                    hidden_states = self.vit_merger(hidden_states, target_sizes)
                    target_sizes = target_sizes // 2
                    cu_seqlens = self._cu_seqlens(target_sizes, hidden_states.device)
        else:
            for encoder_layer in self.encoder.layers:
                hidden_states = encoder_layer(hidden_states, cu_seqlens)
        return self.post_layernorm(hidden_states)


class MiniCPMV47DownsampleMLP(nn.Module):
    def __init__(self, hidden_size: int, llm_embed_dim: int):
        super().__init__()
        merged_hidden_size = hidden_size * 4
        self.pre_norm = nn.LayerNorm(merged_hidden_size, eps=1e-6)
        self.linear_1 = nn.Linear(merged_hidden_size, merged_hidden_size, bias=True)
        self.linear_2 = nn.Linear(merged_hidden_size, llm_embed_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.pre_norm(hidden_states).view(-1, self.linear_1.in_features)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        return self.linear_2(hidden_states)


class MiniCPMV47Merger(nn.Module):
    """Final per-crop 2x2 merge + projection into the LLM hidden size."""

    def __init__(self, vision_hidden_size: int, llm_embed_dim: int, merge_kernel_size=(2, 2), merger_times: int = 1):
        super().__init__()
        self.merge_kernel_size = tuple(merge_kernel_size)
        self.merger_times = merger_times
        mlps = [MiniCPMV47DownsampleMLP(vision_hidden_size, vision_hidden_size) for _ in range(merger_times - 1)]
        mlps.append(MiniCPMV47DownsampleMLP(vision_hidden_size, llm_embed_dim))
        self.mlp = nn.ModuleList(mlps)

    def forward(self, hidden_states: torch.Tensor, target_sizes: torch.Tensor) -> List[torch.Tensor]:
        merge_h, merge_w = self.merge_kernel_size
        start = 0
        processed_features = []
        for batch_idx in range(len(target_sizes)):
            height, width = int(target_sizes[batch_idx][0]), int(target_sizes[batch_idx][1])
            num_patches = height * width
            embed_dim = hidden_states.shape[-1]
            if height % merge_h != 0 or width % merge_w != 0:
                raise ValueError(
                    f"patch grid ({height}, {width}) must be divisible by merge kernel "
                    f"{self.merge_kernel_size}"
                )
            merged_h, merged_w = height // merge_h, width // merge_w
            hidden_state = (
                hidden_states[0, start : start + num_patches, :]
                .view(merged_h, merge_h, merged_w, merge_w, embed_dim)
                .permute(0, 2, 1, 3, 4)
                .reshape(merged_h * merged_w, merge_h * merge_w * embed_dim)
            )
            hidden_state = self.mlp[0](hidden_state)
            for i in range(1, self.merger_times):
                height = height // merge_h
                width = width // merge_w
                inner_dim = hidden_state.shape[-1]
                merged_h, merged_w = height // merge_h, width // merge_w
                hidden_state = (
                    hidden_state.view(merged_h, merge_h, merged_w, merge_w, inner_dim)
                    .permute(0, 2, 1, 3, 4)
                    .reshape(merged_h * merged_w, merge_h * merge_w * inner_dim)
                )
                hidden_state = self.mlp[i](hidden_state)
            start += num_patches
            processed_features.append(hidden_state)
        return processed_features


def minicpmv47_image_features(
    visual: MiniCPMV47VisionModel,
    merger: MiniCPMV47Merger,
    pixel_values: torch.Tensor,
    target_sizes: torch.Tensor,
    downsample_mode: str,
) -> List[torch.Tensor]:
    """Equivalent of MiniCPMV4_7Model.get_image_features: tower + final merger.

    Returns one feature tensor per crop (row-major crop order).
    """
    use_vit_merger = downsample_mode != "4x"
    last_hidden_state = visual(pixel_values, target_sizes, use_vit_merger=use_vit_merger)
    if use_vit_merger:
        target_sizes = target_sizes // 2
    return merger(last_hidden_state, target_sizes)


def _grid_positions(llm_h: int, llm_w: int, h_coords: torch.Tensor, w_coords: torch.Tensor) -> torch.Tensor:
    h_grid = h_coords.view(-1, 1).expand(-1, llm_w)
    w_grid = w_coords.view(1, -1).expand(llm_h, -1)
    t_grid = torch.zeros_like(h_grid)
    return torch.stack((t_grid, h_grid, w_grid), dim=-1).reshape(-1, 3)


def minicpmv47_canvas_position_ids(
    crop_grids: List[List[int]],
    grid: Optional[List[int]],
    downsample_mode: str,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Relative canvas M-RoPE positions (T=0, H, W) for one image's crops.

    Mirrors ``MiniCPMV4_7Model.get_vision_position_ids`` with
    ``start_position=0``: the thumbnail is stretched over the slice canvas
    (linspace) when slices exist; each slice sits at its (row, col) offset.
    ``crop_grids`` are the original patch grids (pre window-merge);
    ``grid`` is the processor's slice grid where grid[0] = slice rows and
    grid[1] = slice columns (verified against the reference processor).
    The caller (C++ MROPE position-id generator) adds the canvas origin to
    all three channels.
    """
    merge_factor = 2 if downsample_mode == "4x" else 4
    if device is None:
        device = torch.device("cpu")
    thumb_h, thumb_w = int(crop_grids[0][0]), int(crop_grids[0][1])
    thumb_llm_h, thumb_llm_w = thumb_h // merge_factor, thumb_w // merge_factor

    positions: List[torch.Tensor] = []
    if grid is None or len(crop_grids) == 1:
        h_coords = torch.arange(thumb_llm_h, device=device)
        w_coords = torch.arange(thumb_llm_w, device=device)
        positions.append(_grid_positions(thumb_llm_h, thumb_llm_w, h_coords, w_coords))
        return torch.cat(positions, dim=0).to(torch.int32)

    num_rows, num_cols = int(grid[0]), int(grid[1])
    if num_rows * num_cols != len(crop_grids) - 1:
        raise ValueError(f"slice grid {grid} does not match crop count {len(crop_grids) - 1}")
    slice_h, slice_w = int(crop_grids[1][0]), int(crop_grids[1][1])
    slice_llm_h, slice_llm_w = slice_h // merge_factor, slice_w // merge_factor
    canvas_h, canvas_w = num_rows * slice_llm_h, num_cols * slice_llm_w
    h_coords = torch.linspace(0, canvas_h - 1, thumb_llm_h, device=device).round().long()
    w_coords = torch.linspace(0, canvas_w - 1, thumb_llm_w, device=device).round().long()
    positions.append(_grid_positions(thumb_llm_h, thumb_llm_w, h_coords, w_coords))
    for k in range(1, len(crop_grids)):
        slice_k_h, slice_k_w = int(crop_grids[k][0]), int(crop_grids[k][1])
        k_llm_h, k_llm_w = slice_k_h // merge_factor, slice_k_w // merge_factor
        h_off = ((k - 1) // num_cols) * slice_llm_h
        w_off = ((k - 1) % num_cols) * slice_llm_w
        k_h = torch.arange(k_llm_h, device=device) + h_off
        k_w = torch.arange(k_llm_w, device=device) + w_off
        positions.append(_grid_positions(k_llm_h, k_llm_w, k_h, k_w))
    return torch.cat(positions, dim=0).to(torch.int32)
