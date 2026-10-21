"""Vendored MOSS-VL vision tower (SigLIP-style) + merger + separator injection.

Reimplements the reference implementation (OpenMOSS-Team/MOSS-VL-Instruct-0708
modeling_moss_vl.py) without transformers remote-code: Conv3d patch embed,
learned absolute pos_embed with bilinear interpolation, 27 pre-norm blocks with
fused QKV + per-frame non-causal varlen attention (eager/SDPA/FA2 siblings
share one signature), DeepStack feature taps at blocks 8/16/24, and the
4-level merger (18432 -> 4096). The learnable separator token lives on this
module so the mixin loader can bind model.separator_token through the same
identity+reshape contract.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

MOSS_VL_DEFAULT_VISION_CONFIG = {
    "deepstack_visual_indexes": [8, 16, 24],
    "depth": 27,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 1152,
    "in_channels": 3,
    "initializer_range": 0.02,
    "intermediate_size": 4304,
    "num_heads": 16,
    "num_position_embeddings": 2304,
    "out_hidden_size": 4096,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 1,
}


class MossVLVisionConfig:
    def __init__(self, config: Dict[str, Any]):
        cfg = dict(MOSS_VL_DEFAULT_VISION_CONFIG)
        cfg.update(config or {})
        self.deepstack_visual_indexes = list(cfg["deepstack_visual_indexes"])
        self.depth = cfg["depth"]
        self.hidden_size = cfg["hidden_size"]
        self.in_channels = cfg["in_channels"]
        self.intermediate_size = cfg["intermediate_size"]
        self.num_heads = cfg["num_heads"]
        self.num_position_embeddings = cfg["num_position_embeddings"]
        self.out_hidden_size = cfg["out_hidden_size"]
        self.patch_size = cfg["patch_size"]
        self.spatial_merge_size = cfg["spatial_merge_size"]
        self.temporal_patch_size = cfg["temporal_patch_size"]
        self.hidden_act = cfg["hidden_act"]
        self.head_dim = self.hidden_size // self.num_heads


class MossVLVisionPatchEmbed(nn.Module):
    def __init__(self, config: MossVLVisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel_size = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(
            self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class MossVLVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Match the reference numerics: rotate in float32, cast back.
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class MossVLVisionAttention(nn.Module):
    """Per-frame non-causal attention with a single forward signature.

    `attn_implementation` selects eager / sdpa / flash_attention_2; all three
    accept the same kwargs so routing to any backend cannot raise a signature
    mismatch (sibling-signature discipline).
    """

    def __init__(self, config: MossVLVisionConfig, attn_implementation: str = "sdpa"):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)
        self.attn_implementation = attn_implementation

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        # [seq, heads, dim] -> [1, heads, seq, dim] batch for the backends.
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        if self.attn_implementation == "flash_attention_2":
            attn_output = self._flash_attn(q, k, v, cu_seqlens, lengths)
        elif self.attn_implementation == "sdpa":
            attn_output = self._sdpa(q, k, v, lengths)
        else:
            attn_output = self._eager(q, k, v, lengths)
        # backends return [1, heads, seq, dim]; merge heads before the output
        # projection (the reference's transformers sdpa wrapper already
        # transposes internally, so it reshapes directly).
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)

    def _sdpa(self, q, k, v, lengths):
        outputs = []
        splits = [torch.split(t, lengths, dim=2) for t in (q, k, v)]
        for qi, ki, vi in zip(*splits):
            outputs.append(
                F.scaled_dot_product_attention(qi, ki, vi, is_causal=False, scale=self.scaling)
            )
        return torch.cat(outputs, dim=2)

    def _eager(self, q, k, v, lengths):
        outputs = []
        splits = [torch.split(t, lengths, dim=2) for t in (q, k, v)]
        for qi, ki, vi in zip(*splits):
            attn_weights = torch.matmul(qi, ki.transpose(2, 3)) * self.scaling
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(qi.dtype)
            outputs.append(torch.matmul(attn_weights, vi))
        return torch.cat(outputs, dim=2)

    def _flash_attn(self, q, k, v, cu_seqlens, lengths):
        from flash_attn import flash_attn_varlen_func

        max_seqlen = max(lengths)
        # [1, heads, seq, dim] -> packed varlen [total, heads, dim]
        q_packed = q.squeeze(0).transpose(0, 1)
        k_packed = k.squeeze(0).transpose(0, 1)
        v_packed = v.squeeze(0).transpose(0, 1)
        out = flash_attn_varlen_func(
            q_packed,
            k_packed,
            v_packed,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=self.scaling,
            causal=False,
        )
        # [total, heads, dim] -> [1, heads, seq, dim]
        return out.transpose(0, 1).unsqueeze(0)


class MossVLVisionMLP(nn.Module):
    def __init__(self, config: MossVLVisionConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class MossVLVisionBlock(nn.Module):
    def __init__(self, config: MossVLVisionConfig, attn_implementation: str = "sdpa"):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = MossVLVisionAttention(config, attn_implementation)
        self.mlp = MossVLVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class MossVLVisionPatchMerger(nn.Module):
    """Fuses the last-layer output with the DeepStack taps.

    Input per merge block: 4 tokens x 4 levels (last, ds8, ds16, ds24), each
    level LayerNormed independently over 1152 dims, concatenated level-minor
    per token then grouped 2x2 -> 18432 -> fc1 (GELU erf) -> fc2 -> 4096.
    """

    def __init__(self, config: MossVLVisionConfig, num_deepstack_features: int = 0):
        super().__init__()
        base_hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.input_hidden_size = base_hidden_size * (1 + num_deepstack_features)
        num_features = 1 + num_deepstack_features
        self.norms = nn.ModuleList(
            [nn.LayerNorm(config.hidden_size, eps=1e-6) for _ in range(num_features)]
        )
        self.hidden_size = config.hidden_size
        self.linear_fc1 = nn.Linear(self.input_hidden_size, self.input_hidden_size, bias=True)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.input_hidden_size, config.out_hidden_size, bias=True)

    def forward(
        self, last_hidden_state: torch.Tensor, deepstack_features: List[torch.Tensor] = ()
    ) -> torch.Tensor:
        all_inputs = [last_hidden_state] + list(deepstack_features)
        outs = [self.norms[i](feat) for i, feat in enumerate(all_inputs)]
        x = torch.cat(outs, dim=-1)
        x = x.view(-1, self.input_hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class MossVLVisionModel(nn.Module):
    def __init__(self, config: MossVLVisionConfig, attn_implementation: str = "sdpa"):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.patch_embed = MossVLVisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = MossVLVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            [MossVLVisionBlock(config, attn_implementation) for _ in range(config.depth)]
        )
        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.merger = MossVLVisionPatchMerger(
            config, num_deepstack_features=len(self.deepstack_visual_indexes)
        )
        # Learnable separator inserted after each media frame's vision tokens;
        # bound from checkpoint key model.separator_token by the mixin loader.
        self.separator_token = nn.Parameter(torch.zeros(config.out_hidden_size))

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self):
        return self.patch_embed.proj.weight.device

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """2D (row, col) patch-level rotary freqs, [total_tokens, head_dim//2]."""
        merge_size = self.spatial_merge_size
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device
        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)
        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra = torch.arange(merge_size, device=device)
            row_idx = block_rows[:, None, None, None] * merge_size + intra[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra[None, None, None, :]
            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)
            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens
        embeddings = freq_table[pos_ids]
        return embeddings.flatten(1)

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
            h_floor = h_idxs.int()
            w_floor = w_idxs.int()
            h_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            dh = h_idxs - h_floor
            dw = w_idxs - w_floor
            base_h = h_floor * self.num_grid_per_side
            base_h_ceil = h_ceil * self.num_grid_per_side
            indices = [
                (base_h[None].T + w_floor[None]).flatten(),
                (base_h[None].T + w_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_floor[None]).flatten(),
                (base_h_ceil[None].T + w_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())
        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])
        merge_size = self.config.spatial_merge_size
        reordered = []
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            reordered.append(pos_embed)
        return torch.cat(reordered)

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """pixel patch tokens [total, 3*tps*16*16] + grid [M,3] -> merged [total/4, out_hidden]."""
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.size()
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_features = []
        for layer_idx, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_idx in self.deepstack_visual_indexes:
                deepstack_features.append(hidden_states)
        hidden_states = self.merger(hidden_states, deepstack_features)
        return hidden_states

    def vision_sequence_per_media(
        self, merged: torch.Tensor, grid_thw: torch.Tensor, pad_multiple: int = 1
    ) -> List[torch.Tensor]:
        """Split merged tokens per media and append one separator per frame.

        Reference MossVLModel.convert_packed_to_batch, per-media (no batch
        padding here; batching belongs to the engine): each image gets 1
        separator, each video frame gets 1 separator after its tokens.
        """
        merge_area = self.spatial_merge_size**2
        separator = self.separator_token.to(merged.dtype)
        chunks = torch.split(merged, (grid_thw.prod(-1) // merge_area).tolist())
        sequences = []
        for chunk, (t, _h, _w) in zip(chunks, grid_thw):
            num_frames = int(t)
            tokens_per_frame = chunk.shape[0] // num_frames
            frames = chunk.view(num_frames, tokens_per_frame, -1)
            separators = separator.view(1, 1, -1).expand(num_frames, 1, -1)
            sequences.append(torch.cat([frames, separators], dim=1).flatten(0, 1))
        if pad_multiple > 1:
            sequences = [
                self._pad_to_multiple(seq, pad_multiple, separator) for seq in sequences
            ]
        return sequences

    @staticmethod
    def _pad_to_multiple(sequence: torch.Tensor, multiple: int, separator: torch.Tensor):
        remainder = sequence.shape[0] % multiple
        if remainder == 0:
            return sequence
        pad = multiple - remainder
        pad_rows = separator.view(1, 1, -1).expand(pad, 1, -1).flatten(0, 1).to(sequence.dtype)
        return torch.cat([sequence, pad_rows], dim=0)

