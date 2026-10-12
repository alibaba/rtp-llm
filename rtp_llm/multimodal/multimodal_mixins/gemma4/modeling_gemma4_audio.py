# Copyright 2026 the HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gemma4 audio encoder, ported from Transformers d4f69a8.

Preserve checkpoint-native clipping, relative attention and convolution math.
Weights are filled by the standard multimodal loader; no HF runtime loader.
"""
from __future__ import annotations

import math
from functools import cached_property
from types import SimpleNamespace
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F

ACT2FN = {"silu": F.silu}


def create_bidirectional_mask(config, inputs_embeds, attention_mask, and_mask_function):
    positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
    allowed = and_mask_function(None, None, positions[:, None], positions[None, :])
    allowed = allowed[None, None].expand(inputs_embeds.shape[0], -1, -1, -1)
    if attention_mask is not None:
        allowed = allowed & attention_mask[:, None, None, :].bool()
    return allowed


class Gemma4ClippableLinear(nn.Module):

    def __init__(
        self,
        config: SimpleNamespace,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.use_clipped_linears = config.use_clipped_linears
        self.linear = nn.Linear(in_features, out_features, bias=False)
        if self.use_clipped_linears:
            self.input_min = nn.Buffer(torch.tensor(-float("inf")))
            self.input_max = nn.Buffer(torch.tensor(float("inf")))
            self.output_min = nn.Buffer(torch.tensor(-float("inf")))
            self.output_max = nn.Buffer(torch.tensor(float("inf")))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.input_min, self.input_max)
        hidden_states = self.linear(hidden_states)
        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.output_min, self.output_max)
        return hidden_states


class Gemma4RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-06, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim), requires_grad=True)

    def _norm(self, hidden_states: torch.Tensor):
        mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        return hidden_states * torch.pow(mean_squared, -0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed_output = self._norm(hidden_states.float())
        if self.with_scale:
            normed_output = normed_output * self.weight.float()
        return normed_output.type_as(hidden_states)


class Gemma4AudioRelPositionalEncoding(nn.Module):
    """Sinusoidal relative positional encoding for the audio encoder.
    Produces position embeddings of shape [1, context_size // 2 + 1, hidden_size] with
    concatenated [sin..., cos...] layout matching the original Gemma4 convention.
    """

    inv_timescales: torch.Tensor

    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.context_size = (
            config.attention_chunk_size
            + config.attention_context_left
            - 1
            + config.attention_context_right
        )
        min_timescale = 1.0
        max_timescale = 10000.0
        num_timescales = self.hidden_size // 2
        log_timescale_increment = math.log(max_timescale / min_timescale) / max(
            num_timescales - 1, 1
        )
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, device="cpu") * -log_timescale_increment
        )
        self.inv_timescales = nn.Buffer(
            inv_timescales.unsqueeze(0).unsqueeze(0), persistent=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        position_ids = torch.arange(
            self.context_size // 2, -1, -1, device=hidden_states.device
        )
        position_ids = position_ids[..., None]
        scaled_time = position_ids * self.inv_timescales.to(device=hidden_states.device)
        pos_embed = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        return pos_embed.to(dtype=hidden_states.dtype)


class Gemma4AudioAttention(nn.Module):
    """Chunked local attention with relative position bias"""

    def __init__(self, config: SimpleNamespace, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_logits_soft_cap = config.attention_logit_cap
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.q_scale = self.head_dim ** (-0.5) / math.log(2)
        self.k_scale = math.log(1 + math.e) / math.log(2)
        self.chunk_size = config.attention_chunk_size
        self.max_past_horizon = config.attention_context_left - 1
        self.max_future_horizon = config.attention_context_right
        self.context_size = (
            self.chunk_size + self.max_past_horizon + self.max_future_horizon
        )
        self.q_proj = Gemma4ClippableLinear(
            config, config.hidden_size, self.num_heads * self.head_dim
        )
        self.k_proj = Gemma4ClippableLinear(
            config, config.hidden_size, self.num_heads * self.head_dim
        )
        self.v_proj = Gemma4ClippableLinear(
            config, config.hidden_size, self.num_heads * self.head_dim
        )
        self.post = Gemma4ClippableLinear(
            config, config.hidden_size, config.hidden_size
        )
        self.relative_k_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.per_dim_scale = nn.Parameter(torch.zeros(self.head_dim))
        self.softcap = nn.Buffer(
            torch.tensor(self.attention_logits_soft_cap), persistent=False
        )

    def _convert_to_block(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Splits a `(batch_size, seq_len, num_heads, head_dim)` tensor into non-overlapping blocks of `chunk_size` along the sequence dim."""
        (batch_size, seq_len, num_heads, head_dim) = hidden_states.shape
        num_blocks = (seq_len + self.chunk_size - 1) // self.chunk_size
        pad = num_blocks * self.chunk_size - seq_len
        hidden_states = F.pad(hidden_states, (0, 0, 0, 0, 0, pad))
        return hidden_states.reshape(
            batch_size, num_blocks, self.chunk_size, num_heads, head_dim
        ).contiguous()

    def _extract_block_context(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Extracts overlapping context windows of `context_size` for every block, strided by `chunk_size`."""
        (batch_size, seq_len, num_heads, head_dim) = hidden_states.shape
        hidden_states = F.pad(
            hidden_states,
            (
                0,
                0,
                0,
                0,
                self.max_past_horizon,
                self.max_future_horizon + self.chunk_size - 1,
            ),
        )
        hidden_states = hidden_states.unfold(1, self.context_size, self.chunk_size)
        hidden_states = torch.movedim(hidden_states, -1, 2)
        return hidden_states.contiguous()

    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Relative position shift for blocked attention. See appendix B of https://huggingface.co/papers/1901.02860."""
        (batch_size, num_heads, num_blocks, block_size, position_length) = x.shape
        context_size = self.context_size
        x = F.pad(x, (0, context_size + 1 - position_length))
        x = x.view(batch_size, num_heads, num_blocks, block_size * (context_size + 1))
        x = x[..., : block_size * context_size]
        return x.view(batch_size, num_heads, num_blocks, block_size, context_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.BoolTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (batch_size, seq_length, _) = hidden_states.shape
        hidden_shape = (batch_size, seq_length, self.num_heads, self.head_dim)
        query_states = self.q_proj(hidden_states).float().view(hidden_shape)
        key_states = self.k_proj(hidden_states).float().view(hidden_shape)
        value_states = self.v_proj(hidden_states).float().view(hidden_shape)
        query_states = query_states * self.q_scale * F.softplus(self.per_dim_scale)
        key_states = key_states * self.k_scale
        query_states = self._convert_to_block(query_states)
        key_states = self._extract_block_context(key_states)
        value_states = self._extract_block_context(value_states)
        num_blocks = query_states.shape[1]
        relative_key_states = self.relative_k_proj(position_embeddings)
        relative_key_states = relative_key_states.view(
            -1, self.num_heads, self.head_dim
        )
        relative_key_states = relative_key_states.to(dtype=query_states.dtype)
        queries = query_states.permute(0, 3, 1, 2, 4)
        matrix_ac = queries @ key_states.permute(0, 3, 1, 4, 2)
        queries_flat = queries.reshape(batch_size, self.num_heads, -1, self.head_dim)
        matrix_bd = queries_flat @ relative_key_states.permute(1, 2, 0)
        matrix_bd = matrix_bd.reshape(
            batch_size, self.num_heads, num_blocks, self.chunk_size, -1
        )
        matrix_bd = self._rel_shift(matrix_bd)
        attn_weights = matrix_ac + matrix_bd
        attn_weights = attn_weights / self.softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * self.softcap
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(
                attention_mask.logical_not(), self.config.attention_invalid_logits_value
            )
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            value_states.dtype
        )
        attn_output = attn_weights @ value_states.permute(0, 3, 1, 2, 4)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(
            batch_size, num_blocks * self.chunk_size, -1
        )
        attn_output = attn_output[:, :seq_length].contiguous()
        attn_output = self.post(attn_output.to(hidden_states.dtype))
        return (attn_output, attn_weights)


class Gemma4AudioSubSampleConvProjectionLayer(nn.Module):

    def __init__(self, in_channels, out_channels, norm_eps):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=1,
            bias=False,
        )
        self.norm = nn.LayerNorm(
            out_channels, eps=norm_eps, elementwise_affine=True, bias=False
        )
        self.act = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None):
        if mask is not None:
            mask = mask.to(device=hidden_states.device)
            hidden_states = hidden_states * mask[:, None, :, None]
        hidden_states = self.conv(hidden_states.to(self.conv.weight.dtype))
        hidden_states = self.act(
            self.norm(hidden_states.permute(0, 2, 3, 1))
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        if mask is not None:
            mask = mask[:, ::2]
        return (hidden_states, mask)


class Gemma4AudioSubSampleConvProjection(nn.Module):

    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.layer0 = Gemma4AudioSubSampleConvProjectionLayer(
            in_channels=1,
            out_channels=config.subsampling_conv_channels[0],
            norm_eps=config.rms_norm_eps,
        )
        self.layer1 = Gemma4AudioSubSampleConvProjectionLayer(
            in_channels=config.subsampling_conv_channels[0],
            out_channels=config.subsampling_conv_channels[1],
            norm_eps=config.rms_norm_eps,
        )
        proj_input_dim = (
            config.subsampling_conv_channels[0]
            // 4
            * config.subsampling_conv_channels[1]
        )
        self.input_proj_linear = nn.Linear(
            proj_input_dim, config.hidden_size, bias=False
        )

    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = input_features.unsqueeze(1)
        (hidden_states, mask) = self.layer0(hidden_states, input_features_mask)
        (hidden_states, mask) = self.layer1(hidden_states, mask)
        (batch_size, _, seq_len, _) = hidden_states.shape
        hidden_states = (
            hidden_states.permute(0, 2, 3, 1)
            .contiguous()
            .reshape(batch_size, seq_len, -1)
        )
        return (self.input_proj_linear(hidden_states), mask)


class Gemma4AudioFeedForward(nn.Module):

    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
        self.ffw_layer_1 = Gemma4ClippableLinear(
            config, config.hidden_size, config.hidden_size * 4
        )
        self.ffw_layer_2 = Gemma4ClippableLinear(
            config, config.hidden_size * 4, config.hidden_size
        )
        self.pre_layer_norm = Gemma4RMSNorm(config.hidden_size)
        self.post_layer_norm = Gemma4RMSNorm(config.hidden_size)
        self.act_fn = ACT2FN[config.hidden_act]
        self.gradient_clipping = config.gradient_clipping
        self.post_layer_scale = config.residual_weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gradient_clipping = min(
            self.gradient_clipping, torch.finfo(hidden_states.dtype).max
        )
        residual = hidden_states
        hidden_states = torch.clamp(
            hidden_states, -gradient_clipping, gradient_clipping
        )
        hidden_states = self.pre_layer_norm(hidden_states)
        hidden_states = self.ffw_layer_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.ffw_layer_2(hidden_states)
        hidden_states = torch.clamp(
            hidden_states, -gradient_clipping, gradient_clipping
        )
        hidden_states = self.post_layer_norm(hidden_states)
        hidden_states *= self.post_layer_scale
        hidden_states += residual
        return hidden_states


class Gemma4AudioCausalConv1d(nn.Conv1d):

    @cached_property
    def left_pad(self):
        effective_kernel_size = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        return effective_kernel_size - self.stride[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.left_pad, 0))
        return super().forward(x)


class Gemma4AudioLightConv1d(nn.Module):

    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
        self.linear_start = Gemma4ClippableLinear(
            config, config.hidden_size, config.hidden_size * 2
        )
        self.linear_end = Gemma4ClippableLinear(
            config, config.hidden_size, config.hidden_size
        )
        self.depthwise_conv1d = Gemma4AudioCausalConv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=config.conv_kernel_size,
            groups=config.hidden_size,
            bias=False,
        )
        self.pre_layer_norm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, with_scale=True
        )
        self.conv_norm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, with_scale=True
        )
        self.act_fn = ACT2FN[config.hidden_act]
        self.gradient_clipping = config.gradient_clipping

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.pre_layer_norm(hidden_states)
        hidden_states = self.linear_start(hidden_states)
        hidden_states = nn.functional.glu(hidden_states, dim=-1)
        hidden_states = self.depthwise_conv1d(hidden_states.transpose(1, 2)).transpose(
            1, 2
        )
        gradient_clipping = min(
            self.gradient_clipping, torch.finfo(hidden_states.dtype).max
        )
        hidden_states = torch.clamp(
            hidden_states, -gradient_clipping, gradient_clipping
        )
        hidden_states = self.conv_norm(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_end(hidden_states)
        hidden_states += residual
        return hidden_states


class Gemma4AudioLayer(nn.Module):

    def __init__(self, config: SimpleNamespace, layer_idx: int):
        super().__init__()
        self.config = config
        self.feed_forward1 = Gemma4AudioFeedForward(config)
        self.feed_forward2 = Gemma4AudioFeedForward(config)
        self.self_attn = Gemma4AudioAttention(config, layer_idx)
        self.lconv1d = Gemma4AudioLightConv1d(config)
        self.norm_pre_attn = Gemma4RMSNorm(config.hidden_size)
        self.norm_post_attn = Gemma4RMSNorm(config.hidden_size)
        self.norm_out = Gemma4RMSNorm(config.hidden_size)
        self.gradient_clipping = config.gradient_clipping

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.BoolTensor | None,
        position_embeddings: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        gradient_clipping = min(
            self.gradient_clipping, torch.finfo(self.norm_pre_attn.weight.dtype).max
        )
        hidden_states = self.feed_forward1(hidden_states)
        residual = hidden_states
        hidden_states = torch.clamp(
            hidden_states, -gradient_clipping, gradient_clipping
        )
        hidden_states = self.norm_pre_attn(hidden_states)
        (hidden_states, _) = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = torch.clamp(
            hidden_states, -gradient_clipping, gradient_clipping
        )
        hidden_states = self.norm_post_attn(hidden_states)
        hidden_states += residual
        hidden_states = self.lconv1d(hidden_states)
        hidden_states = self.feed_forward2(hidden_states)
        hidden_states = torch.clamp(
            hidden_states, -gradient_clipping, gradient_clipping
        )
        hidden_states = self.norm_out(hidden_states)
        return hidden_states


def sliding_window_mask_function(sliding_window: tuple[int, int]) -> Callable:
    """
    This creates uni/bidirectional attention mask with sliding window.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        (left_window_size, right_window_size) = sliding_window
        dist = q_idx - kv_idx
        left_mask = (dist >= 0) & (dist < left_window_size)
        right_mask = (dist < 0) & (-dist < right_window_size)
        return left_mask | right_mask

    return inner_mask


class Gemma4AudioTower(nn.Module):
    """An audio encoder based on the [Universal Speech Model](https://huggingface.co/papers/2303.01037) architecture."""

    config: SimpleNamespace
    main_input_name = "input_features"
    base_model_prefix = "model.audio_tower"
    _can_record_outputs = {
        "hidden_states": Gemma4AudioLayer,
        "attentions": Gemma4AudioAttention,
    }

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        config = SimpleNamespace(**config)
        self.config = config
        self.subsample_conv_projection = Gemma4AudioSubSampleConvProjection(config)
        self.rel_pos_enc = Gemma4AudioRelPositionalEncoding(config)
        self.layers = nn.ModuleList(
            [
                Gemma4AudioLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.output_proj = nn.Linear(
            config.hidden_size, config.output_proj_dims, bias=True
        )

    def _convert_4d_mask_to_blocked_5d(self, mask_4d: torch.Tensor) -> torch.Tensor:
        """
        Convert a standard 4D attention mask `[batch_size, 1, seq_len, seq_len]` to the 5D blocked format
        `[batch_size, 1, num_blocks, chunk_size, context_size]` expected by the chunked local attention,
        """
        (batch_size, _, seq_len, _) = mask_4d.shape
        device = mask_4d.device
        chunk_size = self.config.attention_chunk_size
        max_past_horizon = self.config.attention_context_left - 1
        max_future_horizon = self.config.attention_context_right
        num_blocks = (seq_len + chunk_size - 1) // chunk_size
        padded_seq_len = num_blocks * chunk_size
        pad_amount = padded_seq_len - seq_len
        mask_4d = F.pad(mask_4d, (0, pad_amount, 0, pad_amount), value=False)
        mask_5d = mask_4d.reshape(batch_size, 1, num_blocks, chunk_size, padded_seq_len)
        mask_5d = F.pad(mask_5d, (max_past_horizon, max_future_horizon), value=False)
        block_starts = torch.arange(num_blocks, device=device) * chunk_size
        offsets = torch.arange(
            chunk_size + max_past_horizon + max_future_horizon, device=device
        )
        kv_indices = block_starts[:, None] + offsets[None, :]
        kv_indices = kv_indices[None, None, :, None, :].expand(
            batch_size, 1, -1, chunk_size, -1
        )
        return mask_5d.gather(-1, kv_indices)

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        (hidden_states, output_mask) = self.subsample_conv_projection(
            input_features, attention_mask
        )
        position_embeddings = self.rel_pos_enc(hidden_states)
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=output_mask,
            and_mask_function=sliding_window_mask_function(
                (
                    self.config.attention_context_left - 1,
                    self.config.attention_context_right,
                )
            ),
        )
        if attention_mask is not None:
            attention_mask = self._convert_4d_mask_to_blocked_5d(attention_mask)
        for encoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.output_proj(hidden_states)
        return (hidden_states, output_mask)
