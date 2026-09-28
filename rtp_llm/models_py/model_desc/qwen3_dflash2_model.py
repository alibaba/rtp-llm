# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 DFlash2: DFlash backbone, dynamic convolution and candidate selector.

Convolution wrapper contract follows vLLM PR #52816, commit
3406ec1dae9916f920b90f0dbf90dcf54923d042.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from rtp_llm.models_py.model_desc.qwen3_dflash_model import (
    Qwen3DFlashModel,
    _DFlashDecoderLayer,
)
from rtp_llm.models_py.triton_kernels.common.dflash2_conv import grouped_conv
from rtp_llm.utils.model_weight import W


class DFlash2GroupedConv(nn.Module):
    """Wrap a sublayer using coefficients projected once from its input."""

    def __init__(
        self,
        base_kernel: torch.Tensor,
        kernel_projection: torch.Tensor,
        hidden_size: int,
        taps: int,
        group_size: int,
        query_width: int,
    ) -> None:
        super().__init__()
        if group_size <= 0 or hidden_size % group_size or taps <= 0:
            raise ValueError("invalid DFlash2 dynamic convolution dimensions")
        self.taps = taps
        self.group_size = group_size
        self.groups = hidden_size // group_size
        self.query_width = query_width
        if query_width <= 0 or base_kernel.shape != (2, taps, hidden_size):
            raise ValueError("DFlash2 base_kernel must have shape [2,taps,hidden]")
        if kernel_projection.shape != (2 * taps * self.groups, hidden_size):
            raise ValueError("DFlash2 kernel_projection has incompatible dimensions")
        # Register views without copies: ModelLoader already owns these tensors.
        self.register_buffer("base_kernel", base_kernel)
        self.register_buffer("kernel_projection", kernel_projection)

    def prepare(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = F.linear(hidden, self.kernel_projection).reshape(
            hidden.shape[0], 2, self.taps, self.groups
        )
        return (
            grouped_conv(
                hidden,
                coefficients[:, 0],
                self.base_kernel[0],
                self.query_width,
                self.group_size,
            ),
            coefficients[:, 1],
        )

    def finish(self, hidden: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        return grouped_conv(
            hidden,
            coefficients,
            self.base_kernel[1],
            self.query_width,
            self.group_size,
        )


class _DFlash2DecoderLayer(_DFlashDecoderLayer):
    def __init__(
        self,
        config,
        parallelism_config,
        weights,
        layer_idx,
        layer_type,
        quant_config,
        hw_kernel_config,
    ) -> None:
        super().__init__(
            config,
            parallelism_config,
            weights,
            layer_idx,
            layer_type,
            quant_config,
            hw_kernel_config,
        )
        kwargs = dict(
            hidden_size=config.hidden_size,
            taps=config.dflash2_conv_kernel_size,
            group_size=config.dflash2_conv_group_size,
            query_width=int(config.gen_num_per_cycle) + 1,
        )
        self.attention_conv = DFlash2GroupedConv(
            weights[W.dflash2_attention_conv_base],
            weights[W.dflash2_attention_conv_kernel],
            **kwargs,
        )
        self.mlp_conv = DFlash2GroupedConv(
            weights[W.dflash2_mlp_conv_base],
            weights[W.dflash2_mlp_conv_kernel],
            **kwargs,
        )

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        if kwargs["query_width"] != self.attention_conv.query_width:
            raise ValueError(
                "DFlash2 convolution and attention query widths must match"
            )
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(hidden_states, **kwargs)
        hidden_states = self.attention_conv.finish(hidden_states, coefficients)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
        return residual + hidden_states


class Qwen3DFlash2Model(Qwen3DFlashModel):
    decoder_layer_cls = _DFlash2DecoderLayer

    def __init__(self, config, parallelism_config, weights, *args, **kwargs) -> None:
        super().__init__(config, parallelism_config, weights, *args, **kwargs)
        from rtp_llm.models_py.speculative.dflash2_selector import (
            DFlash2CandidateSelector,
        )

        self.input_embedding_scale = float(config.dflash2_input_embedding_scale)
        self.candidate_selector = DFlash2CandidateSelector(
            weights.get_global_weight(W.dflash2_selector_projection),
            weights.get_global_weight(W.dflash2_selector_predecessor),
            weights.get_global_weight(W.dflash2_selector_successor),
            int(config.dflash2_selector_top_k),
        )

    def embed_query_tokens(self, query_ids: torch.Tensor) -> torch.Tensor:
        hidden = super().embed_query_tokens(query_ids)
        return (
            hidden
            if self.input_embedding_scale == 1.0
            else hidden * self.input_embedding_scale
        )

    def configure_dflash2_graph(self, enabled: bool) -> None:
        """Called by the proposal wrapper after graph availability is resolved."""
        self.candidate_selector.configure_graph(enabled)

    def get_dflash2_graph_stats(self):
        return self.candidate_selector.graph_stats

    def sample_dflash2(
        self, hidden, logits, anchors, temperatures, greedy_mask, uniforms
    ):
        """Rank-local post-head selector; never performs a TP collective."""
        return self.candidate_selector(
            hidden, logits, anchors, temperatures, greedy_mask, uniforms
        )


__all__ = ["DFlash2GroupedConv", "Qwen3DFlash2Model"]
