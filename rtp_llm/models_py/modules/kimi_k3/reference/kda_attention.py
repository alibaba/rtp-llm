"""Kimi K3 Delta Attention module with an explicit Torch fallback backend.

The projections, three independent causal convolutions, gate computation and
output normalization follow ``KimiDeltaAttention`` from the K3 checkpoint.
The actual state update is delegated to the two correctness backends in
``kda_reference``: chunk form for prefill and recurrent form for decode.

This module is intentionally usable for 5-layer/small-shape bring-up.  The
dense affine scan in its prefill backend must be replaced by a Triton/CUDA KDA
kernel before running production sequence lengths.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.models_py.modules.kimi_k3.kda_state import (
    KDAExecutionMode,
    KimiKDAState,
)
from rtp_llm.models_py.modules.kimi_k3.reference.common import KimiRMSGated
from rtp_llm.models_py.modules.kimi_k3.reference.kda_reference import kimi_kda


class CausalDepthwiseConv1dReference(nn.Module):
    """Small Torch oracle matching RTP/FLA causal-convolution semantics."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        if channels <= 0 or kernel_size <= 0:
            raise ValueError("channels and kernel_size must be positive")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(
            self.weight,
            -1.0 / math.sqrt(self.kernel_size),
            1.0 / math.sqrt(self.kernel_size),
        )

    def forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError(
                f"causal conv expects [B,T,{self.channels}], got {tuple(x.shape)}"
            )
        batch = x.shape[0]
        history_size = self.kernel_size - 1
        if state is None:
            state = x.new_zeros(batch, self.channels, history_size)
        expected_state = (batch, self.channels, history_size)
        if state.shape != expected_state:
            raise ValueError(
                f"conv state must have shape {expected_state}, got {tuple(state.shape)}"
            )

        x_channels_first = x.transpose(1, 2)
        combined = torch.cat((state.to(dtype=x.dtype), x_channels_first), dim=-1)
        output = F.conv1d(
            combined.to(dtype=self.weight.dtype),
            self.weight,
            groups=self.channels,
        )
        output = F.silu(output).transpose(1, 2).to(dtype=x.dtype)
        if history_size == 0:
            final_state = combined[:, :, :0]
        else:
            final_state = combined[:, :, -history_size:]
        return output, final_state.to(dtype=x.dtype)


class KimiDeltaAttentionReference(nn.Module):
    """Complete K3 KDA layer for correctness bring-up and differential tests."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        *,
        conv_kernel_size: int = 4,
        rms_norm_eps: float = 1e-6,
        gate_lower_bound: Optional[float] = -5.0,
        use_full_rank_gate: bool = True,
        chunk_size: int = 64,
    ):
        super().__init__()
        if hidden_size <= 0 or num_heads <= 0 or head_dim <= 0:
            raise ValueError("hidden_size, num_heads and head_dim must be positive")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.projection_size = self.num_heads * self.head_dim
        self.gate_lower_bound = gate_lower_bound
        self.use_full_rank_gate = bool(use_full_rank_gate)
        self.chunk_size = int(chunk_size)

        self.q_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.q_conv1d = CausalDepthwiseConv1dReference(
            self.projection_size, conv_kernel_size
        )
        self.k_conv1d = CausalDepthwiseConv1dReference(
            self.projection_size, conv_kernel_size
        )
        self.v_conv1d = CausalDepthwiseConv1dReference(
            self.projection_size, conv_kernel_size
        )

        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads).uniform_(1.0, 16.0))
        )
        self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.f_b_proj = nn.Linear(head_dim, self.projection_size, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(self.projection_size))
        nn.init.uniform_(self.dt_bias, -0.1, 0.1)
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        if self.use_full_rank_gate:
            self.g_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        else:
            self.g_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
            self.g_b_proj = nn.Linear(head_dim, self.projection_size, bias=False)

        self.o_norm = KimiRMSGated(head_dim, eps=rms_norm_eps)
        self.o_proj = nn.Linear(self.projection_size, hidden_size, bias=False)

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(
            tensor.shape[0], tensor.shape[1], self.num_heads, self.head_dim
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: Optional[KimiKDAState] = None,
        *,
        mode: KDAExecutionMode,
    ) -> tuple[torch.Tensor, KimiKDAState]:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_states must have shape [B,T,{self.hidden_size}], got "
                f"{tuple(hidden_states.shape)}"
            )
        q_state = None if state is None else state.q_conv_state
        k_state = None if state is None else state.k_conv_state
        v_state = None if state is None else state.v_conv_state
        recurrent_state = None if state is None else state.recurrent_state

        q, q_final = self.q_conv1d(self.q_proj(hidden_states), q_state)
        k, k_final = self.k_conv1d(self.k_proj(hidden_states), k_state)
        v, v_final = self.v_conv1d(self.v_proj(hidden_states), v_state)
        q, k, v = map(self._reshape_heads, (q, k, v))

        raw_gate = self._reshape_heads(
            self.f_b_proj(self.f_a_proj(hidden_states))
        )
        raw_beta = self.b_proj(hidden_states).float()
        output, recurrent_final = kimi_kda(
            q,
            k,
            v,
            raw_gate,
            raw_beta,
            self.A_log,
            self.dt_bias,
            recurrent_state,
            mode=mode,
            lower_bound=self.gate_lower_bound,
            chunk_size=self.chunk_size,
        )

        if self.use_full_rank_gate:
            output_gate = self.g_proj(hidden_states)
        else:
            output_gate = self.g_b_proj(self.g_a_proj(hidden_states))
        output = self.o_norm(output, self._reshape_heads(output_gate))
        output = self.o_proj(output.reshape(*hidden_states.shape[:2], -1))
        final_state = KimiKDAState(
            q_conv_state=q_final,
            k_conv_state=k_final,
            v_conv_state=v_final,
            recurrent_state=recurrent_final,
        )
        return output, final_state
