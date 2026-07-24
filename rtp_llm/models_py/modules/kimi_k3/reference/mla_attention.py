"""No-RoPE MLA correctness module for Kimi K3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from rtp_llm.models_py.modules.kimi_k3.reference.common import KimiRMSNorm


@dataclass(frozen=True)
class KimiMLAState:
    """Expanded Torch reference cache; production RTP stores compressed MLA KV."""

    key: torch.Tensor
    value: torch.Tensor


class KimiMLAAttentionReference(nn.Module):
    """K3 MLA with a physical no-RoPE suffix and sigmoid output gate.

    Existing RTP MLA kernels rotate the field historically named
    ``qk_rope_head_dim``.  K3 instead treats those 64 dimensions as an ordinary
    suffix, so this explicit implementation is the correctness path until a
    no-RoPE kernel flag is wired through the FMHA factory.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_suffix_head_dim: int,
        v_head_dim: int,
        *,
        rms_norm_eps: float = 1e-6,
        use_output_gate: bool = True,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.q_lora_rank = None if q_lora_rank is None else int(q_lora_rank)
        self.kv_lora_rank = int(kv_lora_rank)
        self.qk_nope_head_dim = int(qk_nope_head_dim)
        self.qk_suffix_head_dim = int(qk_suffix_head_dim)
        self.v_head_dim = int(v_head_dim)
        self.q_head_dim = self.qk_nope_head_dim + self.qk_suffix_head_dim
        self.softmax_scale = self.q_head_dim**-0.5
        self.use_output_gate = bool(use_output_gate)

        if self.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(hidden_size, self.q_lora_rank, bias=False)
            self.q_a_layernorm = KimiRMSNorm(
                self.q_lora_rank, eps=rms_norm_eps
            )
            self.q_b_proj = nn.Linear(
                self.q_lora_rank,
                self.num_heads * self.q_head_dim,
                bias=False,
            )
        else:
            self.q_proj = nn.Linear(
                hidden_size, self.num_heads * self.q_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size,
            self.kv_lora_rank + self.qk_suffix_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = KimiRMSNorm(
            self.kv_lora_rank, eps=rms_norm_eps
        )
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        if self.use_output_gate:
            self.g_proj = nn.Linear(
                hidden_size, self.num_heads * self.v_head_dim, bias=False
            )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: Optional[KimiMLAState] = None,
    ) -> tuple[torch.Tensor, KimiMLAState]:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_states must be [B,T,{self.hidden_size}], got "
                f"{tuple(hidden_states.shape)}"
            )
        batch, query_length, _ = hidden_states.shape
        if self.q_lora_rank is not None:
            query = self.q_b_proj(
                self.q_a_layernorm(self.q_a_proj(hidden_states))
            )
        else:
            query = self.q_proj(hidden_states)
        query = query.reshape(
            batch, query_length, self.num_heads, self.q_head_dim
        )

        compressed = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, key_suffix = torch.split(
            compressed, [self.kv_lora_rank, self.qk_suffix_head_dim], dim=-1
        )
        expanded = self.kv_b_proj(self.kv_a_layernorm(compressed_kv)).reshape(
            batch,
            query_length,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        key_nope, value = torch.split(
            expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        key_suffix = key_suffix.unsqueeze(2).expand(
            batch, query_length, self.num_heads, self.qk_suffix_head_dim
        )
        # Deliberately concatenate without applying RoPE.
        key = torch.cat((key_nope, key_suffix), dim=-1)

        prefix_length = 0
        if state is not None:
            if state.key.shape[0] != batch or state.value.shape[0] != batch:
                raise ValueError("MLA cache batch size does not match hidden_states")
            prefix_length = state.key.shape[1]
            key = torch.cat((state.key, key), dim=1)
            value = torch.cat((state.value, value), dim=1)

        scores = torch.einsum("bthd,bshd->bhts", query.float(), key.float())
        scores = scores * self.softmax_scale
        query_positions = prefix_length + torch.arange(
            query_length, device=hidden_states.device
        )
        key_positions = torch.arange(key.shape[1], device=hidden_states.device)
        causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        scores = scores.masked_fill(
            ~causal_mask.reshape(1, 1, query_length, key.shape[1]),
            torch.finfo(scores.dtype).min,
        )
        probabilities = torch.softmax(scores, dim=-1)
        output = torch.einsum(
            "bhts,bshv->bthv", probabilities, value.float()
        ).to(dtype=hidden_states.dtype)

        if self.use_output_gate:
            output_gate = self.g_proj(hidden_states).reshape(
                batch, query_length, self.num_heads, self.v_head_dim
            )
            output = output * torch.sigmoid(output_gate)
        output = self.o_proj(output.reshape(batch, query_length, -1))
        return output, KimiMLAState(key=key, value=value)
