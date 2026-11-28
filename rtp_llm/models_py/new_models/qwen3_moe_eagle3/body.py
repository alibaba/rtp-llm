"""The one-block, full-vocabulary NVIDIA Qwen3-235B Eagle3 checkpoint body.

The target owns the embedding and vocabulary projection. This module owns only
the tensors in the published draft shard. It does not select a backend or
register a speculative engine: the caller must provide its paged FMHA with
RoPE and its draft KV cache, just as in the other direct decoder owners.
"""

from typing import Any, NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from rtp_llm.models_py.module_base import RtpModule


class Eagle3BodyOutput(NamedTuple):
    # Final norm feeds the target vocabulary projection. The unnormalized
    # residual is the state passed to the *next* draft step.
    logits_input: torch.Tensor
    recurrent_hidden: torch.Tensor


class _Attention(nn.Module):
    def __init__(self, hidden, n_heads, n_kv_heads, head_dim, dtype):
        super().__init__()
        self.q_proj = nn.Linear(2 * hidden, n_heads * head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(
            2 * hidden, n_kv_heads * head_dim, bias=False, dtype=dtype
        )
        self.v_proj = nn.Linear(
            2 * hidden, n_kv_heads * head_dim, bias=False, dtype=dtype
        )
        self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False, dtype=dtype)

    def forward(self, x, fmha, kv_cache):
        packed = torch.cat((self.q_proj(x), self.k_proj(x), self.v_proj(x)), dim=-1)
        out = fmha.forward(packed, kv_cache, 0)
        return self.o_proj(out.reshape(x.shape[0], self.o_proj.in_features))


class _MLP(nn.Module):
    def __init__(self, hidden, intermediate, dtype):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False, dtype=dtype)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _MidLayer(nn.Module):
    def __init__(
        self,
        hidden,
        intermediate,
        n_heads,
        n_kv_heads,
        head_dim,
        eps,
        dtype,
        *,
        attention=_Attention,
        mlp=_MLP,
    ):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(hidden, eps=eps, dtype=dtype)
        self.hidden_norm = nn.RMSNorm(hidden, eps=eps, dtype=dtype)
        self.self_attn = attention(hidden, n_heads, n_kv_heads, head_dim, dtype)
        self.post_attention_layernorm = nn.RMSNorm(hidden, eps=eps, dtype=dtype)
        self.mlp = mlp(hidden, intermediate, dtype)

    def forward(self, embeds, hidden, fmha, kv_cache):
        # EAGLE3 concatenates normalized embeddings first; no Q/K head norms.
        fused = torch.cat((self.input_layernorm(embeds), self.hidden_norm(hidden)), -1)
        residual = hidden + self.self_attn(fused, fmha, kv_cache)
        return residual + self.mlp(self.post_attention_layernorm(residual))


class Eagle3DraftBody(RtpModule):
    """A full-width draft block, independent of target MoE expert placement."""

    def __init__(
        self,
        hidden,
        intermediate,
        n_heads,
        n_kv_heads,
        head_dim,
        eps,
        dtype,
        **layer_factories,
    ):
        super().__init__()
        if min(hidden, intermediate, n_heads, n_kv_heads, head_dim) <= 0:
            raise ValueError("draft dimensions must be positive")
        if n_heads % n_kv_heads or eps <= 0:
            raise ValueError("invalid Eagle3 GQA dimensions or RMSNorm epsilon")
        self.hidden_size = hidden
        self.fc = nn.Linear(3 * hidden, hidden, bias=False, dtype=dtype)
        self.midlayer = _MidLayer(
            hidden,
            intermediate,
            n_heads,
            n_kv_heads,
            head_dim,
            eps,
            dtype,
            **layer_factories,
        )
        self.norm = nn.RMSNorm(hidden, eps=eps, dtype=dtype)
        self.requires_grad_(False)

    def forward(self, embeddings, target_or_draft_hidden, fmha: Any, kv_cache=None):
        if embeddings.ndim != 2 or embeddings.shape[-1] != self.hidden_size:
            raise ValueError("expected a [tokens, hidden] target embedding")
        state = target_or_draft_hidden
        if state.ndim != 2 or state.shape[0] != embeddings.shape[0]:
            raise ValueError("draft hidden rows must align with input tokens")
        if state.shape[-1] == self.fc.in_features:
            state = self.fc(state)
        elif state.shape[-1] != self.hidden_size:
            raise ValueError(
                "expected three ordered target features or one draft state"
            )
        hidden = self.midlayer(embeddings, state, fmha, kv_cache)
        return Eagle3BodyOutput(self.norm(hidden), hidden)
