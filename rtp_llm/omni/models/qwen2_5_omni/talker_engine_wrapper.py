"""Wrapper for talker engine that handles custom embedding + thinker hidden state injection.

The talker's forward pass differs from standard QWenV2:
  1. embed_tokens(codec_token) → [3584]  (embedding_size, not hidden_size)
  2. + thinker_hidden_state[step] → [3584]
  3. thinker_to_talker_proj → [896]  (hidden_size)
  4. transformer layers → [896]
  5. codec_head → [8448]  (vocab_size)

Since this custom embedding logic can't be injected into the C++ engine's
autoregressive loop, we use the engine's loaded weights to do the forward
pass in PyTorch while the engine handles weight management and lifecycle.
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.utils.model_weight import W

logger = logging.getLogger(__name__)


class TalkerEngineWrapper:
    """Wraps a loaded talker LanguageCppEngine for generation with custom embedding logic."""

    def __init__(self, talker_engine):
        self.engine = talker_engine
        model = talker_engine.model
        weight = model.weight

        self.device = model.device
        self.dtype = weight._dtype

        self.embed_tokens_weight = weight.get_global_weight(W.embedding)
        self.codec_head_weight = weight.get_global_weight(W.lm_head)
        self.proj_weight = weight.get_global_weight("thinker_to_talker_proj.weight")
        self.proj_bias = weight.get_global_weight("thinker_to_talker_proj.bias")

        self.layer_weights = weight.weights
        self.num_layers = model.model_config.num_layers
        self.hidden_size = model.model_config.hidden_size
        self.vocab_size = model.model_config.vocab_size

        norm_weight = weight.get_global_weight_or_none("final_ln_gamma")
        if norm_weight is None:
            norm_weight = weight.get_global_weight_or_none(W.final_ln_gamma)
        self.norm_weight = norm_weight
        self.layernorm_eps = model.model_config.layernorm_eps

        config = model.model_config
        self.head_num = config.attn_config.head_num
        self.kv_head_num = config.attn_config.kv_head_num
        self.head_dim = config.attn_config.size_per_head

        logger.info(
            f"TalkerEngineWrapper initialized: "
            f"embed={self.embed_tokens_weight.shape}, "
            f"proj={self.proj_weight.shape}, "
            f"layers={self.num_layers}, "
            f"hidden={self.hidden_size}, "
            f"head_dim={self.head_dim}x{self.head_num}"
        )

    def _embed_and_project(
        self,
        token_ids: torch.Tensor,
        thinker_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Embed codec tokens, add thinker hidden states, and project.

        Args:
            token_ids: [seq_len] codec token IDs
            thinker_hidden: [seq_len, 3584] thinker hidden states
        Returns:
            [seq_len, 896] projected embeddings
        """
        embeds = F.embedding(token_ids, self.embed_tokens_weight)
        combined = embeds + thinker_hidden
        projected = F.linear(combined, self.proj_weight, self.proj_bias)
        return projected

    def _rms_norm(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.layernorm_eps)
        return weight * hidden_states.to(weight.dtype)

    def _layer_forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]],
        position: int,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward through one transformer layer with simple causal attention.

        rtp-llm stores layer weights in [in, out] layout for matmul,
        and merges gate+up into a single w13 weight.
        """
        lw = self.layer_weights[layer_idx]

        pre_ln = lw.get(W.pre_ln_gamma)
        post_ln = lw.get(W.post_ln_gamma)
        qkv_w = lw.get(W.attn_qkv_w)  # [hidden, q+k+v]
        qkv_b = lw.get(W.attn_qkv_b)  # [q+k+v]
        o_w = lw.get(W.attn_o_w)  # [head_num*head_dim, hidden]

        residual = hidden_states
        hidden_states = self._rms_norm(hidden_states, pre_ln)

        # QKV projection: weights are [in, out] layout
        qkv = torch.matmul(hidden_states, qkv_w)
        if qkv_b is not None:
            qkv = qkv + qkv_b

        q_size = self.head_num * self.head_dim
        kv_size = self.kv_head_num * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        seq_len = hidden_states.shape[0]
        q = q.view(seq_len, self.head_num, self.head_dim)
        k = k.view(seq_len, self.kv_head_num, self.head_dim)
        v = v.view(seq_len, self.kv_head_num, self.head_dim)

        cos, sin = self._get_rotary_embedding(position, seq_len)
        q = self._apply_rotary(q, cos, sin)
        k = self._apply_rotary(k, cos, sin)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=0)
            v = torch.cat([past_key_value[1], v], dim=0)
        new_kv = (k, v)

        if self.kv_head_num < self.head_num:
            rep = self.head_num // self.kv_head_num
            k_exp = k.unsqueeze(2).expand(-1, -1, rep, -1).reshape(k.shape[0], self.head_num, self.head_dim)
            v_exp = v.unsqueeze(2).expand(-1, -1, rep, -1).reshape(v.shape[0], self.head_num, self.head_dim)
        else:
            k_exp, v_exp = k, v

        # [seq, heads, dim] → [1, heads, seq, dim] for SDPA
        q = q[-seq_len:].unsqueeze(0).transpose(1, 2)
        k_exp = k_exp.unsqueeze(0).transpose(1, 2)
        v_exp = v_exp.unsqueeze(0).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=(seq_len > 1))
        attn_out = attn_out.squeeze(0).transpose(0, 1).contiguous()
        attn_out = attn_out.view(seq_len, -1)

        # Output projection: [head_num*head_dim, hidden]
        hidden_states = torch.matmul(attn_out, o_w)
        hidden_states = residual + hidden_states

        # FFN with merged gate+up (w13)
        residual = hidden_states
        hidden_states = self._rms_norm(hidden_states, post_ln)

        w13 = lw.get("ffn_weights.intermediate_weight13.kernel")  # [hidden, 2*inter]
        w2 = lw.get(W.ffn_w2)  # [inter, hidden]

        gate_up = torch.matmul(hidden_states, w13)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden_states = torch.matmul(F.silu(gate) * up, w2)
        hidden_states = residual + hidden_states

        return hidden_states, new_kv

    def _get_rotary_embedding(self, position: int, seq_len: int):
        """Generate rotary embeddings for positions [position, position+seq_len)."""
        positions = torch.arange(position, position + seq_len, device=self.proj_weight.device)
        inv_freq = 1.0 / (
            1000000.0 ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.proj_weight.device)
                / self.head_dim
            )
        )
        freqs = torch.outer(positions.float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = torch.cos(emb).unsqueeze(1).to(self.dtype)
        sin = torch.sin(emb).unsqueeze(1).to(self.dtype)
        return cos, sin

    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embedding. x: [seq, heads, dim]"""
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin

    @torch.inference_mode()
    def generate(
        self,
        thinker_hidden_states: torch.Tensor,
        initial_token_ids: torch.Tensor,
        max_new_tokens: int = 4096,
        eos_token_id: int = 8294,
    ) -> torch.Tensor:
        """Generate codec tokens using the talker engine.

        Args:
            thinker_hidden_states: [num_thinker_tokens, 3584] hidden states from thinker
            initial_token_ids: [prompt_len] initial codec token IDs (e.g., speaker BOS)
            max_new_tokens: maximum codec tokens to generate
            eos_token_id: end of sequence token for codec

        Returns:
            [1, num_tokens] generated codec token IDs
        """
        device = self.proj_weight.device
        thinker_hidden_states = thinker_hidden_states.to(device=device, dtype=self.dtype)
        token_ids = initial_token_ids.to(device)

        # Past KV cache for each layer
        past_kvs: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
            None for _ in range(self.num_layers)
        ]

        num_thinker_tokens = thinker_hidden_states.shape[0]
        generated_tokens = []
        position = 0

        # Prefill with initial tokens
        prompt_len = token_ids.shape[0]
        thinker_hs_for_prompt = thinker_hidden_states[:prompt_len]
        if prompt_len > num_thinker_tokens:
            padding = torch.zeros(
                prompt_len - num_thinker_tokens,
                thinker_hidden_states.shape[1],
                device=device, dtype=self.dtype,
            )
            thinker_hs_for_prompt = torch.cat(
                [thinker_hidden_states, padding], dim=0
            )

        hidden = self._embed_and_project(token_ids, thinker_hs_for_prompt)

        for i in range(self.num_layers):
            hidden, new_kv = self._layer_forward(hidden, i, past_kvs[i], position)
            past_kvs[i] = new_kv

        hidden = self._rms_norm(hidden, self.norm_weight)
        logits = F.linear(hidden[-1:], self.codec_head_weight)
        next_token = logits.argmax(dim=-1).squeeze()
        generated_tokens.append(next_token.item())
        position += prompt_len

        # Autoregressive decode
        for step in range(1, max_new_tokens):
            current_token = torch.tensor([next_token.item()], dtype=torch.long, device=device)
            thinker_idx = min(prompt_len + step, num_thinker_tokens - 1)
            thinker_hs = thinker_hidden_states[thinker_idx:thinker_idx+1]

            hidden = self._embed_and_project(current_token, thinker_hs)

            for i in range(self.num_layers):
                hidden, new_kv = self._layer_forward(hidden, i, past_kvs[i], position)
                past_kvs[i] = new_kv

            hidden = self._rms_norm(hidden, self.norm_weight)
            logits = F.linear(hidden, self.codec_head_weight)
            next_token = logits.argmax(dim=-1).squeeze()
            generated_tokens.append(next_token.item())
            position += 1

            if next_token.item() == eos_token_id:
                break

        result = torch.tensor([generated_tokens], dtype=torch.long, device=device)
        logger.info(f"Talker generated {len(generated_tokens)} codec tokens")
        return result
