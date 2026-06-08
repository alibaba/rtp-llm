"""Talker inference: pure PyTorch autoregressive codec token generation.

The talker takes thinker hidden states and generates codec tokens that
encode speech. This follows the HuggingFace Qwen2.5-Omni implementation.
"""
import json
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TalkerRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(self.weight.dtype)


class TalkerRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=32768, base=1000000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        # position_ids: [3, batch, seq_len] for mrope
        # Use temporal dimension (index 0) for simplicity in pure inference
        if position_ids.dim() == 3:
            pos = position_ids[0]  # [batch, seq_len]
        else:
            pos = position_ids

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(pos.shape[0], -1, 1)
        position_ids_expanded = pos[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section):
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.chunk(len(mrope_section) * 3, dim=-1))], dim=-1)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.chunk(len(mrope_section) * 3, dim=-1))], dim=-1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_simple(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TalkerMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TalkerAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, hidden_states, cos, sin, attention_mask=None, past_key_value=None):
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb_simple(q, k, cos.unsqueeze(1), sin.unsqueeze(1))

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=(attention_mask is None and q_len > 1),
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output), new_kv


class TalkerDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, rms_norm_eps):
        super().__init__()
        self.self_attn = TalkerAttention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.mlp = TalkerMLP(hidden_size, intermediate_size)
        self.input_layernorm = TalkerRMSNorm(hidden_size, rms_norm_eps)
        self.post_attention_layernorm = TalkerRMSNorm(hidden_size, rms_norm_eps)

    def forward(self, hidden_states, cos, sin, attention_mask=None, past_key_value=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_kv = self.self_attn(hidden_states, cos, sin, attention_mask, past_key_value)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_kv


class TalkerModel(nn.Module):
    """Pure PyTorch talker transformer for Qwen2.5-Omni codec token generation.

    Architecture: embed_tokens(8448→3584) → thinker_to_talker_proj(3584→896)
                  → 24 transformer layers (896-dim) → codec_head(896→8448)
    """

    def __init__(
        self,
        vocab_size: int = 8448,
        embedding_size: int = 3584,
        hidden_size: int = 896,
        num_layers: int = 24,
        num_heads: int = 12,
        num_kv_heads: int = 4,
        head_dim: int = 128,
        intermediate_size: int = 18944,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        max_position_embeddings: int = 32768,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed_tokens = nn.Embedding(vocab_size, embedding_size)
        self.thinker_to_talker_proj = nn.Linear(embedding_size, hidden_size)
        self.codec_head = nn.Linear(hidden_size, vocab_size, bias=False)

        self.layers = nn.ModuleList([
            TalkerDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                rms_norm_eps=rms_norm_eps,
            )
            for _ in range(num_layers)
        ])
        self.norm = TalkerRMSNorm(hidden_size, rms_norm_eps)
        self.rotary_emb = TalkerRotaryEmbedding(
            head_dim, max_position_embeddings=max_position_embeddings, base=rope_theta
        )

        self.codec_bos_token = 8293
        self.codec_eos_token = 8294
        self.codec_pad_token = 8292
        self.codec_mask_token = 8296

    def forward_one_step(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Run one forward step through the transformer.

        Args:
            inputs_embeds: [batch, seq_len, hidden_size] — already projected
            position_ids: [batch, seq_len] position indices
            past_key_values: list of (k, v) tuples per layer
            attention_mask: optional attention mask
        Returns:
            logits: [batch, seq_len, vocab_size]
            new_past_key_values: updated KV cache
        """
        cos, sin = self.rotary_emb(inputs_embeds, position_ids.unsqueeze(0))

        hidden_states = inputs_embeds
        new_past_key_values = []

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            hidden_states, new_kv = layer(hidden_states, cos, sin, attention_mask, past_kv)
            new_past_key_values.append(new_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.codec_head(hidden_states).float()

        return logits, new_past_key_values

    @torch.no_grad()
    def generate(
        self,
        thinker_hidden_states: List[torch.Tensor],
        thinker_token_embeds: List[torch.Tensor],
        input_ids: torch.LongTensor,
        speaker_bos_token: int,
        thinker_embed_tokens: nn.Embedding,
        max_new_tokens: int = 4096,
        temperature: float = 0.9,
        top_k: int = 40,
        top_p: float = 0.8,
        repetition_penalty: float = 1.05,
    ) -> torch.LongTensor:
        """Generate codec tokens autoregressively.

        Follows HF Qwen2.5-Omni's generate() exactly:
        1. Construct thinker_reply_part from per-token hidden states + embeddings
        2. Build initial inputs_embeds with speaker BOS
        3. Autoregressive loop: codec_embed + thinker_reply_part[:,:1,:] → project → transformer → sample

        Args:
            thinker_hidden_states: list of [1, 1, 3584] tensors, one per generated token
                                   (last-layer hidden state for each token)
            thinker_token_embeds: list of [1, 1, 3584] tensors, one per generated token
                                  (first-layer embedding for each token)
            input_ids: original input token IDs [1, prompt_len]
            speaker_bos_token: speaker-specific BOS token (e.g. 151872 for Chelsie)
            thinker_embed_tokens: thinker's embedding layer for getting BOS/EOS/pad embeddings
            max_new_tokens: maximum codec tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering
            top_p: nucleus sampling threshold
            repetition_penalty: penalty for repeated tokens
        Returns:
            codec_tokens: [1, num_codec_tokens] generated codec token IDs
        """
        device = thinker_hidden_states[0].device
        dtype = thinker_hidden_states[0].dtype

        # Build thinker_reply_part: cat of (hidden_states[i] + token_embeds[i]) for generated tokens
        # thinker_hidden_states[0] is the prompt's hidden state, [1:] are generated tokens
        thinker_reply_part = torch.cat(
            [h + e for h, e in zip(thinker_hidden_states[1:], thinker_token_embeds[1:])],
            dim=1,
        )  # [1, num_generated_tokens-1, 3584]

        # Build initial talker inputs_embeds from prompt hidden state
        # talker_inputs_embeds = prompt_hidden + prompt_embed + speaker_bos_embed + first_reply
        prompt_hidden_embed = thinker_hidden_states[0] + thinker_token_embeds[0]  # [1, prompt_len, 3584]

        speaker_bos = torch.tensor([[speaker_bos_token]], dtype=torch.long, device=device)
        speaker_bos_embed = thinker_embed_tokens(speaker_bos)  # [1, 1, 3584]

        talker_inputs_embeds = torch.cat([
            prompt_hidden_embed,
            speaker_bos_embed,
            thinker_reply_part[:, :1, :],
        ], dim=1)  # [1, prompt_len+2, 3584]

        # Append EOS and PAD embeddings to thinker_reply_part for end-of-sequence
        text_eos_token = torch.tensor([[151861]], dtype=torch.long, device=device)
        text_pad_token = torch.tensor([[151859]], dtype=torch.long, device=device)
        eos_embed = thinker_embed_tokens(text_eos_token)  # [1, 1, 3584]
        pad_embed = thinker_embed_tokens(text_pad_token)  # [1, 1, 3584]

        thinker_reply_part = torch.cat([
            thinker_reply_part[:, 1:, :],
            eos_embed,
            pad_embed,
        ], dim=1)

        # Build initial codec input_ids (all mask tokens for prompt, then pad+bos)
        prompt_len = prompt_hidden_embed.shape[1]
        codec_input_ids = torch.cat([
            torch.full((1, prompt_len), self.codec_mask_token, dtype=torch.long, device=device),
            torch.tensor([[self.codec_pad_token, self.codec_bos_token]], dtype=torch.long, device=device),
        ], dim=1)

        # Add codec BOS and PAD embeddings to last two positions (HF logic)
        codec_bos_embed = self.embed_tokens(
            torch.tensor([self.codec_bos_token], dtype=torch.long, device=device)
        )
        codec_pad_embed = self.embed_tokens(
            torch.tensor([self.codec_pad_token], dtype=torch.long, device=device)
        )
        talker_inputs_embeds[:, -1, :] += codec_bos_embed
        talker_inputs_embeds[:, -2, :] += codec_pad_embed

        # Project to talker hidden size
        projected = self.thinker_to_talker_proj(talker_inputs_embeds)  # [1, seq_len, 896]

        # Position IDs for initial sequence
        seq_len = projected.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # Prefill: run the full initial sequence through transformer
        logits, past_key_values = self.forward_one_step(projected, position_ids)

        # Sample first codec token
        next_token_logits = logits[:, -1, :]
        next_token = self._sample_token(
            next_token_logits, temperature, top_k, top_p, repetition_penalty, []
        )

        generated_tokens = [next_token.item()]
        current_pos = seq_len

        # Autoregressive generation loop
        for step in range(max_new_tokens - 1):
            if next_token.item() in (self.codec_eos_token, self.codec_pad_token):
                break

            # Build next step's input: codec_embed(token) + thinker_reply_part[:,:1,:]
            codec_embed = self.embed_tokens(next_token.view(1, 1))  # [1, 1, 3584]

            # Add thinker_reply_part contribution
            if thinker_reply_part.shape[1] > 0:
                step_embed = codec_embed + thinker_reply_part[:, :1, :]
                if thinker_reply_part.shape[1] > 1:
                    thinker_reply_part = thinker_reply_part[:, 1:, :]
                else:
                    thinker_reply_part = thinker_reply_part[:, :0, :]
            else:
                step_embed = codec_embed

            # Project and run through transformer
            step_projected = self.thinker_to_talker_proj(step_embed)
            step_position = torch.tensor([[current_pos]], device=device)

            logits, past_key_values = self.forward_one_step(
                step_projected, step_position, past_key_values
            )

            next_token_logits = logits[:, -1, :]
            next_token = self._sample_token(
                next_token_logits, temperature, top_k, top_p,
                repetition_penalty, generated_tokens
            )
            generated_tokens.append(next_token.item())
            current_pos += 1

        logger.info(f"Talker generated {len(generated_tokens)} codec tokens")
        return torch.tensor([generated_tokens], dtype=torch.long, device=device)

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        generated_tokens: List[int],
    ) -> torch.Tensor:
        # Apply repetition penalty
        if repetition_penalty != 1.0 and generated_tokens:
            for token_id in set(generated_tokens):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        # Suppress codec BOS token
        logits[0, self.codec_bos_token] = float("-inf")

        # Temperature
        if temperature > 0:
            logits = logits / temperature
        else:
            return logits.argmax(dim=-1)

        # Top-k filtering
        if top_k > 0:
            top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            threshold = top_k_values[0, -1]
            logits[logits < threshold] = float("-inf")

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


class TalkerInference:
    """High-level talker inference wrapper.

    Loads weights from safetensors and provides generate() interface.
    """

    def __init__(self, model: TalkerModel, device: str = "cpu"):
        self.model = model
        self.device = device

    @classmethod
    def from_pretrained(cls, ckpt_path: str, device: str = "cpu") -> "TalkerInference":
        config_path = os.path.join(ckpt_path, "config.json")
        with open(config_path) as f:
            root_config = json.load(f)

        tc = root_config.get("talker_config", root_config)

        model = TalkerModel(
            vocab_size=tc.get("vocab_size", 8448),
            embedding_size=tc.get("embedding_size", 3584),
            hidden_size=tc.get("hidden_size", 896),
            num_layers=tc.get("num_hidden_layers", 24),
            num_heads=tc.get("num_attention_heads", 12),
            num_kv_heads=tc.get("num_key_value_heads", 4),
            head_dim=tc.get("head_dim", 128),
            intermediate_size=tc.get("intermediate_size", 18944),
            rms_norm_eps=tc.get("rms_norm_eps", 1e-6),
            rope_theta=tc.get("rope_theta", 1000000.0),
            max_position_embeddings=tc.get("max_position_embeddings", 32768),
        )

        # Update special tokens from config
        model.codec_bos_token = tc.get("tts_codec_start_token_id", 8293)
        model.codec_eos_token = tc.get("tts_codec_end_token_id", 8294)
        model.codec_pad_token = tc.get("tts_codec_pad_token_id", 8292)
        model.codec_mask_token = tc.get("tts_codec_mask_token_id", 8296)

        # Load weights from safetensors
        index_path = os.path.join(ckpt_path, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)

        talker_files = set()
        for name, shard in index["weight_map"].items():
            if name.startswith("talker."):
                talker_files.add(shard)

        from safetensors.torch import load_file

        state_dict = {}
        for shard_file in talker_files:
            shard_path = os.path.join(ckpt_path, shard_file)
            shard_weights = load_file(shard_path)
            for k, v in shard_weights.items():
                if k.startswith("talker."):
                    new_key = cls._map_weight_key(k[len("talker."):])
                    if new_key is not None:
                        state_dict[new_key] = v

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Talker missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        if unexpected:
            logger.warning(f"Talker unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

        model = model.to(device).to(torch.bfloat16)
        model.eval()
        logger.info(f"TalkerModel loaded from {ckpt_path} on {device}")
        return cls(model, device)

    @staticmethod
    def _map_weight_key(key: str) -> Optional[str]:
        """Map HF weight keys to our model's key naming."""
        # Direct mappings for talker-specific weights
        if key == "codec_head.weight":
            return "codec_head.weight"
        if key == "thinker_to_talker_proj.weight":
            return "thinker_to_talker_proj.weight"
        if key == "thinker_to_talker_proj.bias":
            return "thinker_to_talker_proj.bias"

        # model.embed_tokens
        if key.startswith("model.embed_tokens."):
            return key.replace("model.embed_tokens.", "embed_tokens.")

        # model.norm
        if key.startswith("model.norm."):
            return key.replace("model.norm.", "norm.")

        # model.layers.N.* → layers.N.*
        if key.startswith("model.layers."):
            rest = key[len("model.layers."):]
            # Parse layer index
            parts = rest.split(".", 1)
            if len(parts) != 2:
                return None
            layer_idx, subkey = parts

            # self_attn.{q,k,v,o}_proj
            if subkey.startswith("self_attn."):
                return f"layers.{layer_idx}.self_attn.{subkey[len('self_attn.'):]}"

            # mlp.{gate,up,down}_proj
            if subkey.startswith("mlp."):
                return f"layers.{layer_idx}.mlp.{subkey[len('mlp.'):]}"

            # input_layernorm, post_attention_layernorm
            if subkey.startswith("input_layernorm."):
                return f"layers.{layer_idx}.input_layernorm.{subkey[len('input_layernorm.'):]}"
            if subkey.startswith("post_attention_layernorm."):
                return f"layers.{layer_idx}.post_attention_layernorm.{subkey[len('post_attention_layernorm.'):]}"

        return None

    def generate(
        self,
        thinker_hidden_states: List[torch.Tensor],
        thinker_token_embeds: List[torch.Tensor],
        input_ids: torch.LongTensor,
        speaker_bos_token: int,
        thinker_embed_tokens: nn.Embedding,
        **kwargs,
    ) -> torch.LongTensor:
        return self.model.generate(
            thinker_hidden_states=thinker_hidden_states,
            thinker_token_embeds=thinker_token_embeds,
            input_ids=input_ids,
            speaker_bos_token=speaker_bos_token,
            thinker_embed_tokens=thinker_embed_tokens,
            **kwargs,
        )
