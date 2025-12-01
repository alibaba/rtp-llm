import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.models_py.modules.norm import RMSNormTorch
from rtp_llm.utils.model_weight import W


class DeepseekV3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(
    num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


# Find dim range bounds based on rotations
def yarn_find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class DeepseekV3YarnRotaryEmbedding(DeepseekV3RotaryEmbedding):

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (
            self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
            device=device, dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False
        )


def attention_ref(
    batch_size,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    qo_len = q.shape[0] // batch_size
    kv_len = k.shape[0] // batch_size
    num_qo_heads = q.shape[1]
    head_dim_qk = q.shape[2]
    head_dim_vo = v.shape[2]
    logits = (
        torch.einsum(
            "bmhd,bnhd->bhmn",
            q.view(batch_size, qo_len, num_qo_heads, head_dim_qk).float(),
            k.view(batch_size, kv_len, num_qo_heads, head_dim_qk).float(),
        )
        * sm_scale
    )

    if causal:
        mask = torch.arange(kv_len - qo_len, kv_len, device=q.device).unsqueeze(
            1
        ) >= torch.arange(0, kv_len, device=q.device).unsqueeze(0)
    else:
        mask = torch.ones(qo_len, kv_len, device=q.device)

    logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
    lse_ref = torch.logsumexp(logits, -1).transpose(-1, -2)
    p = torch.softmax(logits, dim=-1)
    o_ref = (
        torch.einsum(
            "bhmn,bnhd->bmhd",
            p,
            v.view(batch_size, kv_len, num_qo_heads, head_dim_vo).float(),
        )
        .contiguous()
        .view(batch_size * qo_len, num_qo_heads, head_dim_vo)
        .to(q)
    )

    return o_ref, lse_ref * math.log2(math.e)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    # not transpose since we already transpose weight
    # b, h, s, d = q.shape
    # q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    # b, h, s, d = k.shape
    # k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MlaAttentionRef(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self, config: GptInitModelParameters, weights, layer_idx: Optional[int] = None
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.weights = weights
        self.num_heads = self.config.head_num
        self.qk_nope_head_dim = self.config.nope_head_dim
        self.qk_rope_head_dim = self.config.rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.kv_lora_rank = self.config.kv_lora_rank
        self.v_head_dim = self.config.v_head_dim
        self.q_lora_rank = self.config.q_lora_rank
        self.softmax_scale = self.qk_head_dim ** (-0.5)
        self.token_per_block = self.config.seq_size_per_block

        if self.q_lora_rank > 0:
            self.fused_qkv_a_proj = LinearFactory.create_linear_from_weights(
                weights, W.mla_fusedqkrope_w, W.mla_fusedqkrope_s, None, config
            )
            self.q_a_layernorm = RMSNormTorch(
                weights.get(W.mla_q_a_ln_gamma, None), eps=config.layernorm_eps
            )
            self.q_b_proj = LinearFactory.create_linear_from_weights(
                weights, W.mla_q_b_w, W.mla_q_b_s, None, config
            )
        else:
            self.fused_qkv_proj = LinearFactory.create_linear_from_weights(
                weights,
                W.mla_fusedqkrope_no_lora_w,
                W.mla_fusedqkrope_no_lora_s,
                None,
                config,
            )

        self.kv_a_layernorm = RMSNormTorch(
            weights.get(W.mla_kv_a_ln_gamma, None), eps=config.layernorm_eps
        )

        self.o_proj = LinearFactory.create_linear_from_weights(
            weights, W.attn_o_w, W.attn_o_s, W.attn_o_b, config
        )

        self.rotary_emb = DeepseekV3YarnRotaryEmbedding(
            dim=self.qk_rope_head_dim,
            max_position_embeddings=163840,
            base=10000,
            scaling_factor=1.0,
            original_max_position_embeddings=4096,
            beta_fast=32,
            beta_slow=1,
            mscale=0.707,
            mscale_all_dim=0.707,
        )

        self.scaling = self.qk_head_dim ** (-0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        if self.q_lora_rank > 0:
            fused_qkv = self.fused_qkv_a_proj(hidden_states)
            kv_offset = self.config.q_lora_rank
            q, compressed_kv = torch.split(
                fused_qkv,
                [
                    kv_offset,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                ],
                dim=-1,
            )
            q = self.q_a_layernorm(q.contiguous())
            q = self.q_b_proj(q)
        else:
            fused_qkv = self.fused_qkv_proj(hidden_states)
            kv_offset = self.config.head_num * self.config.size_per_head
            q, compressed_kv = torch.split(
                fused_qkv,
                [
                    kv_offset,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                ],
                dim=-1,
            )
        q_view = q.reshape(-1, self.num_heads, self.qk_head_dim)

        q_nope, q_pe = torch.split(
            q_view, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_layernorm(compressed_kv.contiguous())

        q_pe = q_pe.unsqueeze(0).transpose(0, 1).unsqueeze(0)
        k_pe = k_pe.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        indexs = [hidden_states.shape[0]]
        cos, sin = self.rotary_emb(
            torch.zeros([1]).float().cuda(), seq_len=indexs[-1] + 1
        )
        q_pe, k_pe = apply_rotary_pos_emb(
            q_pe,
            k_pe,
            cos,
            sin,
            torch.tensor(
                indexs, device=torch.device("cuda"), dtype=torch.int32
            ).unsqueeze(0),
        )
        q_pe = q_pe.view(-1, self.num_heads, self.qk_rope_head_dim).to(q_nope.dtype)
        k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim).to(compressed_kv.dtype)

        k_nope_weight = self.weights.get(W.mla_k_nope_w, None)
        v_weight = self.weights.get(W.mla_v_w, None)

        k_nope = F.linear(compressed_kv, k_nope_weight.transpose(0, 1), None)
        k_nope = k_nope.view(-1, self.num_heads, self.qk_nope_head_dim)
        value_states = F.linear(compressed_kv, v_weight.transpose(0, 1), None)
        value_states = value_states.view(-1, self.num_heads, self.qk_nope_head_dim)

        query_states = torch.cat((q_nope, q_pe), dim=-1)
        k = k_pe.new_empty(
            k_pe.size(0), self.num_heads, self.qk_rope_head_dim + self.qk_nope_head_dim
        )
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe

        attn_output, _ = attention_ref(
            1,
            query_states,
            k,
            value_states,
            causal=True,
            sm_scale=self.scaling,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output
