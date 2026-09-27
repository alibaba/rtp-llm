"""Real preview2 dense/SWA layer building block, independent of model wiring.

The caller supplies already-fused context features and an explicit attention
mask/window contract. This module does not guess auxiliary normalization,
target feature capture boundaries, query alignment, or CP cache ownership.
"""

import flashinfer
import torch
from torch import nn

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules import DenseMLP, LinearFactory, RMSNorm
from rtp_llm.models_py.triton_kernels.dspark_swa import (
    commit_paged_gqa_kv,
    paged_gqa_swa,
)
from rtp_llm.utils.model_weight import W


class MiniMaxM31DSparkLayer(nn.Module):
    """One loaded MXFP8 projection + BF16 GQA SWA + dense SwiGLU-OAI layer."""

    def __init__(self, config, parallelism, weights, hw_kernel_config=None):
        super().__init__()
        self.tp_size = parallelism.get_attn_tp_size()
        attn = config.getAttentionConfigs(self.tp_size)
        self.heads, self.kv_heads = attn.head_num, attn.kv_head_num
        self.dim = attn.size_per_head
        self.q_size, self.kv_size = self.heads * self.dim, self.kv_heads * self.dim
        self.rope_dim = config.attn_config.rope_config.dim
        self.rope_theta = config.attn_config.rope_config.base
        if config.quant_config.get_method() != "MXFP8" or self.dim != 128:
            raise ValueError("preview2 draft layer requires MXFP8 and head_dim=128")

        def linear(mapping, weight, scale):
            return LinearFactory.create_linear_from_weights(
                mapping,
                weight,
                scale,
                quant_config=config.quant_config,
                hw_kernel_config=hw_kernel_config,
            )

        self.qkv_proj = linear(weights, W.attn_qkv_w, W.attn_qkv_s)
        self.o_proj = linear(weights, W.attn_o_w, W.attn_o_s)
        # Commit has no Q consumer. Slice the loaded tensor, not a copy, and
        # avoid computing/discarding 64 Q heads for each committed feature row.
        kv_weights = {
            W.attn_qkv_w: weights[W.attn_qkv_w][self.q_size :],
            W.attn_qkv_s: weights[W.attn_qkv_s][self.q_size :],
        }
        self.kv_proj = linear(kv_weights, W.attn_qkv_w, W.attn_qkv_s)
        self.input_norm = RMSNorm(weights[W.pre_ln_gamma], config.layernorm_eps)
        self.post_norm = RMSNorm(weights[W.post_ln_gamma], config.layernorm_eps)
        self.q_norm = RMSNorm(weights[W.q_ln_gamma], config.layernorm_eps)
        self.k_norm = RMSNorm(weights[W.k_ln_gamma], config.layernorm_eps)
        self.mlp = DenseMLP(
            config.activation_type,
            parallelism,
            weights,
            config.quant_config,
            hw_kernel_config,
            swiglu_oai_params=(config.swiglu_alpha, config.swiglu_limit),
        )

    def _rope(self, q, k, positions):
        flashinfer.apply_rope_pos_ids_inplace(
            q,
            k,
            positions,
            rotary_dim=self.rope_dim,
            interleave=False,
            rope_theta=self.rope_theta,
        )

    def project_context_kv(self, features, positions, *, input_scales=None):
        """Project the same fused context features independently at each layer.

        No layer input norm or attention/MLP is applied to these features.
        An optional MXFP8 input + packed scales lets all five layers reuse one
        feature quantization. Cache ownership/mapping remains the caller's job.
        """
        if features.ndim != 2 or positions.shape != (features.shape[0],):
            raise ValueError("context features/positions must be [T,H] and [T]")
        if input_scales is not None and features.dtype != torch.float8_e4m3fn:
            raise ValueError("prequantized context scales require E4M3 features")
        if features.shape[0] == 0:
            empty = torch.empty(
                0, self.kv_heads, self.dim, dtype=torch.bfloat16, device=features.device
            )
            return empty, empty
        kv = self.kv_proj(features, input_scales=input_scales)
        k, v = kv.split(self.kv_size, dim=-1)
        k = self.k_norm(k.reshape(-1, self.dim)).view(-1, self.kv_heads, self.dim)
        # The paged writer and CP stack accept this interleaved row stride.
        v = v.reshape(-1, self.kv_heads, self.dim)
        # FlashInfer accepts independent Q/K head counts, including zero Q
        # heads. Only K rotates; no dummy Q allocation proportional to tokens.
        empty_q = k.new_empty(k.shape[0], 0, self.dim)
        self._rope(empty_q, k, positions)
        return k, v

    def commit(
        self, features, positions, cache, slot_ids, valid_mask, *, input_scales=None
    ):
        k, v = self.project_context_kv(features, positions, input_scales=input_scales)
        commit_paged_gqa_kv(k, v, cache, slot_ids, valid_mask)

    def forward(
        self,
        hidden,
        positions,
        cache,
        block_table,
        context_lens,
        query_lens,
        *,
        causal: bool,
        window_left: int,
    ):
        """Run real query-block math; query K/V never overwrite context KV."""
        batch, width, hidden_dim = hidden.shape
        if batch == 0:
            return hidden
        residual = hidden.reshape(-1, hidden_dim)
        qkv = self.qkv_proj(self.input_norm(residual))
        q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        q = self.q_norm(q.reshape(-1, self.dim)).view(-1, self.heads, self.dim)
        k = self.k_norm(k.reshape(-1, self.dim)).view(-1, self.kv_heads, self.dim)
        self._rope(q, k, positions.reshape(-1))
        output = paged_gqa_swa(
            q.view(batch, width, self.heads, self.dim),
            k.view(batch, width, self.kv_heads, self.dim),
            v.reshape(batch, width, self.kv_heads, self.dim).contiguous(),
            cache,
            block_table,
            context_lens,
            query_lens,
            causal=causal,
            window_left=window_left,
        )
        output = self.o_proj(output.reshape(batch * width, self.q_size))
        if self.tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        hidden = residual + output
        hidden = hidden + self.mlp(self.post_norm(hidden))
        valid = torch.arange(width, device=hidden.device)[None, :] < query_lens[:, None]
        return torch.where(valid[:, :, None], hidden.view(batch, width, hidden_dim), 0)
