"""Non-Lite GLM-4 GQA decoder and MoE parameters (checkpoint-native names)."""

import torch

from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.layers.linear import QKVParallelLinear, RowParallelLinear
from rtp_llm.models_py.layers.norm import RMSNorm, RMSResNorm
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.deepseek_v3.mlp import DeepSeekV32MLP
from rtp_llm.models_py.new_models.deepseek_v3.moe import (
    DeepSeekV32MoEBlock,
    _select_deepseek_noaux_topk,
)


class Glm4MoeAttention(RtpModule):
    def __init__(
        self,
        *,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        layer_idx,
        tp_size,
        tp_rank,
        quant_config,
        params_dtype,
        use_qk_norm,
        qkv_bias=True,
        rms_norm_eps,
        prefix,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.use_qk_norm = use_qk_norm
        linear = dict(
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            params_dtype=params_dtype,
        )
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            bias=qkv_bias,
            prefix=f"{prefix}.qkv_proj",
            **linear,
        )
        self.o_proj = RowParallelLinear(
            input_size=num_heads * head_dim,
            output_size=hidden_size,
            bias=False,
            prefix=f"{prefix}.o_proj",
            **linear,
        )
        if use_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps, params_dtype=params_dtype)
            self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps, params_dtype=params_dtype)
            self.num_heads_per_partition = self.qkv_proj.num_heads_per_partition
            self.num_kv_heads_per_partition = self.qkv_proj.num_kv_heads_per_partition
            self.q_size, self.kv_size = self.qkv_proj.q_size, self.qkv_proj.kv_size

    def forward(self, hidden_states, fmha_impl, kv_cache=None):
        qkv = self.qkv_proj(hidden_states)
        if self.use_qk_norm:
            if qkv.is_cuda:
                from rtp_llm.models_py.new_models.qwen3.language import Qwen3Attention

                qkv = Qwen3Attention._apply_qk_norm(self, qkv)
            else:
                q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
                q = self.q_norm(q.unflatten(-1, (-1, self.head_dim))).flatten(-2)
                k = self.k_norm(k.unflatten(-1, (-1, self.head_dim))).flatten(-2)
                qkv = torch.cat((q, k, v), dim=-1)
        output = fmha_impl.forward(qkv, kv_cache, self.layer_idx)
        return self.o_proj(output.reshape(*hidden_states.shape[:-1], -1).contiguous())


class Glm4MoeBlock(DeepSeekV32MoEBlock):
    """Reuse expert/storage/runtime owners, preserve GLM's actual old router.

    Main GenericMoeLayer uses GroupTopK when correction bias exists, otherwise
    SelectTopk (softmax, without routed scaling). Unlike the DeepSeek reference,
    GroupTopK normalizes even a single selected expert when requested.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.select_topk = None
        if not self.correction_bias and get_device_type() == DeviceType.Cuda:
            from rtp_llm.models_py.modules import SelectTopk

            self.select_topk = SelectTopk(config=kwargs["model_config"])

    def route(self, logits, *, ids_dtype=None):
        logits = logits.float()
        if ids_dtype is None:
            ids_dtype = self.experts.topk_ids_dtype
        if self.correction_bias and self._use_fast_group_topk and logits.is_cuda:
            weights = torch.empty(
                (logits.shape[0], self.top_k), dtype=torch.float32, device=logits.device
            )
            ids = torch.empty(weights.shape, dtype=ids_dtype, device=logits.device)
            self.group_topk(
                topk_weights=weights,
                topk_ids=ids,
                scores=logits,
                correction_bias=self.gate.e_score_correction_bias,
                n_group=self.n_group,
                topk_group=self.topk_group,
                topk=self.top_k,
                renormalize=self.has_moe_norm,
                routed_scaling_factor=self.routed_scaling_factor,
            )
            return weights, ids
        if self.correction_bias:
            weights, ids = _select_deepseek_noaux_topk(
                logits,
                self.gate.e_score_correction_bias,
                top_k=self.top_k,
                n_group=self.n_group,
                topk_group=self.topk_group,
                renormalize=False,
                routed_scaling_factor=1.0,
            )
            if self.has_moe_norm:
                weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)
            return weights * self.routed_scaling_factor, ids.to(ids_dtype)
        if self.select_topk is not None and logits.is_cuda:
            weights = torch.empty(
                (logits.shape[0], self.top_k), dtype=torch.float32, device=logits.device
            )
            ids = torch.empty(weights.shape, dtype=ids_dtype, device=logits.device)
            self.select_topk(logits, ids, weights)
            return weights, ids
        weights, ids = logits.softmax(-1).topk(self.top_k, dim=-1, sorted=False)
        if self.has_moe_norm:
            weights = weights / weights.sum(-1, keepdim=True)
        return weights, ids.to(ids_dtype)

    def forward(self, hidden_states):
        if hidden_states.dim() != 2:
            raise ValueError("GLM4 MoE expects a two-dimensional token matrix")
        weights, ids = self.route(self.gate(hidden_states))
        if self.fake_balance_expert is not None:
            self.fake_balance_expert(ids, weights)
        return self._merge_experts(hidden_states, weights, ids)


class Glm4MoeDecoderLayer(RtpModule):
    def __init__(
        self,
        cfg,
        layer_idx,
        *,
        dense_width,
        shared_width,
        moe_layers,
        use_qk_norm,
        correction_bias,
        qkv_bias=True,
    ):
        super().__init__()
        prefix = f"layers.{layer_idx}"
        self.input_layernorm = RMSResNorm(
            cfg["hidden_size"],
            eps=cfg["rms_norm_eps"],
            params_dtype=cfg["params_dtype"],
        )
        self.post_attention_layernorm = RMSResNorm(
            cfg["hidden_size"],
            eps=cfg["rms_norm_eps"],
            params_dtype=cfg["params_dtype"],
        )
        self.self_attn = Glm4MoeAttention(
            hidden_size=cfg["hidden_size"],
            num_heads=cfg["num_heads"],
            num_kv_heads=cfg["num_kv_heads"],
            head_dim=cfg["head_dim"],
            layer_idx=layer_idx,
            tp_size=cfg["attn_tp_size"],
            tp_rank=cfg["attn_tp_rank"],
            quant_config=cfg["quant_config"],
            params_dtype=cfg["params_dtype"],
            use_qk_norm=use_qk_norm,
            qkv_bias=qkv_bias,
            rms_norm_eps=cfg["rms_norm_eps"],
            prefix=f"{prefix}.self_attn",
        )
        common = dict(
            hidden_size=cfg["hidden_size"],
            tp_size=cfg["ffn_tp_size"],
            tp_rank=cfg["ffn_tp_rank"],
            quant_config=cfg["quant_config"],
            params_dtype=cfg["params_dtype"],
            prefix=f"{prefix}.mlp",
        )
        if layer_idx not in moe_layers:
            self.mlp = DeepSeekV32MLP(
                intermediate_size=dense_width, reduce_output=True, **common
            )
        else:
            model = cfg["model_config"]
            self.mlp = Glm4MoeBlock(
                moe_intermediate_size=cfg["moe_intermediate_size"],
                num_experts=cfg["num_experts"],
                top_k=cfg["top_k"],
                layer_idx=layer_idx,
                ep_size=cfg["ep_size"],
                ep_rank=cfg["ep_rank"],
                model_config=model,
                parallelism_config=cfg["parallelism_config"],
                moe_config=cfg["moe_config"],
                has_shared_expert=shared_width > 0,
                shared_expert_intermediate_size=shared_width,
                scoring_func=model.scoring_func,
                routed_scaling_factor=model.routed_scaling_factor,
                n_group=model.moe_n_group,
                topk_group=model.moe_topk_group,
                topk_method="noaux_tc" if correction_bias else "greedy",
                has_moe_norm=model.has_moe_norm,
                correction_bias=correction_bias,
                **common,
            )

    def forward(self, hidden_states, residual, fmha_impl, kv_cache=None):
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(hidden_states, fmha_impl, kv_cache)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual
