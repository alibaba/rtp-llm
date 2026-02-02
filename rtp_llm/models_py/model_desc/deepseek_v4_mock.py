import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    CausalAttention,
    DenseMLP,
    Embedding,
    FMHAImplBase,
    FusedMoeFactory,
    GroupTopK,
    LinearFactory,
    MlaAttention,
    RMSNorm,
    RMSResNorm,
    SelectTopk,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.triton_kernels.mhc.group_rmsnorm import GroupRMSNorm
from rtp_llm.models_py.triton_kernels.mhc.scaled_dot_product_gate import (
    scaled_dot_product_gate,
)
from rtp_llm.models_py.utils.deepseek_v4_utils import NgramHashMapping
from rtp_llm.ops import HWKernelConfig, MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import KVCache, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class MultiHeadEmbedding(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_id: int,
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.list_of_N = config.engram_vocab_size
        self.num_heads = len(self.list_of_N)
        self.embedding_weight = weights[W.engram_multihead_embedding]

        offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.long),
                torch.cumsum(
                    torch.tensor(self.list_of_N[:-1], dtype=torch.long), dim=0
                ),
            ]
        )
        self.register_buffer("offsets", offsets)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        shifted_input_ids = input_ids + self.offsets
        shifted_input_ids_cpu = shifted_input_ids.cpu()
        output = F.embedding(shifted_input_ids_cpu, self.embedding_weight)
        return output


class ShortConv(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        activation: bool = True,
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.kernel_size = config.kernel_size
        self.dilation = config.max_ngram_size
        self.hc_mult = config.hc_mult
        self.activation = activation

        self.total_channels = self.hidden_size * self.hc_mult
        self.conv_bias = None
        self.conv_weights = weights[W.engram_conv_w]
        # 3 * (4 - 1)
        self.padding_size = self.dilation * (self.kernel_size - 1)
        self.group_norm = GroupRMSNorm(
            weights[W.engram_conv_norms_w],
            group_size=self.hc_mult,
            eps=config.layernorm_eps,
        )
        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor, inputs: PyModelInputs) -> torch.Tensor:
        """
        Input:  (num_tokens, hc_mult, hidden_size)
        Output: (num_tokens, hc_mult, hidden_size)
        """
        num_tokens, hc_mult, hidden_size = x.shape

        assert (
            hc_mult == self.hc_mult
        ), f"Input groups {hc_mult} != hc_mult {self.hc_mult}"

        x_norm = self.group_norm(x.transpose(0, 1).contiguous())
        # x_bct: [hc_mult * hidden_size, num_tokens]
        x_bct = x_norm.permute(0, 2, 1).reshape(-1, num_tokens).contiguous()

        # 需要cache: [self.padding_size, hidden_size]的大小
        # -----魔改一下--casual conv_1d---哎，它的带dialation-------
        y_bct = F.conv1d(
            input=x_bct,
            weight=self.conv_weights,
            bias=None,
            padding=(self.padding_size,),
            dilation=(self.dilation,),
            groups=self.total_channels,
        )
        y_bct = y_bct[..., :num_tokens]
        if self.activation:
            y_bct = self.act_fn(y_bct)
        y = y_bct.transpose(0, 1).view(num_tokens, hc_mult, hidden_size).contiguous()
        # --------------------------------------------------------
        return y


class Engram(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_id,
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hash_mapping = NgramHashMapping(
            engram_vocab_size=config.engram_vocab_size,
            max_ngram_size=config.max_ngram_size,
            n_embed_per_ngram=config.n_embed_per_ngram,
            n_head_per_ngram=config.n_head_per_ngram,
            layer_ids=[layer_id],
            tokenizer_name_or_path=config.ckpt_path,
            pad_id=config.pad_id,
            seed=config.seed,
        )
        self.multi_head_embedding = MultiHeadEmbedding(
            config=config,
            parallelism_config=parallelism_config,
            weights=weights,
            layer_id=layer_id,
        )

        self.short_conv = ShortConv(
            config=config,
            parallelism_config=parallelism_config,
            weights=weights,
            activation=True,
        )

        self.engram_hidden_size = (config.max_ngram_size - 1) * config.n_embed_per_ngram

        # TODO: with block quant scale
        self.value_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.engram_v_proj_w,
            None,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )
        # merge mhc k_projs
        self.key_projs_weights = weights[W.engram_k_projs_w]

        self.q_group_norm = GroupRMSNorm(
            weights[W.engram_q_norms_w],
            group_size=config.hc_mult,
            eps=config.layernorm_eps,
        )
        self.k_group_norm = GroupRMSNorm(
            weights[W.engram_k_norms_w],
            group_size=config.hc_mult,
            eps=config.layernorm_eps,
        )

    def forward(self, inputs: PyModelInputs, hidden_states: torch.Tensor):
        """
        inputs: PyModelInputs,
        hidden_states: [num_tokens, hc_mult, hidden_size]
        """
        # embedding lookup
        # TODO: 修改hash function支持B*S在一个维度并做对应的padding
        input_ids_cpu = inputs.input_ids.to("cpu").unsqueeze(0)
        hash_input_ids = torch.from_numpy(
            self.hash_mapping.hash(input_ids_cpu)[self.layer_id]
        )
        embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)
        embeddings = embeddings.to(hidden_states.device)

        # k_projs: 一次bmm expand ->[hc_mult, num_tokens, engram_hiddensize] @ [hc_mult, engram_hiddensize, hidden_size]
        keys = torch.bmm(
            embeddings.expand(self.config.hc_mult, -1, -1), self.key_projs_weights
        )

        # qk group norm: [hc_mult, num_tokens, hidden_size]
        key_norm = self.k_group_norm(keys)
        query_norm = self.q_group_norm(hidden_states.transpose(0, 1).contiguous())

        # scaled dot product gate
        eps = 1e-6
        gate = scaled_dot_product_gate(
            key_norm, query_norm, self.config.hidden_size, eps
        )
        # gate transpose回去:  [num_tokens，hc_mult, 1]
        gate = gate.unsqueeze(-1).transpose(0, 1).contiguous()

        # [num_tokens, hc_mult, 1] * [num_tokens, 1, hidden_size] -> [num_tokens, hc_mult, hidden_size]
        value = gate * self.value_proj(embeddings.squeeze(0)).unsqueeze(1)

        # casual conv_1d
        output = value + self.short_conv(value, inputs)
        return output

    def forward_decode(self, inputs: PyModelInputs, hidden_states: torch.Tensor):
        return hidden_states


class ManifoldHyperConnections(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hidden_size = config.hidden_size
        self.max_sk_it = config.max_sk_it

        # h_proj: [pre, post, res]
        self._weight = weights[W.mhc_h_proj_w]
        self._bias = weights[W.mhc_h_proj_b]
        self._alpha = weights[W.mhc_h_proj_alpha]

    def forward(self, x: torch.Tensor):
        """
        RMSNorm weight被吸收到h_proj[self._weight]中, 计算与h_proj合并;
        h_pre, h_post, h_res的linear_proj和add bias拆成了三个kernel去实现;
        当前的linear proj先合并;
        """
        _, hc_mult, hidden_size = x.shape
        x = x.reshape(-1, hc_mult * hidden_size)

        # -----------------single kernel--[deep_gemm.tf32_hc_prenorm_gemm]-------
        # (15) norm:  1/r
        r_inv = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True))
        # (14)
        h_out = F.linear(x.float(), self._weight.T)
        # -----------------------------------------------------------------------

        # ------------single kernel------------
        # (16)
        h_pre = h_out[:, :hc_mult]
        h_post = h_out[:, hc_mult : 2 * hc_mult]
        h_res = h_out[:, 2 * hc_mult :]

        h_pre = r_inv * self._alpha[0] * h_pre + self._bias[:hc_mult]
        h_post = r_inv * self._alpha[1] * h_post + self._bias[hc_mult : 2 * hc_mult]
        h_res = r_inv * self._alpha[2] * h_res + self._bias[2 * hc_mult :]

        # (17) (18)
        h_pre = F.sigmoid(h_pre)
        h_post = 2 * F.sigmoid(h_post)
        # --------------------------------------

        # (19) --Sinkhorn-Knopp--single kernel----
        h_res_exp = torch.exp(h_res.reshape(-1, hc_mult, hc_mult))

        def sinkhorn_knopp_torch(x: torch.Tensor, max_it: int = 20):
            t, n, _ = x.shape
            u = torch.ones(
                (
                    t,
                    n,
                ),
                dtype=torch.float32,
                device=x.device,
            )
            v = torch.ones(
                (
                    t,
                    n,
                ),
                dtype=torch.float32,
                device=x.device,
            )

            for i in range(max_it):
                # 防止数值下溢导致除0
                eps_stable = 1e-12
                for _ in range(max_it):
                    # 更新行缩放向量
                    uv = torch.bmm(x, v.unsqueeze(-1)).squeeze(-1)  # (t, n)
                    u = 1.0 / (uv + eps_stable)
                    # 更新列缩放向量
                    vu = torch.bmm(x.transpose(-2, -1), u.unsqueeze(-1)).squeeze(
                        -1
                    )  # (t, n)
                    v = 1.0 / (vu + eps_stable)
            res = u.unsqueeze(-1) * x * v.unsqueeze(-2)
            return res, u, v

        h_res, _, _ = sinkhorn_knopp_torch(h_res_exp, max_it=self.max_sk_it)
        # -----------------------------------------
        return h_pre, h_post, h_res


class DeepseekV4DecoderLayer(nn.Module):
    """Generic MoE decoder layer supporting Dense/MoE hybrid and shared experts."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        moe_config: MoeConfig,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        quant_config = config.quant_config
        self.config = config

        self.self_attn = MlaAttention(
            config.attn_config,
            parallelism_config,
            weights,
            layer_idx,
            config.layernorm_eps,
            quant_config,
            hw_kernel_config,
        )
        if layer_idx not in config.moe_layer_index:
            self.mlp = DenseMLP(
                config.activation_type, parallelism_config, weights, quant_config
            )
        else:
            self.mlp = GenericMoeLayer(
                config,
                parallelism_config,
                weights,
                moe_config,
                max_generate_batch_size,
                enable_cuda_graph=enable_cuda_graph,
            )
        self.engram = None
        if layer_idx in config.engram_layer_index:
            self.engram = Engram(
                config=config,
                parallelism_config=parallelism_config,
                weights=weights,
                layer_id=layer_idx,
            )

        self.input_layernorm = RMSNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )
        self.mhc = ManifoldHyperConnections(
            config=config,
            parallelism_config=parallelism_config,
            weights=weights,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        inputs: PyModelInputs,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        if self.engram is not None:
            # TODO: 查询和与hidden_states的cross attention拆开两部分
            hidden_states = hidden_states + self.engram(inputs, hidden_states)

        # 计算缩放因子
        h_pre, h_post, h_res = self.mhc(hidden_states)

        # TODO: kernel优化-----------------------------------------------------------
        # [num_token, 1, hc_mult] @ [num_token, hc_mult, hidden_size] -> [num_tokens, 1, hidden_size] -> [num_tokens, hidden_size]
        hidden_pre = torch.matmul(
            h_pre.to(torch.bfloat16).unsqueeze(1), hidden_states
        ).squeeze(1)
        # --------------------------------------------------------------------------
        hidden_pre = self.input_layernorm(hidden_pre)
        hidden_pre = self.self_attn(
            hidden_states=hidden_pre, fmha_impl=fmha_impl, kv_cache=kv_cache
        )
        hidden_pre = self.post_attention_layernorm(hidden_pre)
        hidden_pre = self.mlp(hidden_pre)

        # TODO: kernel优化-------------------------------------------------------------
        # hidden_res + hidden_post的部分也被dpsk优化为一个kernel
        output = torch.matmul(h_res.to(torch.bfloat16), hidden_states) + torch.matmul(
            h_post.to(torch.bfloat16).unsqueeze(2), hidden_pre.unsqueeze(1)
        )
        # ----------------------------------------------------------------------------
        return output


class DeepseekV4Model(GptModelBase):
    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config: MoeConfig,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        self.config = model_config
        # Determine attention_type from model_config.attn_config.use_mla
        self.embed_tokens = Embedding(
            model_config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        # Get enable_cuda_graph from py_hw_kernel_config
        enable_cuda_graph = (
            py_hw_kernel_config.enable_cuda_graph
            if py_hw_kernel_config is not None
            else False
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV4DecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[idx],
                    idx,
                    moe_config,
                    max_generate_batch_size,
                    enable_cuda_graph=enable_cuda_graph,
                    hw_kernel_config=py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=model_config.layernorm_eps
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(
                inputs
            )  # pyright: ignore[reportUnreachable]
            fmha_impl.prepare(inputs.attention_inputs)

        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            if i == 0:
                hidden_states = hidden_states[:, None, :].expand(
                    -1, self.config.hc_mult, -1
                )
            hidden_states = decoder_layer(
                inputs,
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
            if i == self.layer_num - 1:
                hidden_states = hidden_states.sum(dim=-2)
        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


__all__ = [
    "Engram",
    "ManifoldHyperConnections",
    "DeepseekV4DecoderLayer",
    "DeepseekV4Model",
]
