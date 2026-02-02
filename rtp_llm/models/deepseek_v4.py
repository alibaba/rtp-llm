import functools
import json
import os
from typing import List, Optional

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.deepseek_v2 import DeepSeekV2, DeepSeekV2Weight
from rtp_llm.models_py.model_desc.deepseek_v4_mock import DeepseekV4Model
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeModel
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops import MlaOpsType
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    concat_0,
    identity,
    stack_0,
    yarn_get_mscale,
)


def merge_2d_tensors_list(ts: List[torch.Tensor], dim: int):
    """
    Merge a list of tensors along the specified dimension.

    Args:
        ts: List of tensors to merge
        dim: Dimension along which to concatenate tensors

    Returns:
        Merged tensor

    Raises:
        ValueError: If tensor list is empty or tensors have incompatible shapes
    """
    if not ts:
        raise ValueError("Cannot merge empty tensor list")

    if len(ts) == 1:
        return ts[0].contiguous()

    # Check that all non-concat dimensions match
    ref_shape = list(ts[0].shape)
    ndim = len(ref_shape)
    for i, t in enumerate(ts[1:], 1):
        for d in range(ndim):
            if d != dim and t.shape[d] != ref_shape[d]:
                raise ValueError(
                    f"All tensors must have the same size in non-concat dimensions. "
                    f"Dimension {d}: tensor 0 has size {ref_shape[d]}, "
                    f"but tensor {i} has size {t.shape[d]}"
                )
    merged = torch.concat(ts, dim=dim).contiguous()
    return merged


def merge_2d_tensors_list_transpose(ts: List[torch.Tensor], dim: int):
    """
    Transpose each tensor and then merge along the specified dimension.

    Args:
        ts: List of 2D tensors to transpose and merge
        dim: Dimension along which to concatenate tensors after transposing

    Returns:
        Merged tensor

    Raises:
        ValueError: If tensor list is empty or tensors have incompatible shapes
    """
    if not ts:
        raise ValueError("Cannot merge empty tensor list")

    if len(ts) == 1:
        return ts[0].T.contiguous()

    # Transpose all tensors
    transposed = [t.T for t in ts]

    # Check that all non-concat dimensions match after transpose
    ref_shape = list(transposed[0].shape)
    for i, t in enumerate(transposed[1:], 1):
        for d in range(2):
            if d != dim and t.shape[d] != ref_shape[d]:
                raise ValueError(
                    f"All tensors must have the same size in non-concat dimensions after transpose. "
                    f"Dimension {d}: tensor 0 has size {ref_shape[d]}, "
                    f"but tensor {i} has size {t.shape[d]}"
                )
    merged = torch.concat(transposed, dim=dim).contiguous()
    return merged


class DeepSeekV4Weight(DeepSeekV2Weight):
    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config,
        hw_kernel_config,
        kv_cache_config,
        merge_lora: bool = False,
        vit_config=None,
        **kwargs,
    ):
        super().__init__(
            model_config=model_config,
            parallelism_config=parallelism_config,
            hw_kernel_config=hw_kernel_config,
            kv_cache_config=kv_cache_config,
            merge_lora=merge_lora,
            vit_config=vit_config,
            **kwargs,
        )

    def _get_engram_layer_weight_info(self, layer_id: int):
        # hidden dimension multiplier
        hc_mult = self.model_config.hc_mult
        n_head_per_ngram = self.model_config.n_head_per_ngram
        n_embed_per_ngram = self.model_config.n_embed_per_ngram

        attn_config = AttnConfig(
            hidden_size=n_embed_per_ngram,
            size_per_head=n_embed_per_ngram // n_head_per_ngram,
            head_num=n_head_per_ngram,
            head_num_kv=n_head_per_ngram,
        )

        engram_weights = [
            AttnAtomicWeight(
                W.engram_multihead_embedding,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.engram.embedding.weight", identity
                    ),
                ],
                identity,
                config=attn_config,
            ),
            AttnAtomicWeight(
                W.engram_v_proj_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.engram.value_proj.weight", identity
                    ),
                ],
                identity,
                config=attn_config,
            ),
            AttnAtomicWeight(
                W.engram_k_projs_w,
                [
                    CkptWeightInfo(
                        f"model.layers.{{i}}.engram.key_projs.{j}.weight", identity
                    )
                    for j in range(hc_mult)
                ],
                stack_0,
                config=attn_config,
            ),
            AtomicWeight(
                W.engram_q_norms_w,
                [
                    CkptWeightInfo(
                        f"model.layers.{{i}}.engram.q_norms.{j}.weight", identity
                    )
                    for j in range(hc_mult)
                ],
                stack_0,
                data_type=torch.bfloat16,
            ),
            AtomicWeight(
                W.engram_k_norms_w,
                [
                    CkptWeightInfo(
                        f"model.layers.{{i}}.engram.k_norms.{j}.weight", identity
                    )
                    for j in range(hc_mult)
                ],
                stack_0,
                data_type=torch.bfloat16,
            ),
            AtomicWeight(
                W.engram_conv_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.engram.short_conv.conv.weight", identity
                    ),
                ],
                identity,
                data_type=torch.bfloat16,
            ),
            AtomicWeight(
                W.engram_conv_norms_w,
                [
                    CkptWeightInfo(
                        f"model.layers.{{i}}.engram.short_conv.norms.{j}.weight",
                        identity,
                    )
                    for j in range(hc_mult)
                ],
                stack_0,
                data_type=torch.bfloat16,
            ),
        ]
        return engram_weights

    def _get_mhc_layer_weight_info(self, layer_id: int):
        # mhc weights
        mhc_h_proj_w = "mhc.h_proj.kernel"
        mhc_h_proj_b = "mhc.h_proj.bias"
        mhc_h_proj_alpha = "mhc.h_proj.alpha"

        mhc_weights = [
            AtomicWeight(
                W.mhc_h_proj_w,
                [
                    CkptWeightInfo("model.layers.{i}.mhc.h_pre_proj.weight", identity),
                    CkptWeightInfo(
                        "model.layers.{i}.mhc.h_post_proj.weight",
                        identity,
                    ),
                    CkptWeightInfo(
                        "model.layers.{i}.mhc.h_res_proj.weight",
                        identity,
                    ),
                ],
                functools.partial(merge_2d_tensors_list, dim=1),
                data_type=torch.float32,
            ),
            AtomicWeight(
                W.mhc_h_proj_b,
                [
                    CkptWeightInfo("model.layers.{i}.mhc.h_pre_proj.bias", identity),
                    CkptWeightInfo("model.layers.{i}.mhc.h_post_proj.bias", identity),
                    CkptWeightInfo("model.layers.{i}.mhc.h_res_proj.bias", identity),
                ],
                concat_0,
                data_type=torch.float32,
            ),
            AtomicWeight(
                W.mhc_h_proj_alpha,
                [
                    CkptWeightInfo("model.layers.{i}.mhc.alpha_pre", identity),
                    CkptWeightInfo("model.layers.{i}.mhc.alpha_post", identity),
                    CkptWeightInfo("model.layers.{i}.mhc.alpha_res", identity),
                ],
                concat_0,
                data_type=torch.float32,
            ),
        ]
        return mhc_weights

    def _get_weight_info(self):
        weight_info = super()._get_weight_info()
        layer_weights = weight_info.layer_weights

        for layer in range(self._num_layers):
            if layer in self.model_config.engram_layer_index:
                layer_weights[layer].extend(self._get_engram_layer_weight_info(layer))
            layer_weights[layer].extend(self._get_mhc_layer_weight_info(layer))
        return ModelWeightInfo(weights=weight_info.weights, layer_weights=layer_weights)


class DeepSeekV4(DeepSeekV2):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = ModelConfig()
        config.attn_config.head_num = 0
        config.attn_config.kv_head_num = 0
        config.attn_config.size_per_head = 0
        config.num_layers = 0
        config.inter_size = 0
        config.vocab_size = 102400
        config.max_seq_len = 8192
        config.norm_type = "rmsnorm"
        config.has_post_decoder_layernorm = True
        # config.activation_type = "gated-silu"
        config.activation_type = "SiGLU"
        DeepSeekV4._from_hf(config, ckpt_path)
        return config

    def support_cuda_graph(self) -> bool:
        return True

    def _create_python_model(self) -> Optional[GptModelBase]:
        model_config = self.model_config
        parallelism_config = self.parallelism_config
        fmha_config = self.fmha_config
        py_hw_kernel_config = self.hw_kernel_config
        moe_config = self.moe_config
        max_generate_batch_size = self.max_generate_batch_size

        # Use GenericMoeModel with new config architecture
        # attention_type is determined from model_config.attn_config.use_mla

        self.py_model = DeepseekV4Model(
            model_config,
            parallelism_config,
            self.weight,
            moe_config,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )

    @staticmethod
    def _from_hf(config: ModelConfig, ckpt_path: str):
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return
        with open(config_path) as reader:
            content = reader.read()
            config_json = json.loads(content)
            config.inter_size = config_json["intermediate_size"]
            config.attn_config.head_num = config_json["num_attention_heads"]
            config.attn_config.kv_head_num = config_json.get(
                "num_key_value_heads", config.attn_config.head_num
            )
            config.num_layers = config_json["num_hidden_layers"]
            config.attn_config.rope_config.base = int(
                config_json.get("rope_theta", config.attn_config.rope_config.base)
            )
            config.vocab_size = config_json["vocab_size"]
            config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)
            config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
            config.hidden_size = config_json["hidden_size"]

            # MLA config
            config.attn_config.use_mla = True
            q_lora_rank = config_json.get("q_lora_rank")
            config.attn_config.q_lora_rank = (
                int(q_lora_rank) if q_lora_rank is not None else 0
            )
            kv_lora_rank = config_json.get("kv_lora_rank")
            config.attn_config.kv_lora_rank = (
                int(kv_lora_rank) if kv_lora_rank is not None else 0
            )
            config.attn_config.nope_head_dim = config_json["qk_nope_head_dim"]
            config.attn_config.rope_head_dim = config_json["qk_rope_head_dim"]
            config.attn_config.v_head_dim = config_json["v_head_dim"]
            config.attn_config.size_per_head = (
                config.attn_config.nope_head_dim + config.attn_config.rope_head_dim
            )
            config.attn_config.rope_config.dim = config.attn_config.rope_head_dim

            # yarn rotary config
            if config.mla_ops_type != MlaOpsType.MHA:
                config.attn_config.rope_config.style = 0
            else:
                config.attn_config.rope_config.style = 5
            rope_scaling = config_json.get("rope_scaling")
            config.attn_config.rope_config.scale = rope_scaling["factor"]
            config.attn_config.rope_config.factor1 = float(
                rope_scaling.get("beta_slow", 1)
            )
            config.attn_config.rope_config.factor2 = float(
                rope_scaling.get("beta_fast", 32)
            )
            config.attn_config.rope_config.max_pos = rope_scaling[
                "original_max_position_embeddings"
            ]

            scaling_factor = rope_scaling["factor"]
            mscale = rope_scaling["mscale"]
            mscale_all_dim = rope_scaling["mscale_all_dim"]
            config.deepseek_rope_mscale = mscale
            config.deepseek_mscale_all_dim = mscale_all_dim
            config.attn_config.rope_config.mscale = yarn_get_mscale(
                scaling_factor, mscale
            ) / yarn_get_mscale(scaling_factor, mscale_all_dim)
            config.attn_config.rope_config.offset = config.attn_config.nope_head_dim

            # softmax scale config
            softmax_mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
            config.attn_config.softmax_extra_scale = softmax_mscale * softmax_mscale

            # MOE config
            if "scoring_func" in config_json:
                scoring_func = config_json["scoring_func"]
                if scoring_func == "softmax":
                    config.scoring_func = 0
                elif scoring_func == "sigmoid":
                    config.scoring_func = 1
                else:
                    raise ValueError(f"Unknown scoring_func: {scoring_func}")

            config.routed_scaling_factor = config_json["routed_scaling_factor"]
            config.moe_k = config_json["num_experts_per_tok"]
            config.expert_num = config_json["n_routed_experts"]
            moe_intermediate_size = config_json["moe_intermediate_size"]
            config.moe_n_group = config_json.get("n_group", 1)
            config.moe_topk_group = config_json.get("topk_group", 1)

            n_shared_experts = config_json["n_shared_experts"]
            config.inter_size = n_shared_experts * moe_intermediate_size

            config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)
            config.has_moe_norm = config_json.get("norm_topk_prob", False)
            config.moe_style = 2  # shared + expert

            moe_step = config_json["moe_layer_freq"]
            first_k_dense_replace = config_json["first_k_dense_replace"]
            config.moe_layer_index = [
                i
                for i in range(config.num_layers)
                if i >= first_k_dense_replace and i % moe_step == 0
            ]

            config.config_dtype = config_json.get("torch_dtype", None)

            # Engram config
            config.engram_layer_index = config_json.get("engram_layer_index", [])
            config.engram_vocab_size = config_json.get("engram_vocab_size", [])
            config.n_head_per_ngram = config_json.get("n_head_per_ngram", 0)
            config.n_embed_per_ngram = config_json.get("n_embed_per_ngram", 0)
            config.max_ngram_size = config_json.get("max_ngram_size", 0)
            config.pad_id = config_json.get("pad_id", 2)
            config.kernel_size = config_json.get("kernel_size", 4)
            config.seed = config_json.get("seed", 0)

            # manifold hyperconnections (MHC) config
            config.hc_mult = config_json.get("hc_mult", 1)
            config.max_sk_it = config_json.get("max_sk_it", 0)

    @staticmethod
    def get_weight_cls():
        return DeepSeekV4Weight


register_model("deepseek_v4", DeepSeekV4, ["DeepseekV4ForCausalLM"])
