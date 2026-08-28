"""GLM-5.3-Flash configuration and model integration."""

import json
import os
from typing import Any, Dict, List

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.linear_attn_weight import (
    LinearAttnAtomicWeight,
    LinearAttnConfig,
)
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import (
    AtomicWeight,
    CompositeWeight,
    WeightModule,
)
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.deepseek_v2 import DeepSeekV2Weight
from rtp_llm.ops import DataType, HybridAttentionType
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    merge_qkv_hf,
    transpose,
)

_LANGUAGE_MODEL_PREFIX = "model.language_model."
_UNQUANTIZED_WEIGHT_NAMES = {
    W.linear_attn_out_w,
    W.linear_attn_qkv_w,
    W.linear_attn_b_w,
    W.linear_attn_f_a_w,
    W.linear_attn_f_b_w,
    W.linear_attn_g_a_w,
    W.linear_attn_g_b_w,
    W.linear_attn_conv1d_w,
    W.linear_attn_norm_w,
    W.linear_attn_dt_b_kda,
    W.linear_attn_alog,
    W.mla_kv_b_w,
    W.mla_kc,
    W.mla_vc,
    W.mla_indexer_qb_w,
    W.mla_indexer_k_w,
    W.v4_indexer_compressor_wgate,
    W.v4_indexer_compressor_ape,
}


class Glm53FlashModelConfig(ModelConfig):
    """Model config with checkpoint-mandated KDA cache precision."""

    def init_linear_attention_cache_precision(self, kv_cache_config: Any) -> None:
        super().init_linear_attention_cache_precision(kv_cache_config)
        self.linear_attention_config.ssm_state_dtype = DataType.TYPE_FP32


def parse_glm53_flash_config(
    config_json: Dict[str, Any], ckpt_path: str = ""
) -> Glm53FlashModelConfig:
    """Translate the nested Hugging Face config into RTP model config."""
    text_config = config_json.get("text_config", config_json)
    layer_types = text_config["layer_types"]
    mlp_layer_types = text_config["mlp_layer_types"]
    num_layers = int(text_config["num_hidden_layers"])
    if len(layer_types) != num_layers or len(mlp_layer_types) != num_layers:
        raise ValueError(
            "GLM-5.3-Flash layer schedules must match num_hidden_layers: "
            f"attention={len(layer_types)}, mlp={len(mlp_layer_types)}, "
            f"layers={num_layers}"
        )

    attention_type_map = {
        "linear_attention": HybridAttentionType.LINEAR,
        "deepseek_sparse_attention": HybridAttentionType.NONE,
    }
    try:
        hybrid_types: List[HybridAttentionType] = [
            attention_type_map[layer_type] for layer_type in layer_types
        ]
    except KeyError as error:
        raise ValueError(
            f"Unsupported GLM-5.3-Flash attention type: {error.args[0]}"
        ) from error
    unsupported_mlp_types = set(mlp_layer_types) - {"dense", "sparse"}
    if unsupported_mlp_types:
        raise ValueError(
            "Unsupported GLM-5.3-Flash MLP types: "
            + ", ".join(sorted(unsupported_mlp_types))
        )

    config = Glm53FlashModelConfig()
    config.ckpt_path = ckpt_path
    config.tokenizer_path = ckpt_path
    config.model_type = "glm5_3_flash"
    config.num_layers = num_layers
    config.hidden_size = int(text_config["hidden_size"])
    config.vocab_size = int(text_config["vocab_size"])
    config.input_vocab_size = config.vocab_size
    config.max_seq_len = int(text_config["max_position_embeddings"])
    config.layernorm_eps = float(text_config.get("rms_norm_eps", 1e-5))
    config.tie_word_embeddings = bool(text_config.get("tie_word_embeddings", False))
    config.has_lm_head = True
    config.config_dtype = text_config.get("dtype", text_config.get("torch_dtype"))
    config.norm_type = "rmsnorm"
    config.activation_type = "SiGLU"
    config.has_pre_decoder_layernorm = False
    config.has_post_decoder_layernorm = True
    config.qk_norm = True
    config.enable_fp32_lm_head = False

    config.special_tokens.bos_token_id = int(
        config_json.get("bos_token_id", text_config.get("bos_token_id", -1))
    )
    eos_token_ids = config_json.get("eos_token_id", text_config.get("eos_token_id", 0))
    if not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]
    if not eos_token_ids:
        raise ValueError("GLM-5.3-Flash eos_token_id must not be empty")
    config.special_tokens.eos_token_id = int(eos_token_ids[0])
    config.special_tokens.pad_token_id = int(
        config_json.get("pad_token_id", text_config.get("pad_token_id", 0))
    )
    config.special_tokens.stop_words_id_list = [
        [int(token_id)] for token_id in eos_token_ids
    ]

    attn_config = config.attn_config
    attn_config.head_num = int(text_config["num_attention_heads"])
    attn_config.kv_head_num = int(text_config.get("num_key_value_heads", 1))
    attn_config.use_mla = True
    attn_config.q_lora_rank = int(text_config.get("q_lora_rank", 0))
    attn_config.kv_lora_rank = int(text_config["kv_lora_rank"])
    attn_config.nope_head_dim = int(text_config["qk_nope_head_dim"])
    attn_config.rope_head_dim = int(text_config.get("qk_rope_head_dim", 0))
    attn_config.v_head_dim = int(text_config["v_head_dim"])
    attn_config.size_per_head = attn_config.nope_head_dim + attn_config.rope_head_dim
    attn_config.rope_config.dim = attn_config.rope_head_dim
    attn_config.rope_config.offset = attn_config.nope_head_dim
    attn_config.rope_config.indexer_is_neox_style = not bool(
        text_config.get("indexer_rope_interleave", False)
    )
    attn_config.rope_config.style = 0
    attn_config.rope_config.base = int(text_config.get("rope_theta", 10000))
    attn_config.is_causal = True
    config.has_positional_encoding = attn_config.rope_head_dim > 0
    attn_config.is_sparse = True
    attn_config.indexer_head_dim = int(text_config["index_head_dim"])
    attn_config.indexer_head_num = int(text_config["index_n_heads"])
    raw_indexer_topk = int(text_config["index_topk"])
    indexer_compress_ratio = int(text_config.get("index_kpool", 0))
    if not bool(text_config.get("index_kpool_compress", False)):
        raise ValueError("GLM-5.3-Flash requires index_kpool_compress=true")
    if indexer_compress_ratio != 4:
        raise ValueError(
            "GLM-5.3-Flash requires index_kpool=4, got " f"{indexer_compress_ratio}"
        )
    if raw_indexer_topk <= 0 or raw_indexer_topk % indexer_compress_ratio != 0:
        raise ValueError(
            "GLM-5.3-Flash index_topk must be positive and divisible by "
            f"index_kpool: topk={raw_indexer_topk}, "
            f"index_kpool={indexer_compress_ratio}"
        )
    if not bool(text_config.get("index_kpool_always_select_tail", False)):
        raise ValueError("GLM-5.3-Flash requires index_kpool_always_select_tail=true")
    attn_config.indexer_topk = raw_indexer_topk // indexer_compress_ratio
    attn_config.indexer_compress_ratio = indexer_compress_ratio
    attn_config.indexer_compressor_overlap = 0
    attn_config.sparse_attention_topk = raw_indexer_topk + indexer_compress_ratio - 1
    config.indexer_types = list(text_config.get("indexer_types", []))
    config.index_share_for_mtp_iteration = bool(
        text_config.get("index_share_for_mtp_iteration", False)
    )

    config.hybrid_attention_config.enable_hybrid_attention = True
    config.hybrid_attention_config.enable_independent_kv_cache_pools = True
    config.hybrid_attention_config.hybrid_attention_types = hybrid_types
    attn_config.indexer_layer_ids = [
        layer_id
        for layer_id, layer_type in enumerate(hybrid_types)
        if layer_type != HybridAttentionType.LINEAR
    ]
    linear_config = config.linear_attention_config
    linear_hf_config = text_config["linear_attn_config"]
    linear_config.linear_conv_kernel_dim = int(
        linear_hf_config["short_conv_kernel_size"]
    )
    linear_config.linear_key_head_dim = int(linear_hf_config["head_dim"])
    linear_config.linear_value_head_dim = int(linear_hf_config["head_dim"])
    linear_config.linear_num_key_heads = int(linear_hf_config["num_heads"])
    linear_config.linear_num_value_heads = int(linear_hf_config["num_heads"])
    config.kda_gate_lower_bound = (
        float(linear_hf_config["gate_lower_bound"])
        if "gate_lower_bound" in linear_hf_config
        else None
    )

    scoring_func = text_config.get("scoring_func", "sigmoid")
    if scoring_func not in {"softmax", "sigmoid"}:
        raise ValueError(f"Unsupported GLM-5.3-Flash scoring_func: {scoring_func}")
    config.scoring_func = 0 if scoring_func == "softmax" else 1
    config.routed_scaling_factor = float(text_config["routed_scaling_factor"])
    config.moe_k = int(text_config["num_experts_per_tok"])
    config.expert_num = int(text_config["n_routed_experts"])
    config.moe_n_group = int(text_config.get("n_group", 1))
    config.moe_topk_group = int(text_config.get("topk_group", 1))
    config.has_moe_norm = bool(text_config.get("norm_topk_prob", False))
    config.moe_style = 2
    config.inter_size = int(text_config["intermediate_size"])
    config.moe_inter_size = int(text_config["moe_intermediate_size"])
    config.moe_layer_index = [
        i for i, layer_type in enumerate(mlp_layer_types) if layer_type == "sparse"
    ]
    config.hc_mult = int(text_config.get("hc_mult", 1))
    config.hc_sinkhorn_iters = int(text_config.get("hc_sinkhorn_iters", 0))
    config.hc_eps = float(text_config.get("hc_eps", 1e-6))
    config.swiglu_limit = float(text_config.get("swiglu_limit", 0.0))
    vision_config = config_json.get("vision_config")
    if vision_config:
        vision_config = dict(vision_config)
        # Match the reference implementation, which uses 1e-6 for the
        # vision block and post-attention norms despite the checkpoint value.
        vision_config["rms_norm_eps"] = 1e-6
        config.mm_model_config.is_multimodal = True
        config.mm_model_config.mm_sep_tokens = [
            [
                int(config_json.get("image_start_token_id", 154830)),
                int(config_json.get("image_end_token_id", 154831)),
            ],
        ]
        config.mm_model_config.mm_position_ids_style = 0
        config.mm_related_params.special_tokens["default_mm_token"] = (
            "<|begin_of_image|><|image|><|end_of_image|>"
        )
        processor_path = os.path.join(ckpt_path, "processor_config.json")
        if not os.path.exists(processor_path):
            raise FileNotFoundError(
                f"processor_config.json not found for GLM-5.3-Flash VL: {ckpt_path}"
            )
        with open(processor_path, encoding="utf-8") as reader:
            processor_config = json.load(reader)
        config.mm_related_params.config.update(
            {
                "ckpt_path": ckpt_path,
                "vision_config": vision_config,
                "processor_config": processor_config,
                "swiglu_limit": config.swiglu_limit,
            }
        )
    return config


def _merge_conv1d(tensors: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat(tensors, dim=0)


class Glm53FlashWeight(DeepSeekV2Weight):
    """GLM-5.3 manifest reusing the DeepSeek MLA, MoE and FP8 loaders."""

    def _process_meta(self, meta_dict, weight_keys):
        self.q_use_lora = any(
            key.startswith(_LANGUAGE_MODEL_PREFIX + "layers.")
            and key.endswith(".self_attn.q_a_proj.weight")
            for key in weight_keys
        )
        self.has_e_score_correction_bias = any(
            key.startswith(_LANGUAGE_MODEL_PREFIX + "layers.")
            and key.endswith(".mlp.gate.e_score_correction_bias")
            for key in weight_keys
        )

    def _create_rope_w(self):
        # GLM-5.3 is NoPE MLA. Indexer keeps this runtime key for API
        # compatibility but does not read cache values when rope_head_dim is 0.
        return AtomicWeight(
            W.rope_cos_sin_cache,
            [],
            lambda _: torch.empty((1, 0), dtype=torch.float32, device="cuda"),
            data_type=torch.float32,
        )

    def _get_hf_layer_weight_info(self, layer_id: int):
        layer_type = self.model_config.hybrid_attention_config.hybrid_attention_types[
            layer_id
        ]
        if layer_type == HybridAttentionType.LINEAR:
            weights: List[WeightModule] = [
                AtomicWeight(
                    W.pre_ln_gamma,
                    [
                        CkptWeightInfo(
                            "model.layers.{i}.input_layernorm.weight", identity
                        )
                    ],
                ),
                AtomicWeight(
                    W.post_ln_gamma,
                    [
                        CkptWeightInfo(
                            "model.layers.{i}.post_attention_layernorm.weight",
                            identity,
                        )
                    ],
                ),
            ]
            weights.extend(self._get_kda_weight_info())
            weights.extend(self._get_hf_ffn_layer_weight_info(layer_id))
        else:
            weights = super()._get_hf_layer_weight_info(layer_id)
            weights.extend(
                [
                    AtomicWeight(
                        W.v4_indexer_compressor_wgate,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.self_attn.indexer."
                                "index_kpool_compress_gate",
                                identity,
                            )
                        ],
                        identity,
                    ),
                    AtomicWeight(
                        W.v4_indexer_compressor_ape,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.self_attn.indexer."
                                "index_kpool_compress_ape",
                                identity,
                            )
                        ],
                        identity,
                    ),
                ]
            )
        weights.extend(self._get_hc_weight_info())
        return weights

    def _get_kda_weight_info(self) -> List[WeightModule]:
        config = LinearAttnConfig(self.model_config.linear_attention_config)
        prefix = "model.layers.{i}.self_attn."
        return [
            LinearAttnAtomicWeight(
                W.linear_attn_qkv_w,
                [
                    CkptWeightInfo(prefix + "q_proj.weight"),
                    CkptWeightInfo(prefix + "k_proj.weight"),
                    CkptWeightInfo(prefix + "v_proj.weight"),
                ],
                merge_qkv_hf,
                config,
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_b_w,
                [CkptWeightInfo(prefix + "b_proj.weight")],
                transpose,
                config,
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_f_a_w,
                [CkptWeightInfo(prefix + "f_a_proj.weight")],
                transpose,
                config,
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_f_b_w,
                [CkptWeightInfo(prefix + "f_b_proj.weight")],
                transpose,
                config,
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_g_a_w,
                [CkptWeightInfo(prefix + "g_a_proj.weight")],
                transpose,
                config,
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_g_b_w,
                [CkptWeightInfo(prefix + "g_b_proj.weight")],
                transpose,
                config,
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_conv1d_w,
                [
                    CkptWeightInfo(prefix + "q_conv1d.weight"),
                    CkptWeightInfo(prefix + "k_conv1d.weight"),
                    CkptWeightInfo(prefix + "v_conv1d.weight"),
                ],
                _merge_conv1d,
                config,
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_norm_w,
                [CkptWeightInfo(prefix + "o_norm.weight")],
                identity,
                config,
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_dt_b_kda,
                [CkptWeightInfo(prefix + "dt_bias")],
                identity,
                config,
                data_type=torch.float32,
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_alog,
                [CkptWeightInfo(prefix + "A_log")],
                lambda tensors: tensors[0].squeeze(),
                config,
                data_type=torch.float32,
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_out_w,
                [CkptWeightInfo(prefix + "o_proj.weight")],
                transpose,
                config,
            ),
        ]

    @staticmethod
    def _get_hc_weight_info() -> List[WeightModule]:
        weights: List[WeightModule] = []
        for residual in ("attn", "ffn"):
            for suffix in ("base", "fn", "scale"):
                weights.append(
                    AtomicWeight(
                        getattr(W, f"v4_hc_{residual}_{suffix}"),
                        [CkptWeightInfo(f"model.layers.{{i}}.hc_{residual}_{suffix}")],
                        identity,
                        data_type=torch.float32,
                    )
                )
        return weights

    @staticmethod
    def _prefix_checkpoint_names(weight: WeightModule) -> None:
        if weight.name in _UNQUANTIZED_WEIGHT_NAMES:
            weight.quantization_disabled = True
        for checkpoint_weight in getattr(weight, "weights", []):
            if checkpoint_weight.name.startswith(
                "model."
            ) and not checkpoint_weight.name.startswith(
                _LANGUAGE_MODEL_PREFIX,
            ):
                checkpoint_weight.name = (
                    _LANGUAGE_MODEL_PREFIX + checkpoint_weight.name[len("model.") :]
                )
        if isinstance(weight, CompositeWeight):
            for sub_weight in weight.sub_weights.values():
                Glm53FlashWeight._prefix_checkpoint_names(sub_weight)

    def _get_weight_info(self) -> ModelWeightInfo:
        weight_info = super()._get_weight_info()
        for weight in weight_info.weights:
            self._prefix_checkpoint_names(weight)
        for layer_weights in weight_info.layer_weights:
            for weight in layer_weights:
                self._prefix_checkpoint_names(weight)
        return weight_info


class Glm53Flash(BaseModel):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {ckpt_path}")
        with open(config_path, encoding="utf-8") as reader:
            config_json = json.load(reader)
        return parse_glm53_flash_config(config_json, ckpt_path)

    def support_cuda_graph(self) -> bool:
        return True

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.kimi_linear import KimiLinearModel
        from rtp_llm.models_py.utils.arch import is_cuda

        if not is_cuda():
            raise RuntimeError("GLM-5.3-Flash is supported only on CUDA")
        self.py_model = KimiLinearModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model

    @staticmethod
    def get_weight_cls():
        return Glm53FlashWeight


register_model(
    "glm5_3_flash",
    Glm53Flash,
    # This is the architecture string stored in the released checkpoint, not
    # the RTP-facing model/product name.
    ["Glm5NextForConditionalGeneration"],
)


__all__ = [
    "Glm53Flash",
    "Glm53FlashModelConfig",
    "Glm53FlashWeight",
    "parse_glm53_flash_config",
]
