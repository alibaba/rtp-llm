import functools
import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

from rtp_llm.config.model_config import ModelConfig, VitParameters
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, FfnConfig, FfnWeight
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.qwen import QWen
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.qwen2_mtp import Qwen2MtpModel
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    WeightStyle,
    identity,
    merge_qkv_b,
    merge_qkv_hf,
    merge_qkv_lora_A,
    merge_qkv_lora_B,
    sp_0,
    sp_head_lora,
    sp_id,
    sp_neg1,
    transpose,
    transpose_pad,
    zeros,
)


def scale_reshape(ts: List[torch.Tensor]):
    return ts[0].reshape(-1)


class QWenV2Weight(ModelDeployWeightInfo):
    def __init__(self, prefix=None, **kwargs: Any):
        self.prefix = prefix or ""
        self.model_prefix: str = "model."
        self.bias = True
        self.strip_model_prefix = False
        self.lm_head_weight_name: Optional[str] = None
        super().__init__(**kwargs)

    @property
    def support_lora(self):
        return True

    def _process_meta(self, meta_dicts: Any, weight_keys: List[str]):
        # compat for qwen_v2_video
        if self._contains(weight_keys, "language_model."):
            self.prefix = "language_model."
        if self.prefix + "transformer.layers.0.attention.dense.weight" in meta_dicts[0]:
            self.weight_style = WeightStyle.TRT_ENGINE
        if self._exist(weight_keys, "layers.0.input_layernorm.weight"):
            self.model_prefix = ""
        self.transformer_prefix = self.model_prefix + self.prefix
        lm_head_suffix = "lm_head.weight"
        candidate_names = []
        for name in [
            self.transformer_prefix + lm_head_suffix,
            self.model_prefix + lm_head_suffix,
            self.prefix + lm_head_suffix,
            lm_head_suffix,
        ]:
            if not name:
                continue
            if name not in candidate_names:
                candidate_names.append(name)
        self.lm_head_weight_name = candidate_names[0]
        for name in candidate_names:
            if self._exist(weight_keys, name):
                self.lm_head_weight_name = name
                if name != candidate_names[0]:
                    logging.info(
                        "detected lm_head weight name override: %s",
                        self.lm_head_weight_name,
                    )
                break
        else:
            logging.warning(
                "lm_head weight %s not found in checkpoint meta, fallback to default",
                candidate_names[0],
            )
        logging.info(f"weight_style: {self.weight_style}")

    def _get_weight_info(self):
        return self._get_hf_weight_info()

    def _get_hf_ffn_layer_weight_info(self, layer_id: int) -> List[WeightModule]:
        ffn_config = FfnConfig(
            is_gated_activation=self._is_gated_activation,
            align_size=self._align_size,
            is_moe=False,
        )

        align_size = self._align_size
        return [
            FfnWeight(
                sub_weights=[
                    FfnAtomicWeight(
                        W.ffn_w1,
                        [
                            CkptWeightInfo(
                                self.transformer_prefix
                                + "layers.{i}.mlp.gate_proj.weight",
                                identity,
                            )
                        ],
                        functools.partial(transpose_pad, align_size=align_size, dim=0),
                        config=ffn_config,
                        lora_a_process_func=transpose,
                        lora_b_process_func=functools.partial(
                            transpose_pad, align_size=align_size, dim=0
                        ),
                        lora_a_split_func=sp_id,
                        lora_b_split_func=sp_neg1,
                    ),
                    FfnAtomicWeight(
                        W.ffn_w3,
                        [
                            CkptWeightInfo(
                                self.transformer_prefix
                                + "layers.{i}.mlp.up_proj.weight",
                                identity,
                            )
                        ],
                        functools.partial(transpose_pad, align_size=align_size, dim=0),
                        config=ffn_config,
                        lora_a_process_func=transpose,
                        lora_b_process_func=functools.partial(
                            transpose_pad, align_size=align_size, dim=0
                        ),
                        lora_a_split_func=sp_id,
                        lora_b_split_func=sp_neg1,
                    ),
                    FfnAtomicWeight(
                        W.ffn_w2,
                        [
                            CkptWeightInfo(
                                self.transformer_prefix
                                + "layers.{i}.mlp.down_proj.weight",
                                identity,
                            )
                        ],
                        functools.partial(transpose_pad, align_size=align_size, dim=1),
                        config=ffn_config,
                        lora_a_process_func=functools.partial(
                            transpose_pad, align_size=align_size, dim=1
                        ),
                        lora_b_process_func=transpose,
                        lora_a_split_func=sp_0,
                        lora_b_split_func=sp_id,
                    ),
                ],
                config=ffn_config,
            )
        ]

    def _get_hf_layer_weight_info(self, layer_id: int):
        attn_config = AttnConfig(
            hidden_size=self._hidden_size,
            size_per_head=self._size_per_head,
            head_num=self._head_num,
            head_num_kv=self._head_num_kv,
        )

        layer_weights = [
            AtomicWeight(
                W.pre_ln_gamma,
                [
                    CkptWeightInfo(
                        self.transformer_prefix + "layers.{i}.input_layernorm.weight",
                        identity,
                    )
                ],
                identity,
            ),
            AttnAtomicWeight(
                W.attn_qkv_w,
                [
                    CkptWeightInfo(
                        self.transformer_prefix + "layers.{i}.self_attn.q_proj.weight",
                        identity,
                    ),
                    CkptWeightInfo(
                        self.transformer_prefix + "layers.{i}.self_attn.k_proj.weight",
                        identity,
                    ),
                    CkptWeightInfo(
                        self.transformer_prefix + "layers.{i}.self_attn.v_proj.weight",
                        identity,
                    ),
                ],
                functools.partial(merge_qkv_hf),
                config=attn_config,
                lora_a_process_func=functools.partial(
                    merge_qkv_lora_A,
                    allow_empty=False,
                    hidden_size=self._hidden_size,
                    head_num=self._head_num,
                    head_num_kv=self._head_num_kv,
                    size_per_head=self._size_per_head,
                ),
                lora_b_process_func=functools.partial(
                    merge_qkv_lora_B,
                    allow_empty=False,
                    hidden_size=self._hidden_size,
                    head_num=self._head_num,
                    head_num_kv=self._head_num_kv,
                    size_per_head=self._size_per_head,
                ),
                lora_a_split_func=sp_id,
                lora_b_split_func=sp_head_lora,
            ),
            AttnAtomicWeight(
                W.attn_o_w,
                [
                    CkptWeightInfo(
                        self.transformer_prefix + "layers.{i}.self_attn.o_proj.weight",
                        identity,
                    )
                ],
                transpose,
                config=attn_config,
                lora_a_process_func=transpose,
                lora_b_process_func=transpose,
                lora_a_split_func=sp_0,
                lora_b_split_func=sp_id,
            ),
            AtomicWeight(
                W.post_ln_gamma,
                [
                    CkptWeightInfo(
                        self.transformer_prefix
                        + "layers.{i}.post_attention_layernorm.weight",
                        identity,
                    )
                ],
                identity,
                config=attn_config,
            ),
        ]

        if self.bias:
            layer_weights.append(
                AttnAtomicWeight(
                    W.attn_qkv_b,
                    [
                        CkptWeightInfo(
                            self.transformer_prefix
                            + "layers.{i}.self_attn.q_proj.bias",
                            identity,
                        ),
                        CkptWeightInfo(
                            self.transformer_prefix
                            + "layers.{i}.self_attn.k_proj.bias",
                            identity,
                        ),
                        CkptWeightInfo(
                            self.transformer_prefix
                            + "layers.{i}.self_attn.v_proj.bias",
                            identity,
                        ),
                    ],
                    functools.partial(merge_qkv_b),
                    config=attn_config,
                )
            )

        if self._use_qk_norm:
            layer_weights.extend(
                [
                    AttnAtomicWeight(
                        W.q_ln_gamma,
                        [
                            CkptWeightInfo(
                                self.transformer_prefix
                                + "layers.{i}.self_attn.q_norm.weight"
                            )
                        ],
                        config=attn_config,
                    ),
                    AttnAtomicWeight(
                        W.k_ln_gamma,
                        [
                            CkptWeightInfo(
                                self.transformer_prefix
                                + "layers.{i}.self_attn.k_norm.weight"
                            )
                        ],
                        config=attn_config,
                    ),
                ]
            )

        layer_weights.extend(self._get_hf_ffn_layer_weight_info(layer_id))
        return layer_weights

    def _get_hf_weight_info(self):
        if self.weight_style == WeightStyle.TRT_ENGINE:
            weights = [
                AtomicWeight(
                    W.embedding,
                    [CkptWeightInfo("transformer.vocab_embedding.weight", identity)],
                    identity,
                ),
                AtomicWeight(
                    W.lm_head, [CkptWeightInfo("lm_head.weight", identity)], identity
                ),
                AtomicWeight(
                    W.final_ln_gamma,
                    [CkptWeightInfo("transformer.ln_f.weight", identity)],
                    identity,
                ),
                AtomicWeight(
                    W.final_ln_beta,
                    [],
                    functools.partial(zeros, shape=[self._hidden_size]),
                ),
            ]
        else:
            lm_head_name = self.lm_head_weight_name or (
                self.transformer_prefix + "lm_head.weight"
            )
            weights = [
                AtomicWeight(
                    W.embedding,
                    [
                        CkptWeightInfo(
                            self.transformer_prefix + "embed_tokens.weight", identity
                        )
                    ],
                    identity,
                ),
                AtomicWeight(
                    W.lm_head,
                    [CkptWeightInfo("lm_head.weight", identity)],
                    identity,
                ),
                AtomicWeight(
                    W.final_ln_gamma,
                    [CkptWeightInfo(self.transformer_prefix + "norm.weight", identity)],
                    identity,
                ),
                AtomicWeight(
                    W.final_ln_beta,
                    [],
                    functools.partial(zeros, shape=[self._hidden_size]),
                ),
            ]

        layer_weights: List[List[WeightModule]] = []
        for layer in range(self._num_layers):
            w = self._get_hf_layer_weight_info(layer)
            layer_weights.append(w)

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)


class QWenV2(QWen):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.vocab_size = 152064
        config.max_seq_len = 8192
        config.attn_config.rope_config.dim = 128
        config.attn_config.rope_config.style = 1
        config.has_pre_decoder_layernorm = False
        config.special_tokens.bos_token_id = -1
        config.special_tokens.eos_token_id = 151643
        # <|im_start|> and <|im_end|>
        config.special_tokens.stop_words_id_list = [[151645], [151644]]

        QWenV2._from_hf(config, ckpt_path)
        assert (
            config.attn_config.head_num > 0
            and config.attn_config.kv_head_num > 0
            and config.attn_config.size_per_head > 0
            and config.num_layers > 0
            and config.inter_size > 0
        ), f"error config config.attn_config.head_num={config.attn_config.head_num} config.attn_config.kv_head_num={config.attn_config.kv_head_num} config.attn_config.size_per_head={config.attn_config.size_per_head} config.num_layers={config.num_layers} config.inter_size={config.inter_size}"
        return config

    @classmethod
    def _from_hf(cls, config: "ModelConfig", ckpt_path: str):
        config_path = os.path.join(ckpt_path, "config.json")

        if not os.path.exists(config_path):
            return
        with open(config_path) as reader:
            content = reader.read()
            config_json = json.loads(content)
        QWenV2._from_config_json(config, config_json)
        return config

    @staticmethod
    def _from_config_json(config: "ModelConfig", config_json: Dict[str, Any]):
        # config.activation_type = config_json["hidden_act"]
        config.inter_size = config_json["intermediate_size"]
        config.attn_config.head_num = config_json["num_attention_heads"]
        config.attn_config.kv_head_num = config_json.get(
            "num_key_value_heads", config.attn_config.head_num
        )
        config.attn_config.size_per_head = (
            int(config_json.get("head_dim"))
            if "head_dim" in config_json
            else config_json["hidden_size"] // config.attn_config.head_num
        )
        if config_json.get("hidden_size") is not None:
            config.hidden_size = config_json["hidden_size"]
        config.num_layers = config_json["num_hidden_layers"]
        config.attn_config.rope_config.base = int(
            config_json.get("rope_theta", config.attn_config.rope_config.base)
        )
        config.vocab_size = config_json["vocab_size"]
        config.attn_config.rope_config.dim = config.attn_config.size_per_head
        config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
        config.config_dtype = config_json.get("torch_dtype", None)

    @staticmethod
    def get_weight_cls():
        return QWenV2Weight


class QWenV2Embedding(QWenV2):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = QWenV2._create_config(ckpt_path)
        config.attn_config.is_causal = False
        return config


class QwenV2MTPWeight(QWenV2Weight):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_weight_info(self):
        weights = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo(self.prefix + "model.embed_tokens.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo(self.prefix + "lm_head.weight", identity)],
                identity,
            ),
        ]
        layer_weights: List[List[AtomicWeight]] = []
        for layer in range(self._num_layers):
            w = self._get_hf_layer_weight_info(layer)
            layer_weights.append(w)
        for layer_id in range(self._num_layers):
            layer_weights[layer_id].extend(
                [
                    AtomicWeight(
                        W.multi_tokens_predict_enorm,
                        [CkptWeightInfo("model.layers.{i}.e_norm.weight", identity)],
                        identity,
                    ),
                    AtomicWeight(
                        W.multi_tokens_predict_hnorm,
                        [CkptWeightInfo("model.layers.{i}.h_norm.weight", identity)],
                        identity,
                    ),
                    AtomicWeight(
                        W.multi_tokens_predict_eh_proj,
                        [CkptWeightInfo("model.layers.{i}.eh_proj.weight", identity)],
                        identity,
                    ),
                    AtomicWeight(
                        W.multi_tokens_predict_final_ln_gamma,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.final_head.norm.weight", identity
                            )
                        ],
                        identity,
                    ),
                    AtomicWeight(
                        W.multi_tokens_predict_final_ln_beta,
                        [],
                        functools.partial(zeros, shape=[self._hidden_size]),
                    ),
                ]
            )
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)

    def _get_weights(self):
        weights = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo("model.embeddings.weight", concat_1)],
                identity,
            ),
            AtomicWeight(
                W.lm_head, [CkptWeightInfo("lm_head.weight", identity)], identity
            ),
        ]
        return weights


class QwenV2MTP(QWenV2):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = super()._create_config(ckpt_path)
        config.moe_layer_index = [i for i in range(config.num_layers)]
        config.is_mtp = True
        return config

    def _create_python_model(self) -> Optional[GptModelBase]:
        model_config = self.model_config
        parallelism_config = self.parallelism_config
        ffn_disaggregate_config = parallelism_config.ffn_disaggregate_config
        fmha_config = self.fmha_config
        py_hw_kernel_config = self.hw_kernel_config
        quant_config = self.model_config.quant_config
        self.py_model = Qwen2MtpModel(
            model_config,
            parallelism_config,
            self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            quant_config=quant_config,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )

    @staticmethod
    def get_weight_cls():
        return QwenV2MTPWeight


register_model("qwen_2", QWenV2, ["Qwen2ForCausalLM"])
register_model("qwen_agent", QWenV2)
register_model("qwen_2_embedding", QWenV2Embedding)
register_model("qwen_tool", QWenV2)
register_model("qwen_2-mtp", QwenV2MTP)
