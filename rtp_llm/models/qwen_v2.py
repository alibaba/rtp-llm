import functools
import json
import logging
import os
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, FfnConfig, FfnWeight
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.qwen import QWen
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

from rtp_llm.utils.swizzle_utils import do_swizzle

def scale_reshape(ts: List[torch.Tensor]):
    return ts[0].reshape(-1)


class QWenV2Weight(ModelDeployWeightInfo):
    def __init__(self, *args: Any, **kwargs: Any):
        self.prefix: str = kwargs.pop("prefix", "")
        self.model_prefix: str = "model."
        self.bias = True
        self.strip_model_prefix = False
        super().__init__(*args, **kwargs)

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
        self.transformer_prefix = self.prefix + self.model_prefix
        logging.info(f"weight_style: {self.weight_style}")

    def _get_weight_info(self):
        return self._get_hf_weight_info()

    def _get_hf_ffn_layer_weight_info(self, layer_id: int) -> List[WeightModule]:
        ffn_config = FfnConfig(
            is_gated_activation=self._is_gated_activation,
            inter_padding_size=self._inter_padding_size,
            is_moe=False,
        )

        inter_padding_size = (
            self._layer_inter_padding_size[layer_id]
            if self._layer_inter_padding_size
            else self._inter_padding_size
        )
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
                        functools.partial(
                            transpose_pad, inter_padding_size=inter_padding_size, dim=0
                        ),
                        config=ffn_config,
                        lora_a_process_func=transpose,
                        lora_b_process_func=functools.partial(
                            transpose_pad, inter_padding_size=inter_padding_size, dim=0
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
                        functools.partial(
                            transpose_pad, inter_padding_size=inter_padding_size, dim=0
                        ),
                        config=ffn_config,
                        lora_a_process_func=transpose,
                        lora_b_process_func=functools.partial(
                            transpose_pad, inter_padding_size=inter_padding_size, dim=0
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
                        functools.partial(
                            transpose_pad, inter_padding_size=inter_padding_size, dim=1
                        ),
                        config=ffn_config,
                        lora_a_process_func=functools.partial(
                            transpose_pad, inter_padding_size=inter_padding_size, dim=1
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
                    [CkptWeightInfo(self.prefix + "lm_head.weight", identity)],
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
    def _create_config(cls, ckpt_path: str):
        config = GptInitModelParameters(
            head_num=0,
            head_num_kv=0,
            size_per_head=0,
            layer_num=0,
            inter_size=0,  # 13696
            vocab_size=152064,
            max_seq_len=8192,
        )
        config.rotary_embedding_dim = 128
        config.rotary_embedding_style = 1
        config.activation_type = "SiGLU"
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        config.norm_type = "rmsnorm"
        config.special_tokens.bos_token_id = -1
        config.special_tokens.eos_token_id = 151643
        # <|im_start|> and <|im_end|>
        config.special_tokens.stop_words_id_list = [[151645], [151644]]

        cls._from_hf(config, ckpt_path)
        assert (
            config.head_num > 0
            and config.head_num_kv > 0
            and config.size_per_head > 0
            and config.layer_num > 0
            and config.inter_size > 0
        ), f"error config config.head_num={config.head_num} config.head_num_kv={config.head_num_kv} config.size_per_head={config.size_per_head} config.layer_num={config.layer_num} config.inter_size={config.inter_size}"
        return config

    @classmethod
    def _from_hf(cls, config: GptInitModelParameters, ckpt_path: str):
        config_path = os.path.join(ckpt_path, "config.json")

        if not os.path.exists(config_path):
            return
        with open(config_path) as reader:
            content = reader.read()
            config_json = json.loads(content)
        QWenV2._from_config_json(config, config_json)
        return config

    @staticmethod
    def _from_config_json(config: GptInitModelParameters, config_json: Dict[str, Any]):
        # config.activation_type = config_json["hidden_act"]
        config.inter_size = config_json["intermediate_size"]
        config.head_num = config_json["num_attention_heads"]
        config.head_num_kv = config_json.get("num_key_value_heads", config.head_num)
        config.size_per_head = (
            int(config_json.get("head_dim"))
            if "head_dim" in config_json
            else config_json["hidden_size"] // config.head_num
        )
        if config_json.get("hidden_size") is not None:
            config.hidden_size = config_json["hidden_size"]
        config.layer_num = config_json["num_hidden_layers"]
        config.rotary_embedding_base = config_json.get(
            "rope_theta", config.rotary_embedding_base
        )
        config.vocab_size = config_json["vocab_size"]
        config.rotary_embedding_dim = config.size_per_head
        config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
        config.config_dtype = config_json.get("torch_dtype", None)

    @staticmethod
    def get_weight_cls():
        return QWenV2Weight

    def postprocess_weights(self):
        if self.config.hw_kernel_config.use_swizzleA and self.weight.weights[0]["self_attention_weights.query_weight.kernel"].dtype != torch.float8_e4m3fnuz:
            do_swizzle(self.weight.weights)

class QWenV2Embedding(QWenV2):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = QWenV2._create_config(ckpt_path)
        config.is_causal = False
        return config


class QwenV2MTPWeight(QWenV2Weight):
    def __init__(self, config: GptInitModelParameters, tp_size: int, tp_rank: int):
        super().__init__(config, tp_size, tp_rank)

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
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        config.moe_layer_index = [i for i in range(config.layer_num)]
        config.is_mtp = True
        return config

    @staticmethod
    def get_weight_cls():
        return QwenV2MTPWeight


register_model("qwen_2", QWenV2, ["Qwen2ForCausalLM"])
register_model("qwen_agent", QWenV2)
register_model("qwen_2_embedding", QWenV2Embedding)
register_model("qwen_tool", QWenV2)
register_model("qwen_2-mtp", QwenV2MTP)
