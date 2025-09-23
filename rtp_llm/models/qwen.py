import functools
import json
import os
from typing import List, Optional

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, FfnConfig, FfnWeight
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models_py.model_desc.disaggregate_qwen3 import Qwen3DisaggregateModel
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.qwen3 import Qwen3Model
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    sp_0,
    sp_head_lora,
    sp_id,
    sp_neg1,
    transpose,
    transpose_pad,
    zeros,
)


def hidden_to_inter(hidden_size):
    ffn_m = 256
    return int((int(4 * 2 / 3 * hidden_size) * 2 + ffn_m - 1) // ffn_m * ffn_m / 2)


def qkv_transpose(ts, hidden_size):
    return ts[0].reshape(hidden_size, -1)


class QWenWeight(ModelDeployWeightInfo):
    @property
    def support_lora(self):
        return True

    def _get_layer_id_info(self, ckpt_meta):
        try:
            layers = [
                int(name.split(".")[1])
                for name in ckpt_meta["model"]["language_model"]["encoder"].keys()
                if "layers.0.self_attention.query_key_value.weight" in name
            ]
            return layers[0], layers[-1]
        except Exception as _:
            # 'transformer.h.{i}.attn.c_attn.weight'
            layers = [
                int(name.split(".")[2])
                for name in ckpt_meta.keys()
                if ".attn.c_attn.weight" in name
            ]
            return layers[0], layers[-1]

    def _get_weight_info(self):
        return self._get_hf_weight_info()

    def _get_hf_layer_weight_info(self, layer_id):
        inter_padding_size = (
            self._layer_inter_padding_size[layer_id]
            if self._layer_inter_padding_size
            else self._inter_padding_size
        )
        attn_config = AttnConfig(
            hidden_size=self._hidden_size,
            size_per_head=self._size_per_head,
            head_num=self._head_num,
            head_num_kv=self._head_num_kv,
        )
        ffn_config = FfnConfig(
            is_gated_activation=self._is_gated_activation,
            inter_padding_size=inter_padding_size,
            is_moe=False,
        )
        layer_weights = [
            AtomicWeight(
                W.pre_ln_gamma,
                [CkptWeightInfo("transformer.h.{i}.ln_1.weight", identity)],
                identity,
            ),
            AttnAtomicWeight(
                W.attn_qkv_w,
                [CkptWeightInfo("transformer.h.{i}.attn.c_attn.weight", identity)],
                transpose,
                config=attn_config,
                lora_a_process_func=transpose,
                lora_b_process_func=transpose,
                lora_a_split_func=sp_id,
                lora_b_split_func=sp_head_lora,
            ),
            AttnAtomicWeight(
                W.attn_qkv_b,
                [CkptWeightInfo("transformer.h.{i}.attn.c_attn.bias", identity)],
                identity,
                config=attn_config,
            ),
            AttnAtomicWeight(
                W.attn_o_w,
                [CkptWeightInfo("transformer.h.{i}.attn.c_proj.weight", identity)],
                transpose,
                config=attn_config,
                lora_a_process_func=transpose,
                lora_b_process_func=transpose,
                lora_a_split_func=sp_0,
                lora_b_split_func=sp_id,
            ),
            FfnWeight(
                sub_weights=[
                    FfnAtomicWeight(
                        W.ffn_w1,
                        [CkptWeightInfo("transformer.h.{i}.mlp.w2.weight", identity)],
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
                        [CkptWeightInfo("transformer.h.{i}.mlp.w1.weight", identity)],
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
                                "transformer.h.{i}.mlp.c_proj.weight", identity
                            )
                        ],
                        functools.partial(
                            transpose_pad, inter_padding_size=inter_padding_size, dim=1
                        ),
                        config=ffn_config,
                        lora_a_process_func=functools.partial(
                            transpose_pad, inter_padding_size=inter_padding_size, dim=0
                        ),
                        lora_b_process_func=transpose,
                        lora_a_split_func=sp_0,
                        lora_b_split_func=sp_id,
                    ),
                ],
                config=ffn_config,
            ),
            AtomicWeight(
                W.post_ln_gamma,
                [CkptWeightInfo("transformer.h.{i}.ln_2.weight", identity)],
                identity,
            ),
        ]
        return layer_weights

    def _get_hf_weight_info(self):
        weights = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo("transformer.wte.weight", identity)],
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
                W.final_ln_beta, [], functools.partial(zeros, shape=[self._hidden_size])
            ),
        ]

        layer_weights: List[List[WeightModule]] = []
        for layer in range(self._num_layers):
            w = self._get_hf_layer_weight_info(layer)
            layer_weights.append(w)

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)


class QWenBase(BaseModel):
    @staticmethod
    def get_weight_cls():
        return QWenWeight

    def _create_python_model(self) -> Optional[GptModelBase]:
        if self.config.gpt_init_params.ffn_disaggregate_config.enable_ffn_disaggregate:
            self.py_model = Qwen3DisaggregateModel(self.config, self.weight)
        else:
            self.py_model = Qwen3Model(self.config, self.weight)

    def support_cuda_graph(self) -> bool:
        return True

    @staticmethod
    def _common_config(config, ckpt_path: str) -> GptInitModelParameters:
        config.rotary_embedding_dim = 128
        config.rotary_embedding_style = 1
        config.activation_type = "SiGLU"
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        config.norm_type = "rmsnorm"
        config.layernorm_eps = 1e-5
        config.special_tokens.bos_token_id = -1
        config.special_tokens.eos_token_id = 151643
        # <|im_start|> and <|im_end|>
        config.special_tokens.stop_words_id_list = [[151645], [151644]]
        QWen._from_hf(config, ckpt_path)
        return config

    @staticmethod
    def _from_hf(config: GptInitModelParameters, ckpt_path: str):
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return
        with open(config_path) as reader:
            content = reader.read()
            config_json = json.loads(content)

        config.head_num = config_json.get(
            "n_head", config_json.get("num_attention_heads", config.head_num)
        )  # 如果2者不一致就是 attention sparse场景,headnum不能用attention的heads
        config.head_num_kv = config.head_num
        config.size_per_head = config_json.get("kv_channels", config.size_per_head)
        config.hidden_size = config_json.get("hidden_size", config.hidden_size)
        config.inter_size = int(
            config_json.get(
                "intermediate_size",
                config_json.get(
                    "ffn_hidden_size",
                    hidden_to_inter(config.head_num * config.size_per_head) * 2,
                ),
            )
            / 2
        )
        config.layernorm_eps = config_json.get(
            "layer_norm_epsilon", config.layernorm_eps
        )
        config.layer_num = config_json.get(
            "num_hidden_layers", config_json.get("n_layer", config.layer_num)
        )
        config.vocab_size = config_json.get(
            "vocab_size", config_json.get("padded_vocab_size", config.vocab_size)
        )
        config.rotary_embedding_base = config_json.get("rotary_emb_base", 10000)
        config.rotary_embedding_dim = config.size_per_head
        config.special_tokens.eos_token_id = config_json.get(
            "eos_token_id", config.special_tokens.eos_token_id
        )
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)

        if config_json.get("use_dynamic_ntk"):
            config.rotary_embedding_style = 4
        config.org_embedding_max_pos = config_json.get("seq_length", 8192)
        config.use_logn_attn = config_json.get("use_logn_attn")


class QWen(QWenBase):
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
        QWenBase._common_config(config, ckpt_path)
        assert (
            config.head_num > 0
            and config.head_num_kv > 0
            and config.size_per_head > 0
            and config.layer_num > 0
            and config.inter_size > 0
        ), "error config"
        return config


class QWen_7B(QWenBase):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = GptInitModelParameters(
            head_num=32,
            head_num_kv=32,
            size_per_head=128,
            layer_num=32,
            inter_size=hidden_to_inter(4096),  # 11008
            vocab_size=151936,
            max_seq_len=8192,
        )
        QWenBase._common_config(config, ckpt_path)
        return config


class QWen_13B(QWenBase):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = GptInitModelParameters(
            head_num=40,
            head_num_kv=40,
            size_per_head=128,
            layer_num=40,
            inter_size=hidden_to_inter(5120),  # 13696
            vocab_size=152064,
            max_seq_len=8192,
        )
        QWen._common_config(config, ckpt_path)
        return config


class QWen_1B8(QWenBase):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = GptInitModelParameters(
            head_num=16,
            head_num_kv=16,
            size_per_head=128,
            layer_num=24,
            inter_size=hidden_to_inter(2048),  # 5504
            vocab_size=151936,
            max_seq_len=2048,
        )
        QWenBase._common_config(config, ckpt_path)
        return config


register_model("qwen", QWen, ["QWenLMHeadModel"])
register_model("qwen_7b", QWen_7B)
register_model("qwen_13b", QWen_13B)
register_model("qwen_1b8", QWen_1B8)
