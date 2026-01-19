import functools
import json
import os
from typing import List, Optional

from rtp_llm.config.model_config import ModelConfig
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
        align_size = self._align_size
        attn_config = AttnConfig(
            hidden_size=self._hidden_size,
            size_per_head=self._size_per_head,
            head_num=self._head_num,
            head_num_kv=self._head_num_kv,
        )
        ffn_config = FfnConfig(
            is_gated_activation=self._is_gated_activation,
            align_size=align_size,
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
                        [CkptWeightInfo("transformer.h.{i}.mlp.w1.weight", identity)],
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
                                "transformer.h.{i}.mlp.c_proj.weight", identity
                            )
                        ],
                        functools.partial(transpose_pad, align_size=align_size, dim=1),
                        config=ffn_config,
                        lora_a_process_func=functools.partial(
                            transpose_pad, align_size=align_size, dim=0
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
        model_config = self.model_config
        parallelism_config = self.parallelism_config
        ffn_disaggregate_config = parallelism_config.ffn_disaggregate_config
        fmha_config = self.fmha_config
        py_hw_kernel_config = self.hw_kernel_config
        quant_config = self.model_config.quant_config

        if ffn_disaggregate_config.enable_ffn_disaggregate:
            self.py_model = Qwen3DisaggregateModel(
                model_config,
                parallelism_config,
                self.weight,
                max_generate_batch_size=self.max_generate_batch_size,
                fmha_config=fmha_config,
                py_hw_kernel_config=py_hw_kernel_config,
                device_resource_config=self.device_resource_config,
            )
        else:
            self.py_model = Qwen3Model(
                model_config,
                parallelism_config,
                self.weight,
                max_generate_batch_size=self.max_generate_batch_size,
                quant_config=quant_config,
                fmha_config=fmha_config,
                py_hw_kernel_config=py_hw_kernel_config,
                device_resource_config=self.device_resource_config,
            )

    def support_cuda_graph(self) -> bool:
        return True

    @staticmethod
    def _common_config(config: ModelConfig, ckpt_path: str) -> ModelConfig:
        from rtp_llm.model_config_creators.qwen import create_qwen_base_config

        create_qwen_base_config(config, ckpt_path)
        return config


class QWen(QWenBase):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        from rtp_llm.model_config_creators.qwen import create_qwen_config

        config = create_qwen_config(ckpt_path)
        return config


class QWen_7B(QWenBase):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        from rtp_llm.model_config_creators.qwen import create_qwen_7b_config

        config = create_qwen_7b_config(ckpt_path)
        return config


class QWen_13B(QWenBase):
    pass


class QWen_1B8(QWenBase):
    pass


register_model("qwen", QWen, ["QWenLMHeadModel"])
register_model("qwen_7b", QWen_7B)
register_model("qwen_13b", QWen_13B)
register_model("qwen_1b8", QWen_1B8)
