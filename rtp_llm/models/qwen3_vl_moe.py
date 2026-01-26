import json
import os
from typing import Any, List, Optional

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig, MoeWeight
from rtp_llm.models.qwen3_vl import QWen3_VL
from rtp_llm.models.qwen_v2_moe import Qwen2Moe
from rtp_llm.models.qwen_v3_moe import Qwen3Moe, QWenV3MoeWeight
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.qwen3vl_moe import Qwen3VLMoeModel
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    convert_down_proj_,
    convert_gate_up_proj_,
    identity,
    transpose,
)


class QWen3VLMoeWeightInfo(QWenV3MoeWeight):
    def __init__(self, **kwargs):
        QWenV3MoeWeight.__init__(self, **kwargs)
        self.bias = False
        self._use_qk_norm = True

    def _process_meta(self, meta_dicts: Any, weight_keys: List[str]):
        super()._process_meta(meta_dicts, weight_keys)
        self._use_stack_weight = False
        for key in weight_keys:
            if "experts.down_proj" in key or "experts.gate_up_proj" in key:
                self._use_stack_weight = True
                break

    def _get_weight_info(self):
        weights = self._get_hf_weight_info()
        return weights

    def _get_hf_ffn_layer_weight_info(self, layer_id: int):
        moe_config = MoeConfig(
            expert_num=self.expert_num_,
            align_size=self._align_size,
            routed_scaling_factor=1.0,
            weight_stack=True,
        )
        return [
            MoeWeight(
                sub_weights=[
                    MoeAtomicWeight(
                        W.moe_gate,
                        [
                            CkptWeightInfo(
                                self.transformer_prefix + "layers.{i}.mlp.gate.weight",
                                identity,
                            )
                        ],
                        transpose,
                        config=moe_config,
                    ),
                    MoeAtomicWeight(
                        W.moe_w1,
                        [
                            CkptWeightInfo(
                                self.transformer_prefix
                                + "layers.{i}.mlp.experts.gate_up_proj",
                                convert_gate_up_proj_,
                            )
                        ],
                        identity,
                        config=moe_config,
                    ),
                    MoeAtomicWeight(
                        W.moe_w2,
                        [
                            CkptWeightInfo(
                                self.transformer_prefix
                                + "layers.{i}.mlp.experts.down_proj",
                                convert_down_proj_,
                            )
                        ],
                        identity,
                        config=moe_config,
                    ),
                ],
                config=moe_config,
            )
        ]


class QWen3_VL_MOE(Qwen3Moe):
    def _create_python_model(self) -> Optional[GptModelBase]:
        model_config = self.model_config
        parallelism_config = self.parallelism_config
        fmha_config = self.fmha_config
        py_hw_kernel_config = self.hw_kernel_config
        moe_config = self.moe_config
        self.py_model = Qwen3VLMoeModel(
            model_config,
            parallelism_config,
            self.weight,
            moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )

    @staticmethod
    def get_weight_cls():
        return QWen3VLMoeWeightInfo

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.attn_config.rope_config.dim = 128
        config.activation_type = "SiGLU"
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        config.norm_type = "rmsnorm"
        config.qk_norm = True
        cls._from_hf(config, ckpt_path)
        return config

    @classmethod
    def _from_hf(cls, config: ModelConfig, ckpt_path: str):
        config_path = os.path.join(ckpt_path, "config.json")

        if not os.path.exists(config_path):
            return
        with open(config_path) as reader:
            content = reader.read()
            config_json = json.loads(content)
        Qwen2Moe.load_moe_config(config, config_json["text_config"])
        QWen3_VL._from_config_json(config, config_json)
        return config


register_model("qwen3_vl_moe", QWen3_VL_MOE, ["Qwen3VLMoeForConditionalGeneration"])
