import json
import os
from typing import Any, List

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig, MoeWeight
from rtp_llm.models.multimodal.multimodal_mixin import (
    BaseMultiModalWeightInfo,
    MultiModalMixin,
)
from rtp_llm.models.qwen3_vl.qwen3_vl import (
    QWen3_VL,
    Qwen3_VLImageEmbedding,
    Qwen3VLVitWeight,
)
from rtp_llm.models.qwen_v2_moe import Qwen2Moe
from rtp_llm.models.qwen_v3_moe import Qwen3Moe, QWenV3MoeWeight
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    convert_down_proj_,
    convert_gate_up_proj_,
    identity,
    stack_,
    transpose,
)


class QWen3VLMoeWeightInfo(QWenV3MoeWeight, BaseMultiModalWeightInfo):
    def __init__(self, vit_weights, **kwargs):
        QWenV3MoeWeight.__init__(self, **kwargs)
        BaseMultiModalDeployWeightInfo.__init__(self, vit_weights=vit_weights, **kwargs)
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
            inter_padding_size=(
                self._layer_inter_padding_size[layer_id]
                if self._layer_inter_padding_size
                else self._inter_padding_size
            ),
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


class QWen3_VL_MOE(Qwen3Moe, MultiModalMixin):
    def _init_multimodal(self):
        self.mm_part = Qwen3_VLImageEmbedding(self.model_config.mm_related_params)
        self.model_config.mm_related_params.vit_weights = Qwen3VLVitWeight(
            {"vit": self.mm_part.visual}
        )

    @classmethod
    def _get_mm_module(cls, config: ModelConfig):
        return Qwen3_VLImageEmbedding(config).visual

    @staticmethod
    def get_weight_cls():
        return QWen3VLMoeWeightInfo

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = ModelConfig()
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
        QWen3_VL._from_config_json(config, config_json)
        Qwen2Moe.load_moe_config(config, config_json["text_config"])
        return config


register_model("qwen3_vl_moe", QWen3_VL_MOE, ["Qwen3VLMoeForConditionalGeneration"])
