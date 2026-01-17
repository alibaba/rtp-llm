import functools
import json
import os

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, FfnWeight
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.base_model import BaseModel
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    qkv_gather,
    transpose,
)


class FalconWeightInfo(ModelDeployWeightInfo):
    def _process_meta(self, meta_dicts, weight_keys):
        if "transformer.h.0.ln_attn.weight" in weight_keys:
            self.falcon_40b = True
        elif "transformer.h.0.input_layernorm.weight" in weight_keys:
            self.falcon_40b = False

    def _get_weight_info(self):
        attn_config = self.attn_config
        ffn_config = self.ffn_config
        weights = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo("transformer.word_embeddings.weight", identity)],
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
                [CkptWeightInfo("transformer.ln_f.bias", identity)],
                identity,
            ),
        ]

        layer_weights = [
            AttnAtomicWeight(
                W.attn_o_w,
                [
                    CkptWeightInfo(
                        "transformer.h.{i}.self_attention.dense.weight", identity
                    )
                ],
                transpose,
                config=attn_config,
            ),
            FfnWeight(
                sub_weights=[
                    FfnAtomicWeight(
                        W.ffn_w3,
                        [
                            CkptWeightInfo(
                                "transformer.h.{i}.mlp.dense_h_to_4h.weight", identity
                            )
                        ],
                        transpose,
                        config=ffn_config,
                    ),
                    FfnAtomicWeight(
                        W.ffn_w2,
                        [
                            CkptWeightInfo(
                                "transformer.h.{i}.mlp.dense_4h_to_h.weight", identity
                            )
                        ],
                        transpose,
                        config=ffn_config,
                    ),
                ],
                config=ffn_config,
            ),
        ]

        if self.falcon_40b:
            layer_weights.extend(
                [
                    AttnAtomicWeight(
                        W.attn_qkv_w,
                        [
                            CkptWeightInfo(
                                "transformer.h.{i}.self_attention.query_key_value.weight",
                                identity,
                            )
                        ],
                        functools.partial(
                            qkv_gather,
                            dim0=self._hidden_size,
                            head_num=self._head_num,
                            head_num_kv=self._head_num_kv,
                        ),
                        config=attn_config,
                    ),
                    AtomicWeight(
                        W.pre_ln_beta,
                        [CkptWeightInfo("transformer.h.{i}.ln_mlp.bias", identity)],
                        identity,
                    ),
                    AtomicWeight(
                        W.pre_ln_gamma,
                        [CkptWeightInfo("transformer.h.{i}.ln_mlp.weight", identity)],
                        identity,
                    ),
                    AtomicWeight(
                        W.pre_attn_ln_beta,
                        [CkptWeightInfo("transformer.h.{i}.ln_attn.bias", identity)],
                        identity,
                    ),
                    AtomicWeight(
                        W.pre_attn_ln_gamma,
                        [CkptWeightInfo("transformer.h.{i}.ln_attn.weight", identity)],
                        identity,
                    ),
                ]
            )

        else:
            layer_weights.extend(
                [
                    AttnAtomicWeight(
                        W.attn_qkv_w,
                        [
                            CkptWeightInfo(
                                "transformer.h.{i}.self_attention.query_key_value.weight",
                                identity,
                            )
                        ],
                        transpose,
                        config=attn_config,
                    ),
                    AtomicWeight(
                        W.pre_ln_beta,
                        [
                            CkptWeightInfo(
                                "transformer.h.{i}.input_layernorm.bias", identity
                            )
                        ],
                        identity,
                    ),
                    AtomicWeight(
                        W.pre_ln_gamma,
                        [
                            CkptWeightInfo(
                                "transformer.h.{i}.input_layernorm.weight", identity
                            )
                        ],
                        identity,
                    ),
                ]
            )

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)


class Falcon(BaseModel):
    @staticmethod
    def get_weight_cls():
        return FalconWeightInfo

    @classmethod
    def _create_config(cls, ckpt_path: str):
        from rtp_llm.model_config_creators.falcon import create_falcon_config

        config = create_falcon_config(ckpt_path)
        return config


register_model("falcon", Falcon, ["FalconForCausalLM"])
