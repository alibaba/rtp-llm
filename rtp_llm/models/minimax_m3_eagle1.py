import functools
import json
import logging
import os
from typing import List

import torch

from rtp_llm.config.model_config import ModelConfig as PyModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, FfnConfig, FfnWeight
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.qwen_v2 import QWenV2
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    concat_0,
    concat_1,
    identity,
    transpose,
    yarn_get_mscale,
    zeros,
)


class MiniMaxM3Eagle1WeightNames:
    WQ = "layers.{i}.self_attn.q_proj.weight"
    WK = "layers.{i}.self_attn.k_proj.weight"
    WV = "layers.{i}.self_attn.v_proj.weight"
    BQ = "layers.{i}.self_attn.q_proj.bias"
    BK = "layers.{i}.self_attn.k_proj.bias"
    BV = "layers.{i}.self_attn.v_proj.bias"
    WO = "layers.{i}.self_attn.o_proj.weight"
    FFW1 = "layers.{i}.mlp.gate_proj.weight"
    FFW2 = "layers.{i}.mlp.down_proj.weight"
    FFW3 = "layers.{i}.mlp.up_proj.weight"
    ATTEN_NORM = "layers.{i}.input_layernorm.weight"
    FFN_NORM = "layers.{i}.post_attention_layernorm.weight"
    HIDDEN_NORM = "h_norm.weight"
    EMBEDDING_NORM = "e_norm.weight"
    TOKEN_EMBEDDING = "embed_tokens.weight"
    NORM = "norm.weight"
    FC = "eh_proj.weight"


def _merge_qkv_weight(ts: List[torch.Tensor]) -> torch.Tensor:
    q, k, v = ts
    return torch.concat([q.T, k.T, v.T], dim=1).contiguous()


def _merge_qkv_bias(ts: List[torch.Tensor]) -> torch.Tensor:
    q, k, v = ts
    return torch.concat([q, k, v], dim=0).contiguous()


class MiniMaxM3Eagle1WeightInfo(ModelDeployWeightInfo):
    def _process_meta(self, meta_dicts, weight_keys):
        if MiniMaxM3Eagle1WeightNames.FC not in weight_keys:
            raise Exception(
                "unsupported MiniMax-M3 EAGLE1 checkpoint: missing eh_proj.weight. "
                "Only the HASS draft bundle format is supported."
            )
        self._names = MiniMaxM3Eagle1WeightNames

    def _get_weight_info(self):
        names = self._names
        attn_config = AttnConfig(
            hidden_size=self._hidden_size,
            size_per_head=self._size_per_head,
            head_num=self._head_num,
            head_num_kv=self._head_num_kv,
        )
        ffn_config = FfnConfig(
            is_gated_activation=True,
            align_size=self._align_size,
            is_moe=False,
        )
        weights = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo(names.TOKEN_EMBEDDING, concat_1)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo(names.NORM, identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_beta,
                [],
                functools.partial(zeros, shape=[self._hidden_size]),
            ),
        ]
        layer_weights = [
            AtomicWeight(
                W.multi_tokens_predict_eh_proj,
                [CkptWeightInfo(names.FC, identity)],
                transpose,
            ),
            AtomicWeight(
                W.multi_tokens_predict_hnorm,
                [CkptWeightInfo(names.HIDDEN_NORM, identity)],
                identity,
            ),
            AtomicWeight(
                W.multi_tokens_predict_enorm,
                [CkptWeightInfo(names.EMBEDDING_NORM, identity)],
                identity,
            ),
            AtomicWeight(
                W.pre_ln_gamma,
                [CkptWeightInfo(names.ATTEN_NORM, identity)],
                identity,
            ),
            AtomicWeight(
                W.post_ln_gamma,
                [CkptWeightInfo(names.FFN_NORM, identity)],
                identity,
            ),
            AttnAtomicWeight(
                W.attn_o_w,
                [CkptWeightInfo(names.WO, concat_1)],
                transpose,
                config=attn_config,
            ),
            FfnWeight(
                sub_weights=[
                    FfnAtomicWeight(
                        W.ffn_w1,
                        [CkptWeightInfo(names.FFW1, concat_0)],
                        transpose,
                        config=ffn_config,
                    ),
                    FfnAtomicWeight(
                        W.ffn_w3,
                        [CkptWeightInfo(names.FFW3, concat_0)],
                        transpose,
                        config=ffn_config,
                    ),
                    FfnAtomicWeight(
                        W.ffn_w2,
                        [CkptWeightInfo(names.FFW2, concat_1)],
                        transpose,
                        config=ffn_config,
                    ),
                ],
                config=ffn_config,
            ),
            AttnAtomicWeight(
                W.attn_qkv_w,
                [
                    CkptWeightInfo(names.WQ, concat_0),
                    CkptWeightInfo(names.WK, concat_0),
                    CkptWeightInfo(names.WV, concat_0),
                ],
                _merge_qkv_weight,
                config=attn_config,
            ),
        ]
        layer_weights.append(
            AttnAtomicWeight(
                W.attn_qkv_b,
                [
                    CkptWeightInfo(names.BQ, identity),
                    CkptWeightInfo(names.BK, identity),
                    CkptWeightInfo(names.BV, identity),
                ],
                _merge_qkv_bias,
                config=attn_config,
            )
        )
        return ModelWeightInfo(layer_weights=[layer_weights], weights=weights)


class MiniMaxM3Eagle1(QWenV2):
    def _load(self, device: str):
        super()._load(device)

        from rtp_llm.models.minimax_m3 import _get_target_lm_head

        target_lm_head = _get_target_lm_head(device)
        draft_placeholder = self.weight.get_global_weight(W.lm_head)
        if (
            target_lm_head.dtype != draft_placeholder.dtype
            or target_lm_head.device != draft_placeholder.device
        ):
            raise RuntimeError(
                "MiniMax-M3 EAGLE1 target lm_head and draft weights must have "
                "matching dtype and device; "
                f"target={tuple(target_lm_head.shape)}/{target_lm_head.dtype}/"
                f"{target_lm_head.device}, "
                f"draft={tuple(draft_placeholder.shape)}/{draft_placeholder.dtype}/"
                f"{draft_placeholder.device}"
            )
        self.weight.set_global_weight(W.lm_head, target_lm_head)
        logging.info(
            "MiniMax-M3 EAGLE1 reuses target lm_head on %s "
            "(target_data_ptr=%d, discarded_placeholder_data_ptr=%d)",
            device,
            target_lm_head.data_ptr(),
            draft_placeholder.data_ptr(),
        )

    @classmethod
    def _create_config(cls, ckpt_path: str) -> PyModelConfig:
        config = PyModelConfig()
        config.ckpt_path = ckpt_path
        config.attn_config.rope_config.dim = 128
        config.attn_config.rope_config.style = 1
        # Keep the draft output head in checkpoint-native BF16. The framework
        # default upcasts lm_head to FP32, which dispatches a slow full-vocab
        # SIMT GEMM during both draft decode and draft refresh.
        config.enable_fp32_lm_head = False
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            raise Exception("MiniMax-M3 EAGLE1 parameter from unknown source")
        with open(config_path) as reader:
            config_json = json.loads(reader.read())
        QWenV2._from_config_json(config, config_json)
        rope_scaling = config_json.get("rope_scaling")
        if rope_scaling is not None:
            rope_type = rope_scaling.get("type", rope_scaling.get("rope_type"))
            if rope_type != "yarn":
                raise ValueError(
                    "MiniMax-M3 EAGLE1 only supports YaRN rope_scaling, "
                    f"got {rope_type}"
                )
            config.attn_config.rope_config.style = 5
            config.attn_config.rope_config.scale = rope_scaling["factor"]
            config.attn_config.rope_config.factor1 = rope_scaling.get("beta_slow", 1)
            config.attn_config.rope_config.factor2 = rope_scaling.get("beta_fast", 32)
            config.attn_config.rope_config.max_pos = rope_scaling[
                "original_max_position_embeddings"
            ]
            config.attn_config.rope_config.mscale = yarn_get_mscale(
                config.attn_config.rope_config.scale
            )
        if config.num_layers != 1:
            raise ValueError(
                "MiniMax-M3 EAGLE1 checkpoint must define exactly one draft layer, "
                f"got {config.num_layers}"
            )
        rope_parameters = config_json.get("rope_parameters") or {}
        if "rope_theta" in rope_parameters:
            config.attn_config.rope_config.base = int(rope_parameters["rope_theta"])
        config.vocab_size = int(config_json.get("draft_vocab_size", config.vocab_size))
        config.activation_type = "SiGLU"
        config.model_type = "minimax_m3_eagle1"
        config.use_opaque_kv_cache_store = True
        return config

    @staticmethod
    def get_weight_cls():
        return MiniMaxM3Eagle1WeightInfo

    def support_cuda_graph(self) -> bool:
        return True

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.minimax_m3_eagle1 import MiniMaxM3Eagle1Model

        self.py_model = MiniMaxM3Eagle1Model(
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


register_model(
    "minimax_m3_eagle1",
    MiniMaxM3Eagle1,
    ["Qwen2ForCausalLMEagle1HASS"],
)
