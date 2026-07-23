import functools
import json
import os
from typing import List

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, FfnConfig, FfnWeight
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.qwen_v2 import QWenV2
from rtp_llm.ops import SpeculativeType
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    concat_0,
    concat_1,
    identity,
    stack_,
    transpose,
    zeros,
)


class MiniMaxM3Eagle3WeightNames:
    TOKEN_EMBEDDING = "embed_tokens.weight"
    LM_HEAD = "lm_head.weight"
    FINAL_NORM = "norm.weight"
    FC = "fc.weight"
    FC_NORMS = [f"fc_norm.{i}.weight" for i in range(3)]
    HIDDEN_NORM = "layers.0.hidden_norm.weight"
    INPUT_NORM = "layers.0.input_layernorm.weight"
    POST_ATTN_NORM = "layers.0.post_attention_layernorm.weight"
    WQ = "layers.0.self_attn.q_proj.weight"
    WK = "layers.0.self_attn.k_proj.weight"
    WV = "layers.0.self_attn.v_proj.weight"
    WO = "layers.0.self_attn.o_proj.weight"
    FFW1 = "layers.0.mlp.gate_proj.weight"
    FFW2 = "layers.0.mlp.down_proj.weight"
    FFW3 = "layers.0.mlp.up_proj.weight"

    @classmethod
    def required(cls) -> set[str]:
        return {
            cls.TOKEN_EMBEDDING,
            cls.LM_HEAD,
            cls.FINAL_NORM,
            cls.FC,
            *cls.FC_NORMS,
            cls.HIDDEN_NORM,
            cls.INPUT_NORM,
            cls.POST_ATTN_NORM,
            cls.WQ,
            cls.WK,
            cls.WV,
            cls.WO,
            cls.FFW1,
            cls.FFW2,
            cls.FFW3,
        }


def _merge_qkv_weight(tensors: List[torch.Tensor]) -> torch.Tensor:
    q, k, v = tensors
    return torch.concat([q.T, k.T, v.T], dim=1).contiguous()


class MiniMaxM3Eagle3WeightInfo(ModelDeployWeightInfo):
    def _process_meta(self, meta_dicts, weight_keys):
        required = MiniMaxM3Eagle3WeightNames.required()
        actual = set(weight_keys)
        missing = sorted(required - actual)
        unexpected = sorted(actual - required)
        if missing or unexpected:
            details = []
            if missing:
                details.append("missing weights: " + ", ".join(missing))
            if unexpected:
                details.append("unexpected weights: " + ", ".join(unexpected))
            raise ValueError(
                "unsupported MiniMax-M3 EAGLE3 checkpoint; " + "; ".join(details)
            )

    def _get_weight_info(self):
        names = MiniMaxM3Eagle3WeightNames
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
        global_weights = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo(names.TOKEN_EMBEDDING, concat_1)],
                identity,
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo(names.LM_HEAD, identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo(names.FINAL_NORM, identity)],
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
                W.eagle3_fc_proj,
                [CkptWeightInfo(names.FC, identity)],
                transpose,
            ),
            AtomicWeight(
                W.eagle3_aux_norm_gamma,
                [CkptWeightInfo(name, identity) for name in names.FC_NORMS],
                stack_,
            ),
            AtomicWeight(
                W.eagle3_fc_norm_gamma,
                [CkptWeightInfo(names.HIDDEN_NORM, identity)],
                identity,
            ),
            AtomicWeight(
                W.eagle3_input_norm_gamma,
                [CkptWeightInfo(names.INPUT_NORM, identity)],
                identity,
            ),
            AtomicWeight(
                W.post_ln_gamma,
                [CkptWeightInfo(names.POST_ATTN_NORM, identity)],
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
        return ModelWeightInfo(
            layer_weights=[layer_weights],
            weights=global_weights,
        )


class MiniMaxM3Eagle3(QWenV2):
    _NUM_AUX_HIDDEN_STATES = 3

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"MiniMax-M3 EAGLE3 config does not exist: {config_path}"
            )
        with open(config_path, encoding="utf-8") as reader:
            config_json = json.load(reader)

        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.attn_config.rope_config.style = 1
        QWenV2._from_config_json(config, config_json)

        if config.num_layers != 1:
            raise ValueError(
                "MiniMax-M3 EAGLE3 checkpoint must define exactly one draft layer, "
                f"got {config.num_layers}"
            )
        if config.attn_config.size_per_head != 128:
            raise ValueError(
                "MiniMax-M3 EAGLE3 requires head_dim=128, "
                f"got {config.attn_config.size_per_head}"
            )
        if not bool(config_json.get("fc_norm", False)):
            raise ValueError("MiniMax-M3 EAGLE3 checkpoint requires fc_norm=true")
        if not bool(config_json.get("norm_output", False)):
            raise ValueError("MiniMax-M3 EAGLE3 checkpoint requires norm_output=true")

        aux_layer_ids = config_json.get("eagle_aux_hidden_state_layer_ids")
        if aux_layer_ids:
            config._minimax_m3_eagle3_aux_hidden_state_layer_ids = tuple(
                int(layer_id) for layer_id in aux_layer_ids
            )
        # The draft's recurrent hidden state is H-wide. The three target
        # hidden states are a one-time FC input, not an expanded residual.
        config.hc_mult = 1
        config.vocab_size = int(config_json.get("draft_vocab_size", config.vocab_size))
        config.activation_type = "SiGLU"
        config.model_type = "minimax_m3_eagle3"
        config.has_post_decoder_layernorm = True
        config.enable_fp32_lm_head = False
        config.use_opaque_kv_cache_store = True
        return config

    @classmethod
    def configure_speculative_model(
        cls,
        sp_config,
        target_config: ModelConfig,
        draft_config: ModelConfig,
    ) -> None:
        if sp_config.type != SpeculativeType.EAGLE3:
            raise ValueError(
                "MiniMax-M3 EAGLE3 draft requires SP_TYPE=eagle3, "
                f"got {sp_config.type.name.lower()}"
            )

        layer_ids = getattr(
            draft_config,
            "_minimax_m3_eagle3_aux_hidden_state_layer_ids",
            None,
        )
        if not layer_ids:
            if target_config.num_layers >= 6:
                layer_ids = (
                    2,
                    target_config.num_layers // 2,
                    target_config.num_layers - 3,
                )
            else:
                layer_ids = (
                    1,
                    (target_config.num_layers + 1) // 2,
                    target_config.num_layers,
                )

        layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        if (
            len(layer_ids) != cls._NUM_AUX_HIDDEN_STATES
            or layer_ids != tuple(sorted(set(layer_ids)))
            or any(
                layer_id < 0 or layer_id > target_config.num_layers
                for layer_id in layer_ids
            )
        ):
            raise ValueError(
                "invalid MiniMax-M3 EAGLE3 auxiliary hidden-state layers: "
                f"layers={layer_ids}, target_layers={target_config.num_layers}"
            )

        target_config._minimax_m3_eagle3_aux_hidden_state_layer_ids = layer_ids
        target_config.hc_mult = cls._NUM_AUX_HIDDEN_STATES
        draft_config.hc_mult = 1

    @staticmethod
    def get_weight_cls():
        return MiniMaxM3Eagle3WeightInfo

    def support_cuda_graph(self) -> bool:
        return True

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.minimax_m3_eagle3 import MiniMaxM3Eagle3Model

        self.py_model = MiniMaxM3Eagle3Model(
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
    "minimax_m3_eagle3",
    MiniMaxM3Eagle3,
    ["LlamaForCausalLMEagle3"],
)
