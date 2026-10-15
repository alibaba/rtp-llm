import functools
from typing import Any, List

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.ffn_weight import (
    FfnAtomicWeight,
    FfnConfig,
    FfnWeight,
    MoeAtomicWeight,
    MoeConfig,
    MoeWeight,
)
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.qwen_v2 import QWenV2, QWenV2Weight
from rtp_llm.models.qwen_v2_moe import Qwen2Moe, QWenV2MoeWeight
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    merge_qkv_hf,
    stack_,
    stack_moe_w1,
    transpose,
    transpose_pad,
)
from rtp_llm.utils.util import check_get_config_from_path


class QWenV3MoeWeight(QWenV2MoeWeight):
    def __init__(
        self,
        model_config,
        parallelism_config,
        hw_kernel_config,
        kv_cache_config,
        merge_lora=False,
        vit_config=None,
        prefix="",
        **kwargs: Any
    ):
        super().__init__(
            model_config=model_config,
            parallelism_config=parallelism_config,
            hw_kernel_config=hw_kernel_config,
            kv_cache_config=kv_cache_config,
            merge_lora=merge_lora,
            vit_config=vit_config,
            prefix=prefix,
            **kwargs
        )
        self.bias = False

    def _get_hf_ffn_layer_weight_info(self, layer_id: int):
        moe_config = MoeConfig(
            expert_num=self.expert_num_,
            align_size=self._align_size,
            routed_scaling_factor=1.0,
        )
        return [
            MoeWeight(
                sub_weights=[
                    MoeAtomicWeight(
                        W.moe_gate,
                        [CkptWeightInfo("model.layers.{i}.mlp.gate.weight", identity)],
                        transpose,
                        config=moe_config,
                    ),
                    MoeAtomicWeight(
                        W.moe_w1,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.experts.{expert_id}.up_proj.weight",
                                identity,
                            )
                        ]
                        + [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.experts.{expert_id}.gate_proj.weight",
                                identity,
                            )
                        ],
                        stack_moe_w1,
                        config=moe_config,
                    ),
                    MoeAtomicWeight(
                        W.moe_w2,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.experts.{expert_id}.down_proj.weight",
                                identity,
                            )
                        ],
                        stack_,
                        config=moe_config,
                    ),
                ],
                config=moe_config,
            ),
        ]


class Qwen3Moe(Qwen2Moe):
    @staticmethod
    def get_weight_cls():
        return QWenV3MoeWeight

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        config.qk_norm = True
        config.moe_style = 1
        return config

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.generic_moe import GenericMoeModel

        model_config = self.model_config
        parallelism_config = self.parallelism_config
        fmha_config = self.fmha_config
        py_hw_kernel_config = self.hw_kernel_config
        moe_config = self.moe_config
        max_generate_batch_size = self.max_generate_batch_size

        self.py_model = GenericMoeModel(
            model_config,
            parallelism_config,
            self.weight,
            moe_config,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model


class AngelSlimQwen3Eagle3Weight(QWenV2Weight):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bias = False

    @staticmethod
    def decode_draft_to_target_map(tensors: List[torch.Tensor]) -> torch.Tensor:
        offsets = tensors[0].to(dtype=torch.int64)
        return offsets + torch.arange(
            offsets.numel(), dtype=torch.int64, device=offsets.device
        )

    def _get_weight_info(self):
        assert self._num_layers == 1
        attn_config = AttnConfig(
            hidden_size=self._hidden_size,
            size_per_head=self._size_per_head,
            head_num=self._head_num,
            head_num_kv=self._head_num_kv,
        )
        ffn_config = FfnConfig(
            is_gated_activation=self._is_gated_activation,
            align_size=self._align_size,
            is_moe=False,
        )
        layer_weights: List[List[WeightModule]] = [
            [
                AttnAtomicWeight(
                    W.attn_qkv_w,
                    [
                        CkptWeightInfo("midlayer.self_attn.q_proj.weight", identity),
                        CkptWeightInfo("midlayer.self_attn.k_proj.weight", identity),
                        CkptWeightInfo("midlayer.self_attn.v_proj.weight", identity),
                    ],
                    merge_qkv_hf,
                    config=attn_config,
                ),
                AttnAtomicWeight(
                    W.attn_o_w,
                    [CkptWeightInfo("midlayer.self_attn.o_proj.weight", identity)],
                    transpose,
                    config=attn_config,
                ),
                AtomicWeight(
                    W.post_ln_gamma,
                    [
                        CkptWeightInfo(
                            "midlayer.post_attention_layernorm.weight", identity
                        )
                    ],
                    identity,
                ),
                FfnWeight(
                    sub_weights=[
                        FfnAtomicWeight(
                            W.ffn_w1,
                            [CkptWeightInfo("midlayer.mlp.gate_proj.weight", identity)],
                            functools.partial(
                                transpose_pad, align_size=self._align_size, dim=0
                            ),
                            config=ffn_config,
                        ),
                        FfnAtomicWeight(
                            W.ffn_w3,
                            [CkptWeightInfo("midlayer.mlp.up_proj.weight", identity)],
                            functools.partial(
                                transpose_pad, align_size=self._align_size, dim=0
                            ),
                            config=ffn_config,
                        ),
                        FfnAtomicWeight(
                            W.ffn_w2,
                            [CkptWeightInfo("midlayer.mlp.down_proj.weight", identity)],
                            functools.partial(
                                transpose_pad, align_size=self._align_size, dim=1
                            ),
                            config=ffn_config,
                        ),
                    ],
                    config=ffn_config,
                ),
                AtomicWeight(
                    W.eagle3_fc_proj,
                    [CkptWeightInfo("fc.weight", identity)],
                    transpose,
                ),
                AtomicWeight(
                    W.eagle3_fc_norm_gamma,
                    [CkptWeightInfo("midlayer.hidden_norm.weight", identity)],
                    identity,
                ),
                AtomicWeight(
                    W.eagle3_input_norm_gamma,
                    [CkptWeightInfo("midlayer.input_layernorm.weight", identity)],
                    identity,
                ),
            ]
        ]
        weights = [
            AtomicWeight(W.embedding, [], identity),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo("lm_head.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo("norm.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.multi_tokens_predict_d2t_map,
                [CkptWeightInfo("d2t", identity)],
                self.decode_draft_to_target_map,
                data_type=torch.int64,
            ),
            AtomicWeight(
                W.multi_tokens_predict_t2d_map,
                [CkptWeightInfo("t2d", identity)],
                identity,
                data_type=torch.bool,
            ),
        ]
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)


class AngelSlimQwen3Eagle3(QWenV2):
    @classmethod
    def target_aux_hidden_capture_layer_ids(
        cls,
        target_model_config: ModelConfig,
        draft_model_config: ModelConfig,
    ) -> tuple[int, ...]:
        return (
            2,
            target_model_config.num_layers // 2,
            target_model_config.num_layers - 3,
        )

    @classmethod
    def speculative_weight_alias_names(cls, target_model, draft_model_config):
        target_config = target_model.model_config
        if target_config.model_type not in ("qwen_3", "qwen_3_moe"):
            raise TypeError(
                "AngelSlim Qwen3 Eagle3 requires a qwen_3 or qwen_3_moe target model"
            )
        mismatches = []
        for name in ("hidden_size", "data_type"):
            if getattr(target_config, name) != getattr(draft_model_config, name):
                mismatches.append(name)
        if target_config.vocab_size != draft_model_config.input_vocab_size:
            mismatches.append("input_vocab_size")
        if mismatches:
            raise ValueError(
                "AngelSlim Qwen3 Eagle3 cannot alias an incompatible target embedding: "
                + ", ".join(mismatches)
            )
        return (W.embedding,)

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        config_json = check_get_config_from_path(ckpt_path)
        config.input_vocab_size = config.vocab_size
        config.vocab_size = int(config_json.get("draft_vocab_size", config.vocab_size))
        config.qk_norm = False
        config.is_mtp = True
        return config

    @staticmethod
    def get_weight_cls():
        return AngelSlimQwen3Eagle3Weight

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.qwen3 import AngelSlimQwen3Eagle3Model

        self.py_model = AngelSlimQwen3Eagle3Model(
            self.model_config,
            self.parallelism_config,
            self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            quant_config=self.model_config.quant_config,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model


register_model("qwen_3_moe", Qwen3Moe, ["Qwen3MoeForCausalLM"])
register_model(
    "angelslim_qwen3_eagle3",
    AngelSlimQwen3Eagle3,
    ["LlamaForCausalLMEagle3"],
)
register_model("qwen3_coder_moe", Qwen3Moe, [])
