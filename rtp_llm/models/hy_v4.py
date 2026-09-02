"""HY V4 model registration, configuration and checkpoint mapping."""

from __future__ import annotations

import functools
import json
import logging
import os
from typing import Any, List, Optional

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.attn_weight import MlaAttnAtomicWeight, MlaConfig
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
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.deepseek_v2 import DeepSeekV2Weight
from rtp_llm.ops import MlaOpsType
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    stack_,
    transpose,
    transpose_pad,
    zeros,
)


def _transpose_stacked_gate_up(ts: List[torch.Tensor]) -> torch.Tensor:
    """Convert per-expert checkpoint ``[gate, up]`` to RTP ``[up, gate]``."""
    # ``stack_`` has an uint8-view fallback for FP8 tensors on runtimes where
    # torch.stack does not implement FP8 directly.
    stacked = stack_(ts)
    if stacked.dim() != 3 or stacked.size(1) % 2:
        raise ValueError(
            "HY V4 fused gate_up_proj must split evenly on dimension 1, "
            f"got {tuple(stacked.shape)}"
        )
    half = stacked.size(1) // 2
    return torch.cat((stacked[:, half:, :], stacked[:, :half, :]), dim=1)


class Hy4Weight(DeepSeekV2Weight):
    """Checkpoint mapping for HY V4 backbone weights."""

    has_fused_experts = False
    fused_gate_up_suffix = ""
    fused_down_suffix = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Metadata may be processed once for pretrain shards and again for a
        # finetune overlay. Keep the detection state per loader instance and
        # accumulate it across both passes.
        self.q_use_lora = False
        self.has_e_score_correction_bias = False
        self.has_fused_experts = False
        self.fused_gate_up_suffix = ""
        self.fused_down_suffix = ""
        super().__init__(*args, **kwargs)

    def _process_meta(self, meta_dict: Any, weight_keys: List[str]):
        super()._process_meta(meta_dict, weight_keys)
        self.has_fused_experts = self.has_fused_experts or any(
            ".mlp.experts.gate_up_proj" in key for key in weight_keys
        )
        if any(
            key.endswith(".mlp.experts.gate_up_proj.weight")
            for key in weight_keys
        ):
            self.fused_gate_up_suffix = ".weight"
        if any(key.endswith(".mlp.experts.down_proj.weight") for key in weight_keys):
            self.fused_down_suffix = ".weight"

    def _mla_config(self) -> MlaConfig:
        return MlaConfig(
            head_num=self._head_num,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            ope_head_dim=self.nope_head_dim,
            v_head_dim=self.v_head_dim,
            use_mla=self.model_config.attn_config.use_mla
            and self.model_config.mla_ops_type != MlaOpsType.MHA,
            q_use_lora=self.q_use_lora,
        )

    def _get_hf_layer_weight_info(self, layer_id: int):
        layer_weights = super()._get_hf_layer_weight_info(layer_id)
        attn_config = self._mla_config()
        layer_weights.extend(
            [
                MlaAttnAtomicWeight(
                    W.attn_gate_w,
                    [
                        CkptWeightInfo(
                            "model.layers.{i}.self_attn.linear_gate.weight", identity
                        )
                    ],
                    transpose,
                    config=attn_config,
                ),
                AtomicWeight(
                    W.hy4_attn_sink,
                    [
                        CkptWeightInfo(
                            "model.layers.{i}.self_attn.learnable_sink_param",
                            identity,
                        )
                    ],
                    identity,
                    data_type=torch.float32,
                ),
            ]
        )
        if self.model_config.enable_ihc:
            for kind in ("attn", "mlp"):
                prefix = f"model.layers.{{i}}.hc_{kind}_layer.hc_pre"
                layer_weights.extend(
                    [
                        AtomicWeight(
                            getattr(W, f"hy4_ihc_{kind}_fn"),
                            [CkptWeightInfo(prefix + ".hc_fn", identity)],
                            identity,
                            data_type=torch.float32,
                        ),
                        AtomicWeight(
                            getattr(W, f"hy4_ihc_{kind}_scale"),
                            [CkptWeightInfo(prefix + ".hc_scale", identity)],
                            identity,
                            data_type=torch.float32,
                        ),
                        AtomicWeight(
                            getattr(W, f"hy4_ihc_{kind}_base"),
                            [CkptWeightInfo(prefix + ".hc_base", identity)],
                            identity,
                            data_type=torch.float32,
                        ),
                    ]
                )
        # Quantization exclusion entries in ModelOpt are concrete per-layer
        # module names. Preserve the layer id on each descriptor so the MXFP8
        # loader can leave excluded BF16 modules (for example linear_gate)
        # untouched without disabling quantization for every layer.
        for weight in layer_weights:
            for component in weight.get_components():
                component.layer_id = layer_id
        return layer_weights

    def _get_hf_ffn_layer_weight_info(self, layer_id: int):
        if layer_id not in self.moe_layer_index_ or not self.has_fused_experts:
            layer_weights = super()._get_hf_ffn_layer_weight_info(layer_id)
            if layer_id in self.moe_layer_index_:
                for weight in layer_weights:
                    for component in weight.get_components():
                        if component.name == W.moe_gate:
                            component.data_type = torch.float32
            return layer_weights

        align_size = self._align_size
        ffn_config = FfnConfig(
            align_size=align_size,
            is_gated_activation=self._is_gated_activation,
            is_moe=False,
        )
        moe_config = MoeConfig(align_size=align_size, expert_num=self.expert_num_)
        layer_weights: List[WeightModule] = [
            FfnWeight(
                sub_weights=[
                    FfnAtomicWeight(
                        W.ffn_w1,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.shared_experts.gate_proj.weight",
                                identity,
                            )
                        ],
                        functools.partial(transpose_pad, align_size=align_size, dim=0),
                        config=ffn_config,
                    ),
                    FfnAtomicWeight(
                        W.ffn_w2,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.shared_experts.down_proj.weight",
                                identity,
                            )
                        ],
                        functools.partial(transpose_pad, align_size=align_size, dim=1),
                        config=ffn_config,
                    ),
                    FfnAtomicWeight(
                        W.ffn_w3,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.shared_experts.up_proj.weight",
                                identity,
                            )
                        ],
                        functools.partial(transpose_pad, align_size=align_size, dim=0),
                        config=ffn_config,
                    ),
                ],
                config=ffn_config,
            ),
            MoeWeight(
                sub_weights=[
                    MoeAtomicWeight(
                        W.moe_gate,
                        [CkptWeightInfo("model.layers.{i}.mlp.gate.weight", identity)],
                        transpose,
                        data_type=torch.float32,
                        config=moe_config,
                    ),
                    MoeAtomicWeight(
                        W.moe_w2,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.experts.down_proj"
                                + self.fused_down_suffix
                            )
                        ],
                        stack_,
                        config=moe_config,
                        stacked_ckpt_keys=True,
                    ),
                    MoeAtomicWeight(
                        W.moe_w1,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.experts.gate_up_proj"
                                + self.fused_gate_up_suffix
                            )
                        ],
                        _transpose_stacked_gate_up,
                        config=moe_config,
                        stacked_ckpt_keys=True,
                    ),
                ],
                config=moe_config,
            ),
        ]
        if self.has_e_score_correction_bias:
            layer_weights.append(
                AtomicWeight(
                    W.e_score_correction_b,
                    [
                        CkptWeightInfo(
                            "model.layers.{i}.mlp.gate.e_score_correction_bias",
                            identity,
                        )
                    ],
                    identity,
                    data_type=torch.float32,
                )
            )
        return layer_weights

    def _get_weight_info(self):
        info = super()._get_weight_info()
        if self.model_config.enable_ihc:
            info.weights.extend(
                [
                    AtomicWeight(
                        W.hy4_ihc_head_fn,
                        [CkptWeightInfo("model.hc_head.hc_head_fn", identity)],
                        identity,
                        data_type=torch.float32,
                    ),
                    AtomicWeight(
                        W.hy4_ihc_head_scale,
                        [CkptWeightInfo("model.hc_head.hc_head_scale", identity)],
                        identity,
                        data_type=torch.float32,
                    ),
                    AtomicWeight(
                        W.hy4_ihc_head_base,
                        [CkptWeightInfo("model.hc_head.hc_head_base", identity)],
                        identity,
                        data_type=torch.float32,
                    ),
                ]
            )
        return info


class Hy4MtpWeight(Hy4Weight):
    """Load the single draft block stored below ``model.mtp_layers.0``."""

    def _process_meta(self, meta_dict: Any, weight_keys: List[str]):
        # Parent metadata probing only visits backbone layer names.
        self.q_use_lora = self.q_use_lora or (
            "model.mtp_layers.0.self_attn.q_a_proj.weight" in weight_keys
        )
        self.has_e_score_correction_bias = self.has_e_score_correction_bias or (
            "model.mtp_layers.0.mlp.gate.e_score_correction_bias" in weight_keys
        )
        self.has_fused_experts = self.has_fused_experts or any(
            key.startswith("model.mtp_layers.0.mlp.experts.gate_up_proj")
            for key in weight_keys
        )
        if "model.mtp_layers.0.mlp.experts.gate_up_proj.weight" in weight_keys:
            self.fused_gate_up_suffix = ".weight"
        if "model.mtp_layers.0.mlp.experts.down_proj.weight" in weight_keys:
            self.fused_down_suffix = ".weight"

    @staticmethod
    def _remap_to_mtp(layer_weights: List[WeightModule]) -> None:
        for weight in layer_weights:
            for component in weight.get_components():
                for ckpt in getattr(component, "weights", ()):
                    ckpt.name = ckpt.name.replace(
                        "model.layers.{i}", "model.mtp_layers.0"
                    )

    def _get_weight_info(self):
        assert self._num_layers == 1
        layer_weights = [self._get_hf_layer_weight_info(0)]
        self._remap_to_mtp(layer_weights[0])
        weights: List[WeightModule] = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo("model.embed_tokens.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo("lm_head.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.multi_tokens_predict_enorm,
                [CkptWeightInfo("model.mtp_layers.0.enorm.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.multi_tokens_predict_hnorm,
                [CkptWeightInfo("model.mtp_layers.0.hnorm.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.multi_tokens_predict_eh_proj,
                [CkptWeightInfo("model.mtp_layers.0.eh_proj.weight", identity)],
                transpose,
            ),
            AtomicWeight(
                W.multi_tokens_predict_final_ln_gamma,
                [
                    CkptWeightInfo(
                        "model.mtp_layers.0.final_layernorm.weight", identity
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
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)


class Hy4(BaseModel):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        cmp_value = os.environ.get("RTP_LLM_GLM5_CMP", "0").strip().lower()
        if cmp_value in ("1", "true", "yes", "on"):
            raise ValueError(
                "RTP_LLM_GLM5_CMP is GLM-5-specific and cannot be enabled for HY V4"
            )
        if cmp_value not in ("0", "false", "no", "off", ""):
            raise ValueError(f"invalid RTP_LLM_GLM5_CMP={cmp_value!r}")
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {ckpt_path}")
        with open(config_path) as reader:
            raw = json.load(reader)

        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.model_type = "hy_v4"
        config.norm_type = "rmsnorm"
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        config.activation_type = "SiGLU"

        config.hidden_size = int(raw["hidden_size"])
        config.num_layers = int(raw["num_hidden_layers"])
        config.vocab_size = int(raw["vocab_size"])
        config.max_seq_len = int(raw["max_position_embeddings"])
        config.layernorm_eps = float(raw.get("rms_norm_eps", 1e-6))
        config.tie_word_embeddings = bool(raw.get("tie_word_embeddings", False))
        config.enable_fp32_lm_head = bool(raw.get("enable_lm_head_fp32", True))
        config.config_dtype = raw.get("dtype", raw.get("torch_dtype"))

        attn = config.attn_config
        attn.use_mla = True
        attn.head_num = int(raw["num_attention_heads"])
        attn.kv_head_num = int(raw.get("num_key_value_heads", attn.head_num))
        attn.q_lora_rank = int(raw["q_lora_rank"])
        attn.kv_lora_rank = int(raw["kv_lora_rank"])
        attn.nope_head_dim = int(raw["qk_nope_head_dim"])
        attn.rope_head_dim = int(raw["qk_rope_head_dim"])
        attn.v_head_dim = int(raw["v_head_dim"])
        attn.size_per_head = attn.nope_head_dim + attn.rope_head_dim
        attn.rope_config.dim = attn.rope_head_dim
        rope_parameters = raw.get("rope_parameters") or {}
        attn.rope_config.base = float(
            rope_parameters.get("rope_theta", raw.get("rope_theta", 10_000_000))
        )
        attn.rope_config.style = 0 if config.mla_ops_type != MlaOpsType.MHA else 5
        attn.rope_config.offset = attn.nope_head_dim
        attn.rope_config.is_neox_style = not bool(raw.get("rope_interleave", True))
        attn.rope_config.indexer_is_neox_style = not bool(
            raw.get("indexer_rope_interleave", True)
        )

        attn.is_sparse = True
        attn.indexer_head_dim = int(raw["index_head_dim"])
        attn.indexer_head_num = int(raw["index_n_heads"])
        attn.indexer_topk = int(raw["index_topk"])
        config.indexer_types = list(raw.get("indexer_types") or [])
        if not config.indexer_types:
            raise KeyError("HY V4 config must contain indexer_types")
        if len(config.indexer_types) != config.num_layers:
            raise ValueError(
                "HY V4 indexer_types length must equal num_hidden_layers, got "
                f"{len(config.indexer_types)} and {config.num_layers}"
            )
        invalid_indexers = set(config.indexer_types) - {"full", "shared"}
        if invalid_indexers:
            raise ValueError(
                f"unsupported HY V4 indexer types: {sorted(invalid_indexers)}"
            )
        if config.indexer_types[0] != "full":
            raise ValueError("HY V4 layer 0 indexer must be 'full'")

        config.enable_ihc = bool(raw.get("enable_ihc", True))
        config.hc_mult = int(raw.get("hc_mult", 4))
        config.hc_magnitude = float(raw.get("hc_magnitude", 2.0))
        config.hc_eps = float(raw.get("hc_eps", 1e-6))
        config.gated_mla = bool(raw.get("gated_mla", True))
        config.gating_type = str(raw.get("gating_type", "elementwise"))
        config.learnable_sink = bool(raw.get("learnable_sink", True))
        config.learnable_sink_init = float(raw.get("learnable_sink_init", 0.0))
        if not config.enable_ihc or config.hc_mult != 4:
            raise ValueError("HY V4 backbone requires enable_ihc=true and hc_mult=4")
        if not config.gated_mla or config.gating_type != "elementwise":
            raise ValueError("HY V4 requires elementwise gated MLA")
        if not config.learnable_sink:
            raise ValueError("HY V4 requires learnable attention sinks")
        config.force_sparse_mla = True

        config.dense_inter_size = int(raw["intermediate_size"])
        config.moe_inter_size = int(raw["moe_intermediate_size"])
        n_shared = int(raw.get("n_shared_experts", 1))
        config.inter_size = n_shared * config.moe_inter_size
        config.expert_num = int(raw["n_routed_experts"])
        config.moe_k = int(raw["num_experts_per_tok"])
        config.moe_style = 2
        config.moe_n_group = int(raw.get("n_group", 1))
        config.moe_topk_group = int(raw.get("topk_group", 1))
        scoring = raw.get("scoring_func", "sigmoid")
        if scoring != "sigmoid":
            raise ValueError(f"HY V4 only supports sigmoid routing, got {scoring!r}")
        config.scoring_func = 1
        config.has_moe_norm = bool(raw.get("norm_topk_prob", True))
        config.routed_scaling_factor = float(raw.get("routed_scaling_factor", 2.827))
        config.swiglu_limit = float(raw.get("swiglu_limit", 10.0))
        if config.swiglu_limit <= 0:
            raise ValueError("HY V4 routed experts require a positive swiglu_limit")
        config.mlp_layer_types = list(raw.get("mlp_layer_types") or [])
        if not config.mlp_layer_types:
            raise KeyError("HY V4 config must contain mlp_layer_types")
        if len(config.mlp_layer_types) != config.num_layers:
            raise ValueError(
                "HY V4 mlp_layer_types length must equal num_hidden_layers, got "
                f"{len(config.mlp_layer_types)} and {config.num_layers}"
            )
        invalid_mlp = set(config.mlp_layer_types) - {"dense", "sparse"}
        if invalid_mlp:
            raise ValueError(f"unsupported HY V4 MLP layer types: {invalid_mlp}")
        config.moe_layer_index = [
            i for i, layer_type in enumerate(config.mlp_layer_types)
            if layer_type == "sparse"
        ]
        logging.info(
            "HY V4 config loaded: layers=%d hidden=%d heads=%d iHC=%d "
            "experts=%d topk=%d",
            config.num_layers,
            config.hidden_size,
            attn.head_num,
            config.hc_mult,
            config.expert_num,
            attn.indexer_topk,
        )
        return config

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.hy_v4_model import Hy4Model

        self.py_model = Hy4Model(
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

    def support_cuda_graph(self) -> bool:
        return True

    @staticmethod
    def get_weight_cls():
        return Hy4Weight


class Hy4Mtp(Hy4):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = super()._create_config(ckpt_path)
        raw_path = os.path.join(ckpt_path, "config.json")
        with open(raw_path) as reader:
            raw = json.load(reader)
        config.num_layers = int(raw.get("num_nextn_predict_layers", 1)) or 1
        if config.num_layers != 1:
            raise ValueError(
                f"HY V4 RTP draft currently supports one MTP layer, got {config.num_layers}"
            )
        config.enable_ihc = False
        # HY V4 applies hc_head in the target model before handing hidden states
        # to MTP.  Its draft therefore consumes one hidden-size stream, unlike
        # DeepSeek V4 MTP which consumes the pre-hc_head hc_mult streams.
        config.hc_mult = 1
        config.mlp_layer_types = ["sparse"]
        config.moe_layer_index = [0]
        config.indexer_types = ["full"]
        config.index_share_for_mtp_iteration = False
        config.is_mtp = True
        config.model_type = "hy_v4_mtp"
        return config

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.hy_v4_mtp_model import Hy4MtpModel

        self.py_model = Hy4MtpModel(
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

    @staticmethod
    def get_weight_cls():
        return Hy4MtpWeight


register_model("hy_v4", Hy4, ["HYV4ForCausalLM"])
register_model("hy_v4_mtp", Hy4Mtp, ["HYV4ForCausalLMNextN"])
