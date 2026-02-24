import functools
import json
import logging
import os
from typing import List, Optional

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
    MoeWithSharedWeight,
)
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.rotary_embedding.deepseek_rotary_embedding import (
    DeepseekV3RotaryEmbedding,
    DeepseekV3YarnRotaryEmbedding,
)
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeModel
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops import MlaOpsType
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    concat_0_tranpose,
    identity,
    mla_pad_t,
    stack_,
    stack_moe_w1,
    transpose,
    transpose_pad,
    transpose_slice_k,
    transpose_slice_v,
    yarn_get_mscale,
    zeros,
)


class DeepSeekV2Weight(ModelDeployWeightInfo):
    q_use_lora = False
    has_e_score_correction_bias = False

    def _process_meta(self, meta_dict, weight_keys):
        if "model.layers.0.self_attn.q_a_proj.weight" in weight_keys:
            self.q_use_lora = True
        for layer_id in range(self._num_layers):
            if (
                f"model.layers.{layer_id}.mlp.gate.e_score_correction_bias"
                in weight_keys
            ):
                self.has_e_score_correction_bias = True
                break

    def _get_hf_layer_weight_info(self, layer_id: int):
        attn_config = MlaConfig(
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
        layer_weights: List[WeightModule] = [
            AtomicWeight(
                W.pre_ln_gamma,
                [CkptWeightInfo("model.layers.{i}.input_layernorm.weight", identity)],
                identity,
            ),
            MlaAttnAtomicWeight(
                W.attn_o_w,
                [CkptWeightInfo("model.layers.{i}.self_attn.o_proj.weight", identity)],
                functools.partial(
                    mla_pad_t,
                    head_num=self._head_num,
                    nope_head_dim=self.v_head_dim,
                    rope_head_dim=0,
                ),
                config=attn_config,
            ),
            MlaAttnAtomicWeight(
                W.post_ln_gamma,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.post_attention_layernorm.weight", identity
                    )
                ],
                identity,
            ),
        ]
        mla_layer_weights: List[AtomicWeight] = [
            MlaAttnAtomicWeight(
                W.mla_kv_b_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.kv_b_proj.weight", identity
                    )
                ],
                transpose,
                config=attn_config,
            ),
            MlaAttnAtomicWeight(
                W.mla_kv_a_ln_gamma,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.kv_a_layernorm.weight", identity
                    )
                ],
                identity,
                config=attn_config,
            ),
        ]

        if self.q_use_lora:
            mla_layer_weights.extend(
                [
                    MlaAttnAtomicWeight(
                        W.mla_q_b_w,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.self_attn.q_b_proj.weight",
                                identity,
                            )
                        ],
                        transpose,
                        config=attn_config,
                    ),
                    MlaAttnAtomicWeight(
                        W.mla_q_a_ln_gamma,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.self_attn.q_a_layernorm.weight"
                            )
                        ],
                        identity,
                        config=attn_config,
                    ),
                ]
            )
            mla_layer_weights.append(
                MlaAttnAtomicWeight(
                    W.mla_fusedqkrope_w,
                    [
                        CkptWeightInfo(
                            "model.layers.{i}.self_attn.q_a_proj.weight", identity
                        ),
                        CkptWeightInfo(
                            "model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight",
                            identity,
                        ),
                    ],
                    concat_0_tranpose,
                    config=attn_config,
                )
            )
        else:
            mla_layer_weights.append(
                AtomicWeight(
                    W.mla_fusedqkrope_no_lora_w,
                    [
                        CkptWeightInfo(
                            "model.layers.{i}.self_attn.q_proj.weight", identity
                        ),
                        CkptWeightInfo(
                            "model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight",
                            identity,
                        ),
                    ],
                    concat_0_tranpose,
                    config=attn_config,
                )
            )

        # indexer weight
        if self.model_config.attn_config.is_sparse:
            mla_layer_weights.extend(
                [
                    MlaAttnAtomicWeight(
                        W.mla_indexer_qb_w,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.self_attn.indexer.wq_b.weight",
                                identity,
                            )
                        ],
                        identity,
                    ),
                    MlaAttnAtomicWeight(
                        W.mla_indexer_k_w,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.self_attn.indexer.wk.weight", identity
                            )
                        ],
                    ),
                    AtomicWeight(
                        W.mla_indexer_k_norm_w,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.self_attn.indexer.k_norm.weight",
                                identity,
                            )
                        ],
                        identity,
                    ),
                    AtomicWeight(
                        W.mla_indexer_k_norm_b,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.self_attn.indexer.k_norm.bias",
                                identity,
                            )
                        ],
                        identity,
                    ),
                    AtomicWeight(
                        W.mla_indexer_weights_proj_w,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.self_attn.indexer.weights_proj.weight",
                                identity,
                            )
                        ],
                        transpose,
                        data_type=torch.float32,
                    ),
                ]
            )

        if (
            self.model_config.attn_config.use_mla
            and self.model_config.mla_ops_type != MlaOpsType.MHA
        ):
            mla_layer_weights.append(
                MlaAttnAtomicWeight(
                    W.mla_kc,
                    [
                        CkptWeightInfo(
                            "model.layers.{i}.self_attn.kv_b_proj.weight", identity
                        )
                    ],
                    functools.partial(
                        transpose_slice_k,
                        head_num=self._head_num,
                        nope_head_dim=self.nope_head_dim,
                        v_head_dim=self.v_head_dim,
                        lora_rank=self.kv_lora_rank,
                    ),
                    config=attn_config,
                )
            )
            mla_layer_weights.append(
                MlaAttnAtomicWeight(
                    W.mla_vc,
                    [
                        CkptWeightInfo(
                            "model.layers.{i}.self_attn.kv_b_proj.weight", identity
                        )
                    ],
                    functools.partial(
                        transpose_slice_v,
                        head_num=self._head_num,
                        nope_head_dim=self.nope_head_dim,
                        v_head_dim=self.v_head_dim,
                        lora_rank=self.kv_lora_rank,
                    ),
                    config=attn_config,
                )
            )

        layer_weights.extend(mla_layer_weights)
        layer_weights.extend(self._get_hf_ffn_layer_weight_info(layer_id))
        return layer_weights

    def _get_hf_ffn_layer_weight_info(self, layer_id: int):
        align_size = self._align_size

        ffn_config = FfnConfig(
            align_size=align_size,
            is_gated_activation=self._is_gated_activation,
            is_moe=False,
        )

        if layer_id in self.moe_layer_index_:
            moe_config = MoeConfig(
                align_size=align_size,
                expert_num=self.expert_num_,
            )
            layer_weights = [
                MoeWithSharedWeight(
                    sub_weights=[
                        MoeAtomicWeight(
                            W.moe_gate,
                            [
                                CkptWeightInfo(
                                    "model.layers.{i}.mlp.gate.weight", identity
                                )
                            ],
                            transpose,
                            config=moe_config,
                        ),
                        FfnAtomicWeight(
                            W.ffn_w1,
                            [
                                CkptWeightInfo(
                                    "model.layers.{i}.mlp.shared_experts.gate_proj.weight",
                                    identity,
                                )
                            ],
                            functools.partial(
                                transpose_pad,
                                align_size=align_size,
                                dim=0,
                            ),
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
                            functools.partial(
                                transpose_pad,
                                align_size=align_size,
                                dim=1,
                            ),
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
                            functools.partial(
                                transpose_pad,
                                align_size=align_size,
                                dim=0,
                            ),
                            config=ffn_config,
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
                    ],
                    config=moe_config,
                )
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
        else:

            return [
                FfnWeight(
                    sub_weights=[
                        FfnAtomicWeight(
                            W.ffn_w1,
                            [
                                CkptWeightInfo(
                                    "model.layers.{i}.mlp.gate_proj.weight", identity
                                )
                            ],
                            functools.partial(
                                transpose_pad,
                                align_size=align_size,
                                dim=0,
                            ),
                            config=ffn_config,
                        ),
                        FfnAtomicWeight(
                            W.ffn_w2,
                            [
                                CkptWeightInfo(
                                    "model.layers.{i}.mlp.down_proj.weight", identity
                                )
                            ],
                            functools.partial(
                                transpose_pad,
                                align_size=align_size,
                                dim=1,
                            ),
                            config=ffn_config,
                        ),
                        FfnAtomicWeight(
                            W.ffn_w3,
                            [
                                CkptWeightInfo(
                                    "model.layers.{i}.mlp.up_proj.weight", identity
                                )
                            ],
                            functools.partial(
                                transpose_pad,
                                align_size=align_size,
                                dim=0,
                            ),
                            config=ffn_config,
                        ),
                    ],
                    config=ffn_config,
                ),
            ]

    def _create_rope_w(self) -> Optional[AtomicWeight]:
        if self.model_config.mla_ops_type == MlaOpsType.MHA:
            return None
        config = self.model_config

        def __create_rope_w(ts: List[torch.Tensor], config: ModelConfig):
            logging.info(
                f"initialize rope cos sin cache with seq_len: {config.max_seq_len}"
            )

            # Determine RoPE type based on whether rope_scaling exists
            # max_pos is only set when rope_scaling is present (original_max_position_embeddings)
            has_yarn_scaling = config.attn_config.rope_config.max_pos > 0

            if not has_yarn_scaling:
                # Use simple RoPE (GLM-5 style) - no rope_scaling in config.json
                logging.info("Using DeepseekV3RotaryEmbedding (simple RoPE, no YaRN)")
                rotary_emb = DeepseekV3RotaryEmbedding(
                    dim=config.attn_config.rope_config.dim,
                    max_position_embeddings=config.max_seq_len,
                    base=config.attn_config.rope_config.base,
                    device="cuda",
                )
            else:
                # Use YaRN RoPE (DeepSeek V2/V3 style) - has rope_scaling in config.json
                logging.info("Using DeepseekV3YarnRotaryEmbedding (YaRN with scaling)")
                rotary_emb = DeepseekV3YarnRotaryEmbedding(
                    config.attn_config.rope_config.dim,
                    config.max_seq_len,
                    config.attn_config.rope_config.base,
                    scaling_factor=config.attn_config.rope_config.scale,
                    original_max_position_embeddings=config.attn_config.rope_config.max_pos,
                    beta_fast=config.attn_config.rope_config.factor2,
                    beta_slow=config.attn_config.rope_config.factor1,
                    mscale=config.deepseek_rope_mscale,
                    mscale_all_dim=config.deepseek_mscale_all_dim,
                )

            # Extract cos/sin cache (same process for both types)
            half_rope_dim = config.attn_config.rope_config.dim // 2
            cos_cache = rotary_emb.cos_cached[:, :half_rope_dim]
            sin_cache = rotary_emb.sin_cached[:, :half_rope_dim]
            # cos sin cache must be float32
            cos_sin_cache = torch.cat([cos_cache, sin_cache], dim=-1).contiguous()
            return cos_sin_cache

        return AtomicWeight(
            W.rope_cos_sin_cache,
            [],
            process_fun=functools.partial(__create_rope_w, config=config),
            data_type=torch.float32,
        )

    def _get_weight_info(self):
        layer_weights: List[List[WeightModule]] = []
        weights = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo("model.embed_tokens.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo("model.norm.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_beta, [], functools.partial(zeros, shape=[self._hidden_size])
            ),
            AtomicWeight(
                W.lm_head, [CkptWeightInfo("lm_head.weight", identity)], identity
            ),
        ]
        for layer in range(self._num_layers):
            layer_weights.append(self._get_hf_layer_weight_info(layer))
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)


class DeepSeekV2(BaseModel):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = ModelConfig()
        config.attn_config.head_num = 0
        config.attn_config.kv_head_num = 0
        config.attn_config.size_per_head = 0
        config.num_layers = 0
        config.inter_size = 0
        config.vocab_size = 102400
        config.max_seq_len = 8192
        config.norm_type = "rmsnorm"
        config.has_post_decoder_layernorm = True
        # config.activation_type = "gated-silu"
        config.activation_type = "SiGLU"
        DeepSeekV2._from_hf(config, ckpt_path)
        return config

    def support_cuda_graph(self) -> bool:
        return True

    def _create_python_model(self) -> Optional[GptModelBase]:
        model_config = self.model_config
        parallelism_config = self.parallelism_config
        fmha_config = self.fmha_config
        py_hw_kernel_config = self.hw_kernel_config
        moe_config = self.moe_config
        max_generate_batch_size = self.max_generate_batch_size

        # Use GenericMoeModel with new config architecture
        # attention_type is determined from model_config.attn_config.use_mla
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

    @staticmethod
    def _from_hf(config: ModelConfig, ckpt_path: str):
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return
        with open(config_path) as reader:
            content = reader.read()
            config_json = json.loads(content)
            config.inter_size = config_json["intermediate_size"]
            config.attn_config.head_num = config_json["num_attention_heads"]
            config.attn_config.kv_head_num = config_json.get(
                "num_key_value_heads", config.attn_config.head_num
            )
            config.num_layers = config_json["num_hidden_layers"]
            # Check rope_parameters for GLM-5 style config
            rope_parameters = config_json.get("rope_parameters", {})
            rope_theta = rope_parameters.get("rope_theta")
            if rope_theta is not None:
                config.attn_config.rope_config.base = rope_theta
            else:
                config.attn_config.rope_config.base = int(
                    config_json.get("rope_theta", config.attn_config.rope_config.base)
                )
            logging.info(
                f"config.attn_config.rope_config.base: {config.attn_config.rope_config.base}"
            )
            config.vocab_size = config_json["vocab_size"]
            config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)
            config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
            config.hidden_size = config_json["hidden_size"]

            # MLA config
            config.attn_config.use_mla = True
            q_lora_rank = config_json.get("q_lora_rank")
            config.attn_config.q_lora_rank = (
                int(q_lora_rank) if q_lora_rank is not None else 0
            )
            kv_lora_rank = config_json.get("kv_lora_rank")
            config.attn_config.kv_lora_rank = (
                int(kv_lora_rank) if kv_lora_rank is not None else 0
            )
            config.attn_config.nope_head_dim = config_json["qk_nope_head_dim"]
            config.attn_config.rope_head_dim = config_json["qk_rope_head_dim"]
            config.attn_config.v_head_dim = config_json["v_head_dim"]
            config.attn_config.size_per_head = (
                config.attn_config.nope_head_dim + config.attn_config.rope_head_dim
            )
            config.attn_config.rope_config.dim = config.attn_config.rope_head_dim

            # yarn rotary config
            if config.mla_ops_type != MlaOpsType.MHA:
                config.attn_config.rope_config.style = 0
            else:
                config.attn_config.rope_config.style = 5
            # Check rope_scaling for YaRN (DeepSeek V2/V3 style)
            rope_scaling = config_json.get("rope_scaling")
            if rope_scaling is not None:
                config.attn_config.rope_config.scale = rope_scaling["factor"]
                config.attn_config.rope_config.factor1 = float(
                    rope_scaling.get("beta_slow", 1)
                )
                config.attn_config.rope_config.factor2 = float(
                    rope_scaling.get("beta_fast", 32)
                )
                config.attn_config.rope_config.max_pos = rope_scaling[
                    "original_max_position_embeddings"
                ]

                scaling_factor = rope_scaling["factor"]
                mscale = rope_scaling["mscale"]
                mscale_all_dim = rope_scaling["mscale_all_dim"]
                config.deepseek_rope_mscale = mscale
                config.deepseek_mscale_all_dim = mscale_all_dim
                config.attn_config.rope_config.mscale = yarn_get_mscale(
                    scaling_factor, mscale
                ) / yarn_get_mscale(scaling_factor, mscale_all_dim)

                # softmax scale config (only for YaRN models)
                softmax_mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                config.attn_config.softmax_extra_scale = softmax_mscale * softmax_mscale

            config.attn_config.rope_config.offset = config.attn_config.nope_head_dim
            # rope interleave config
            # default params for deepseek, override params for glm5
            rope_interleave = config_json.get("rope_interleave", True)
            config.attn_config.rope_config.is_neox_style = not rope_interleave
            indexer_rope_interleave = config_json.get("indexer_rope_interleave", False)
            config.attn_config.rope_config.indexer_is_neox_style = (
                not indexer_rope_interleave
            )

            # MOE config
            if "scoring_func" in config_json:
                scoring_func = config_json["scoring_func"]
                if scoring_func == "softmax":
                    config.scoring_func = 0
                elif scoring_func == "sigmoid":
                    config.scoring_func = 1
                else:
                    raise ValueError(f"Unknown scoring_func: {scoring_func}")

            config.routed_scaling_factor = config_json["routed_scaling_factor"]
            config.moe_k = config_json["num_experts_per_tok"]
            config.expert_num = config_json["n_routed_experts"]
            moe_intermediate_size = config_json["moe_intermediate_size"]
            config.moe_n_group = config_json.get("n_group", 1)
            config.moe_topk_group = config_json.get("topk_group", 1)

            n_shared_experts = config_json["n_shared_experts"]
            config.inter_size = n_shared_experts * moe_intermediate_size

            config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)
            config.has_moe_norm = config_json.get("norm_topk_prob", False)
            config.moe_style = 2  # shared + expert

            moe_step = config_json["moe_layer_freq"]
            first_k_dense_replace = config_json["first_k_dense_replace"]
            config.moe_layer_index = [
                i
                for i in range(config.num_layers)
                if i >= first_k_dense_replace and i % moe_step == 0
            ]

            config.config_dtype = config_json.get("torch_dtype", None)

            if config_json.get("index_topk") is not None:
                config.attn_config.is_sparse = True
                config.attn_config.indexer_head_dim = config_json["index_head_dim"]
                config.attn_config.indexer_head_num = config_json["index_n_heads"]
                config.attn_config.indexer_topk = config_json["index_topk"]

    @staticmethod
    def get_weight_cls():
        return DeepSeekV2Weight


class DeepSeekV3MtpWeight(DeepSeekV2Weight):

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config,
        hw_kernel_config,
        kv_cache_config,
        merge_lora: bool = False,
        vit_config=None,
        **kwargs,
    ):
        super().__init__(
            model_config=model_config,
            parallelism_config=parallelism_config,
            hw_kernel_config=hw_kernel_config,
            kv_cache_config=kv_cache_config,
            merge_lora=merge_lora,
            vit_config=vit_config,
            **kwargs,
        )

    def _get_weight_info(self):
        layer_weights: List[List[WeightModule]] = []
        weights = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo("model.layers.0.embed_tokens.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo("model.layers.0.shared_head.head.weight", identity)],
                identity,
            ),
        ]
        assert self._num_layers == 1
        for layer in range(self._num_layers):
            layer_weights_tmp = self._get_hf_layer_weight_info(layer)
            layer_weights_tmp.extend(
                [
                    AtomicWeight(
                        W.multi_tokens_predict_final_ln_gamma,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.shared_head.norm.weight", identity
                            )
                        ],
                        identity,
                    ),
                    AtomicWeight(
                        W.multi_tokens_predict_final_ln_beta,
                        [],
                        functools.partial(zeros, shape=[self._hidden_size]),
                    ),
                    AtomicWeight(
                        W.multi_tokens_predict_enorm,
                        [CkptWeightInfo("model.layers.{i}.enorm.weight", identity)],
                        identity,
                    ),
                    AtomicWeight(
                        W.multi_tokens_predict_hnorm,
                        [CkptWeightInfo("model.layers.{i}.hnorm.weight", identity)],
                        identity,
                    ),
                    AtomicWeight(
                        W.multi_tokens_predict_eh_proj,
                        [CkptWeightInfo("model.layers.{i}.eh_proj.weight", identity)],
                        transpose,
                    ),
                ]
            )
            layer_weights.append(layer_weights_tmp)

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)


class DeepSeekV3Mtp(DeepSeekV2):

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        config.moe_layer_index = [i for i in range(config.num_layers)]
        config.reverse_e_h_norm = True
        config.is_mtp = True
        return config

    @staticmethod
    def get_weight_cls():
        return DeepSeekV3MtpWeight


register_model("deepseek2", DeepSeekV2, ["DeepseekV2ForCausalLM"])
register_model("deepseek3", DeepSeekV2, ["DeepseekV3ForCausalLM"])
register_model("deepseek-v3-mtp", DeepSeekV3Mtp, ["DeepseekV3ForCausalLMNextN"])
register_model("kimi_k2", DeepSeekV2, [])
register_model("deepseek_v31", DeepSeekV2, [])
register_model("deepseek_v32", DeepSeekV2, ["DeepseekV32ForCausalLM"])
register_model("glm_5", DeepSeekV2, ["GlmMoeDsaForCausalLM"])
