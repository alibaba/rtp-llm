import functools
import json
import os
from typing import Any, Dict, List

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, FfnConfig, FfnWeight
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.multimodal.multimodal_mixin import (
    BaseMultiModalWeightInfo,
    BaseVitWeights,
    MultiModalMixin,
)
from rtp_llm.models.qwen2_vl.qwen2_vl_vit import Qwen2VLImageEmbedding
from rtp_llm.models.qwen_vl import QWen_VL
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    merge_qkv_b,
    merge_qkv_hf,
    merge_qkv_lora_A,
    merge_qkv_lora_B,
    sp_0,
    sp_head_lora,
    sp_id,
    sp_neg1,
    transpose,
    transpose_pad,
    zeros,
)


class QwenVL2VitWeight(BaseVitWeights):
    def _set_weight_prefix(self):
        self._ckpt_prefix = "visual."
        self._ft_prefix = "self.mm_part.visual."


class QWen2VLWeightInfo(ModelDeployWeightInfo, BaseMultiModalWeightInfo):
    def __init__(self, vit_weights, **kwargs):
        ModelDeployWeightInfo.__init__(self, **kwargs)
        BaseMultiModalWeightInfo.__init__(self, vit_weights=vit_weights, **kwargs)

    @property
    def support_lora(self) -> bool:
        return True

    def _get_weight_info(self):
        weights = self._get_hf_weight_info()
        return weights

    def _get_hf_weight_info(self):
        align_size = self._align_size
        weights = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo("model.embed_tokens.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.lm_head, [CkptWeightInfo("lm_head.weight", identity)], identity
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo("model.norm.weight", identity)],
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

    def _get_hf_layer_weight_info(self, layer_id):
        align_size = self._align_size
        attn_config: AttnConfig = self.attn_config
        ffn_config: FfnConfig = self.ffn_config
        layer_weights = [
            AtomicWeight(
                W.pre_ln_gamma,
                [CkptWeightInfo("model.layers.{i}.input_layernorm.weight", identity)],
                identity,
            ),
            AttnAtomicWeight(
                W.attn_qkv_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.q_proj.weight", identity
                    ),
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.k_proj.weight", identity
                    ),
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.v_proj.weight", identity
                    ),
                ],
                functools.partial(merge_qkv_hf),
                config=attn_config,
                lora_a_process_func=functools.partial(
                    merge_qkv_lora_A,
                    allow_empty=False,
                    hidden_size=self._hidden_size,
                    head_num=self._head_num,
                    head_num_kv=self._head_num_kv,
                    size_per_head=self._size_per_head,
                ),
                lora_b_process_func=functools.partial(
                    merge_qkv_lora_B,
                    allow_empty=False,
                    hidden_size=self._hidden_size,
                    head_num=self._head_num,
                    head_num_kv=self._head_num_kv,
                    size_per_head=self._size_per_head,
                ),
                lora_a_split_func=sp_id,
                lora_b_split_func=sp_head_lora,
            ),
            AttnAtomicWeight(
                W.attn_qkv_b,
                [
                    CkptWeightInfo("model.layers.{i}.self_attn.q_proj.bias", identity),
                    CkptWeightInfo("model.layers.{i}.self_attn.k_proj.bias", identity),
                    CkptWeightInfo("model.layers.{i}.self_attn.v_proj.bias", identity),
                ],
                functools.partial(merge_qkv_b),
                config=attn_config,
                lora_a_process_func=transpose,
                lora_b_process_func=transpose,
                lora_a_split_func=sp_0,
                lora_b_split_func=sp_id,
            ),
            AttnAtomicWeight(
                W.attn_o_w,
                [CkptWeightInfo("model.layers.{i}.self_attn.o_proj.weight", identity)],
                transpose,
                config=attn_config,
            ),
            FfnWeight(
                sub_weights=[
                    FfnAtomicWeight(
                        W.ffn_w1,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.gate_proj.weight", identity
                            )
                        ],
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
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.up_proj.weight", identity
                            )
                        ],
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
                                "model.layers.{i}.mlp.down_proj.weight", identity
                            )
                        ],
                        functools.partial(transpose_pad, align_size=align_size, dim=1),
                        config=ffn_config,
                        lora_a_process_func=functools.partial(
                            transpose_pad, align_size=align_size, dim=1
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
                [
                    CkptWeightInfo(
                        "model.layers.{i}.post_attention_layernorm.weight", identity
                    )
                ],
                identity,
            ),
        ]
        return layer_weights


class QWen2_VL(QWen_VL, MultiModalMixin):
    def _init_multimodal(
        self,
        mm_model_config,
        vit_config: VitConfig,
    ):
        # mm_related_params is in model_config, not mm_model_config
        self.mm_part = Qwen2VLImageEmbedding(
            self.model_config.mm_related_params, model_config=self.model_config
        )
        self.model_config.mm_related_params.vit_weights = QwenVL2VitWeight(
            {"vit": self.mm_part.visual}
        )

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = ModelConfig()
        config.ckpt_path = ckpt_path

        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return config
        with open(config_path) as reader:
            content = reader.read()
            config_json = json.loads(content)

        QWen2_VL._from_hf(config, config_json)
        QWen2_VL._load_vit_param(config, config_json)
        config.mm_related_params.config["ckpt_path"] = ckpt_path

        return config

    @staticmethod
    def _load_vit_param(config: ModelConfig, config_json: Dict[str, Any]):
        config.mm_related_params.config = config_json["vision_config"]
        if config.mm_related_params.special_tokens is None:
            from rtp_llm.config.model_config import SpecialTokens

            config.mm_related_params.special_tokens = SpecialTokens()
        config.mm_related_params.special_tokens.update({"default_mm_token": "<img/>"})
        config.mm_related_params.eval_param_count = QWen2_VL.eval_vit_param_count
        config.mm_related_params.eval_model_size = QWen2_VL.eval_vit_model_size
        config.mm_model_config.mm_sep_tokens = [
            [config_json["vision_start_token_id"], config_json["vision_end_token_id"]]
        ]

    @staticmethod
    def _from_hf(config: ModelConfig, config_json: Dict[str, Any]):
        config.vocab_size = config_json["vocab_size"]
        config.max_seq_len = 10240
        config.activation_type = "SiGLU"
        config.attn_config.head_num = config_json["num_attention_heads"]
        config.attn_config.kv_head_num = config_json["num_key_value_heads"]
        config.hidden_size = config_json["hidden_size"]
        config.attn_config.size_per_head = (
            int(config_json.get("head_dim"))
            if "head_dim" in config_json
            else config_json["hidden_size"] // config.attn_config.head_num
        )
        config.num_layers = config_json["num_hidden_layers"]
        config.inter_size = config_json["intermediate_size"]
        config.norm_type = "rmsnorm"
        config.layernorm_eps = config_json["rms_norm_eps"]
        config.has_post_decoder_layernorm = True
        config.special_tokens.bos_token_id = config_json.get("bos_token_id", -1)
        config.special_tokens.eos_token_id = config_json.get("eos_token_id", 0)
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
        config.mm_model_config.mm_position_ids_style = 2
        rope_config = config.attn_config.rope_config
        rope_config.style = 7
        rope_config.base = int(config_json["rope_theta"])
        # mrope_section is not available in RopeConfig, using default value
        mrope_section = config_json["rope_scaling"].get("mrope_section", [16, 24, 24])
        rope_config.index_factor = len(mrope_section)
        rope_config.mrope_dim1 = mrope_section[0]
        rope_config.mrope_dim2 = mrope_section[1]
        rope_config.mrope_dim3 = mrope_section[2]
        rope_config.dim = 128

    @staticmethod
    def get_weight_cls():
        return QWen2VLWeightInfo

    @classmethod
    def eval_vit_model_size(cls, mm_related_params):
        data_width = 4
        return QWen2_VL.eval_vit_param_count(mm_related_params) * data_width

    @classmethod
    def eval_vit_param_count(cls, mm_related_params):
        vit_config = mm_related_params.config
        embed_dim = vit_config.get("embed_dim", 1280)
        hidden_size = vit_config.get("hidden_size", 3584)
        vit_size = (
            vit_config.get("temporal_patch_size", 2)
            * vit_config.get("spatial_patch_size", 14) ** 2
            * vit_config.get("in_chans", 3)
            * embed_dim
        )
        patch_merger_size = embed_dim * vit_config.get("spatial_merge_size", 2) ** 2
        vit_size += patch_merger_size**2 + patch_merger_size * hidden_size + embed_dim
        mlp_hidden_dim = embed_dim * vit_config.get("mlp_ratio", 4)
        vit_size += vit_config.get("depth", 32) * (
            embed_dim * 2 + embed_dim * mlp_hidden_dim + embed_dim * embed_dim * 4
        )

        return vit_size


register_model("qwen2_vl", QWen2_VL, ["Qwen2VLForConditionalGeneration"])
