import functools
import json
import logging
import os
from typing import Any, Dict, List

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.mrope_utils import apply_mrope_section
from rtp_llm.models.qwen_v3 import QwenV3, QWenV3Weight
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    WeightStyle,
    identity,
    merge_qkv_hf,
    transpose,
)

# 9-char plain-text media placeholders emitted by the MOSS-VL chat template
# (angle-bracket + '|image|' / '|video|'). The processor, not the tokenizer,
# replaces them with real media tokens, so renderer/processor wiring must
# reproduce these strings verbatim; assembled by concatenation to keep the
# source free of tokenizer-adjacent magic strings.
MOSS_VL_IMAGE_PLACEHOLDER = "<" + "|image" + "|>"
MOSS_VL_VIDEO_PLACEHOLDER = "<" + "|video" + "|>"


class MossVLWeight(QWenV3Weight):
    """Weight descriptor for MOSS-VL: 36 self-attn layers + 12 cross-attn layers.

    Cross-attn layers replace self-attention entirely (Flamingo-style): they
    carry cross_attn.{q,k,v,o}_proj + q/k norm + two tanh gates, plus the usual
    input/post layernorms and gated MLP. Self-attn layers follow the Qwen
    template unchanged. The vision tower (model.visual.*) and
    model.separator_token are loaded by the multimodal mixin, not here.
    """

    def __init__(self, *args, **kwargs):
        self.cross_attention_layers: List[int] = []
        super().__init__(*args, **kwargs)

    def _process_meta(self, meta_dicts: Any, weight_keys: List[str]):
        super()._process_meta(meta_dicts, weight_keys)
        self.cross_attention_layers = list(
            self.model_config.mm_related_params.config.get("cross_attention_layers", [])
        )
        if self.cross_attention_layers:
            logging.info(
                f"moss_vl cross attention layer indices: {self.cross_attention_layers}"
            )

    def _get_cross_attn_layer_weight_info(self, layer_id: int) -> List[WeightModule]:
        attn_config = AttnConfig(
            hidden_size=self._hidden_size,
            size_per_head=self._size_per_head,
            head_num=self._head_num,
            head_num_kv=self._head_num_kv,
        )

        layer_weights = [
            AtomicWeight(
                W.pre_ln_gamma,
                [
                    CkptWeightInfo(
                        self.transformer_prefix + "layers.{i}.input_layernorm.weight",
                        identity,
                    )
                ],
                identity,
            ),
            # cross_attn.{q,k,v}_proj share GQA geometry with self-attn
            # (32Q/8KV x 128), so the merged-QKV contract is identical.
            AttnAtomicWeight(
                W.attn_qkv_w,
                [
                    CkptWeightInfo(
                        self.transformer_prefix + "layers.{i}.cross_attn.q_proj.weight",
                        identity,
                    ),
                    CkptWeightInfo(
                        self.transformer_prefix + "layers.{i}.cross_attn.k_proj.weight",
                        identity,
                    ),
                    CkptWeightInfo(
                        self.transformer_prefix + "layers.{i}.cross_attn.v_proj.weight",
                        identity,
                    ),
                ],
                functools.partial(merge_qkv_hf),
                config=attn_config,
            ),
            AttnAtomicWeight(
                W.attn_o_w,
                [
                    CkptWeightInfo(
                        self.transformer_prefix + "layers.{i}.cross_attn.o_proj.weight",
                        identity,
                    )
                ],
                transpose,
                config=attn_config,
            ),
            AtomicWeight(
                W.post_ln_gamma,
                [
                    CkptWeightInfo(
                        self.transformer_prefix
                        + "layers.{i}.post_attention_layernorm.weight",
                        identity,
                    )
                ],
                identity,
                config=attn_config,
            ),
        ]

        if self._use_qk_norm:
            layer_weights.extend(
                [
                    AttnAtomicWeight(
                        W.q_ln_gamma,
                        [
                            CkptWeightInfo(
                                self.transformer_prefix
                                + "layers.{i}.cross_attn.q_norm.weight"
                            )
                        ],
                        config=attn_config,
                    ),
                    AttnAtomicWeight(
                        W.k_ln_gamma,
                        [
                            CkptWeightInfo(
                                self.transformer_prefix
                                + "layers.{i}.cross_attn.k_norm.weight"
                            )
                        ],
                        config=attn_config,
                    ),
                ]
            )

        # tanh-gated residual scalars (checkpoint keys cross_attn_attn_gate /
        # cross_attn_mlp_gate, shape [1] BF16).
        layer_weights.extend(
            [
                AtomicWeight(
                    W.cross_attn_attn_gate,
                    [
                        CkptWeightInfo(
                            self.transformer_prefix + "layers.{i}.cross_attn_attn_gate",
                            identity,
                        )
                    ],
                    identity,
                    config=attn_config,
                ),
                AtomicWeight(
                    W.cross_attn_mlp_gate,
                    [
                        CkptWeightInfo(
                            self.transformer_prefix + "layers.{i}.cross_attn_mlp_gate",
                            identity,
                        )
                    ],
                    identity,
                    config=attn_config,
                ),
            ]
        )

        layer_weights.extend(self._get_hf_ffn_layer_weight_info(layer_id))
        return layer_weights

    def _get_hf_weight_info(self):
        if self.weight_style != WeightStyle.NONE:
            raise NotImplementedError(
                f"moss_vl only supports HF-style checkpoints, got {self.weight_style}"
            )
        weights = [
            AtomicWeight(
                W.embedding,
                [
                    CkptWeightInfo(
                        self.transformer_prefix + "embed_tokens.weight", identity
                    )
                ],
                identity,
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo("lm_head.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo(self.transformer_prefix + "norm.weight", identity)],
                identity,
            ),
        ]

        layer_weights: List[List[WeightModule]] = []
        for layer in range(self._num_layers):
            if layer in self.cross_attention_layers:
                w = self._get_cross_attn_layer_weight_info(layer)
            else:
                w = self._get_hf_layer_weight_info(layer)
            layer_weights.append(w)

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)


class MossVL(QwenV3):
    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.mossvl import MossVLModel

        self.py_model = MossVLModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            quant_config=self.model_config.quant_config,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )

    @staticmethod
    def get_weight_cls():
        return MossVLWeight

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.attn_config.rope_config.dim = 128
        config.attn_config.rope_config.style = 1
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
        MossVL._from_config_json(config, config_json)
        return config

    @staticmethod
    def _from_config_json(config: ModelConfig, config_json: Dict[str, Any]):
        # MOSS-VL keeps media token ids at the config top level and nests the
        # LLM under text_config / the vision tower under vision_config. Capture
        # everything top-level before rebinding to text_config.
        vision_start_token_id = config_json["vision_start_token_id"]
        vision_end_token_id = config_json["vision_end_token_id"]
        image_token_id = config_json["image_token_id"]
        video_token_id = config_json["video_token_id"]

        config.mm_related_params.special_tokens.update(
            {"default_mm_token": MOSS_VL_IMAGE_PLACEHOLDER}
        )
        config.mm_model_config.mm_sep_tokens = [
            [vision_start_token_id, vision_end_token_id]
        ]
        # marker-mode metadata consumed by the multimodal mixin / engine:
        # images expand to ONE image token (no sep wrapper); videos wrap in
        # vision_start/end with one image token + timestamp text per frame.
        mm_config = config.mm_related_params.config
        mm_config["ckpt_path"] = config.ckpt_path
        mm_config["image_token_id"] = image_token_id
        mm_config["video_token_id"] = video_token_id
        mm_config["vision_start_token_id"] = vision_start_token_id
        mm_config["vision_end_token_id"] = vision_end_token_id
        mm_config["vision_seq_pad_multiple"] = config_json.get(
            "vision_seq_pad_multiple", 1
        )
        mm_config["vision_config"] = config_json.get("vision_config", {})
        mm_config["image_placeholder"] = MOSS_VL_IMAGE_PLACEHOLDER
        mm_config["video_placeholder"] = MOSS_VL_VIDEO_PLACEHOLDER

        config_json = config_json["text_config"]
        mm_config["cross_attention_layers"] = list(
            config_json.get("cross_attention_layers", [])
        )
        config.inter_size = config_json["intermediate_size"]
        config.attn_config.head_num = config_json["num_attention_heads"]
        config.attn_config.kv_head_num = config_json.get(
            "num_key_value_heads", config.attn_config.head_num
        )
        config.attn_config.size_per_head = (
            int(config_json["head_dim"])
            if "head_dim" in config_json
            else config_json["hidden_size"] // config.attn_config.head_num
        )
        if config_json.get("hidden_size") is not None:
            config.hidden_size = config_json["hidden_size"]
        config.num_layers = config_json["num_hidden_layers"]
        config.attn_config.rope_config.base = config_json.get(
            "rope_theta", config.attn_config.rope_config.base
        )
        config.vocab_size = config_json["vocab_size"]
        config.attn_config.rope_config.dim = config.attn_config.size_per_head
        config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
        config.config_dtype = config_json.get("torch_dtype", None)
        if config_json.get("max_position_embeddings"):
            config.max_seq_len = config_json["max_position_embeddings"]

        config.attn_config.rope_config.style = 7
        # MOSS-VL interleaves T/H/W rotary pairs with the same 128-dim layout
        # as Qwen3-VL (section [24, 20, 20], 64 pairs total).
        rope_scaling = config_json.get("rope_scaling", {})
        if "mrope_section" not in rope_scaling:
            logging.warning(
                "MOSS-VL config does not specify mrope_section; using the "
                "upstream 128-dim fallback [24, 20, 20]"
            )
        mrope_section = rope_scaling.get("mrope_section", [24, 20, 20])
        apply_mrope_section(
            config.attn_config.rope_config,
            mrope_section,
            model_name="MOSS-VL",
            interleaved=rope_scaling.get("mrope_interleaved", True),
        )
        # MARKER_MROPE (3): one image-token marker per frame; text positions
        # advance max(eh, ew)+1 per marker (PositionIdsGenerator marker branch).
        config.mm_model_config.mm_position_ids_style = 3
        config.mm_model_config.mm_marker_token = image_token_id
        config.mm_model_config.is_multimodal = True


register_model("moss_vl", MossVL, ["MossVLForConditionalGeneration"])
