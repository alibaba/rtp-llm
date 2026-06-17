"""MiniMax-M3 VL (vision-language) model registration for rtp-llm.

Subclasses the text-only :class:`MiniMaxM3` backbone and layers on the
multi-modal configuration parsed from the top level of the VL checkpoint's
``config.json``. The LLM half is fully reused via inheritance — the
checkpoint stores LLM tensors under ``language_model.*`` which the
:class:`MiniMaxM3Weight` loader already handles.

What this file is responsible for:

* Marking the model as multimodal (``mm_model_config.is_multimodal``).
* Wiring the VISION_START / VISION_END token IDs that bracket the image
  placeholder span in input ids — used by rtp-llm's multimodal pipeline to
  splice ViT features into the LLM embedding stream.
* Stuffing the vision-side ckpt parameters (vision_config,
  projector_hidden_size, multimodal_projector_bias,
  img_token_compression_config, projector_hidden_act, image/video token
  indices) into ``mm_related_params.config`` so the future VL mixin /
  multimodal model description can build the ViT + projector.
* Delegating all text-side parsing to ``MiniMaxM3._from_text_config`` so
  there is exactly one source of truth for the LLM config.

Notes:
* ``mm_position_ids_style = 0`` — unlike Qwen2/3-VL which use M-RoPE (style
  2) with a 3D position layout for image tokens, MiniMax-M3 keeps 1D
  position ids for the LLM; the 3D spatial/temporal RoPE is applied
  *inside* the vision tower itself (see ``modeling_minimax_m3_vit.py``
  in the checkpoint).
* The HF arch ``MiniMaxM3SparseForConditionalGeneration`` is registered
  here for the VL model. The text-only ``minimax_m3`` entry registers the
  same arch — at wire-up time the auto-arch map is expected to resolve to
  the VL entry when a ``vision_config`` is present, otherwise to the text
  entry.
"""

import json
import os

from rtp_llm.model_factory_register import register_model
from rtp_llm.models.minimax_m3 import MiniMaxM3, MiniMaxM3Weight


class MiniMaxM3_VL(MiniMaxM3):
    """MiniMax-M3 VL container — text backbone + (future) ViT + projector."""

    @classmethod
    def _create_config(cls, ckpt_path):
        # Build the base text config first (this invokes our overridden
        # _from_hf, which handles both text + multimodal parsing).
        return super()._create_config(ckpt_path)

    def _create_python_model(self):
        """Wire MiniMax-M3 VL into the Python forward path.

        Reuses MiniMaxM3's text-side wiring (MSA sparse attention, FP8 fusion,
        Gemma-norm-fused QK, SwiGLU-OAI) by going through MultimodalGenericModel,
        which is a strict superset of GenericMoeModel (the desc that MiniMaxM3
        inherits via DeepSeekV2). All M3-specific behaviour is driven by
        ``model_config`` fields (``msa_sparse_config``, ``swiglu_alpha/limit``,
        ``qk_norm``, ``quant_config``) and the per-layer weight dict, so no
        further override is required here. The only delta over the text-only
        path is ``MultimodalEmbeddingInjector``, which splices ViT features
        into the embedding stream at the VISION_START/END placeholder spans
        before the decoder stack runs.
        """
        from rtp_llm.models_py.model_desc.multimodal_generic import (
            MultimodalGenericModel,
        )

        self.py_model = MultimodalGenericModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )

    @classmethod
    def _from_hf(cls, config, ckpt_path):
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return
        with open(config_path) as reader:
            config_json = json.loads(reader.read())

        # ----- Multi-modal flag -----
        config.mm_model_config.is_multimodal = True

        # ----- Image / video placeholder token IDs -----
        # MiniMax-M3's chat_template (see Minimax-M3-preview/chat_template.jinja
        # lines 10-11, 50-53, 222-225) emits a SINGLE token per multimodal
        # content part:
        #   image -> ']<]image[>[' (id 200025)
        #   video -> ']<]video[>[' (id 200026)
        # It does NOT emit the bracket tokens 200029/200030 (start/end of image)
        # or 200031/200032 (start/end of video) — those exist in the tokenizer
        # but the reference template never writes them.
        #
        # rtp-llm's C++ multimodal pipeline supports single-token mode (see
        # MultimodalProcessor.cc::getMultimodalTags, sep.size()==1 branch):
        # each occurrence is one slot, replaced in expandTokenIds() by N
        # feature rows from the mixin's embedding() return tensor.
        image_token_id = config_json.get("image_token_index", 200025)
        video_token_id = config_json.get("video_token_index", 200026)
        config.mm_model_config.mm_sep_tokens = [[image_token_id], [video_token_id]]

        # ----- Position-id style -----
        # No M-RoPE on the LLM side for M3; the ViT internally does 3D RoPE
        # while the LLM consumes 1D positions for image tokens.
        config.mm_model_config.mm_position_ids_style = 0

        # ----- Vision-side params (consumed by the VL mixin / py model) -----
        # Stash the full vision_config + projector knobs in the per-VitParam
        # config dict so downstream code can build the ViT without re-reading
        # config.json.
        mm_cfg = config.mm_related_params.config
        mm_cfg["ckpt_path"] = ckpt_path
        mm_cfg["vision_config"] = config_json.get("vision_config", {})
        mm_cfg["image_token_index"] = config_json.get("image_token_index", 200025)
        mm_cfg["video_token_index"] = config_json.get("video_token_index", 200026)
        mm_cfg["projector_hidden_size"] = config_json.get("projector_hidden_size")
        mm_cfg["multimodal_projector_bias"] = config_json.get(
            "multimodal_projector_bias", True
        )
        mm_cfg["img_token_compression_config"] = config_json.get(
            "img_token_compression_config", {}
        )
        mm_cfg["projector_hidden_act"] = config_json.get("projector_hidden_act", "gelu")
        mm_cfg["vision_feature_layer"] = config_json.get("vision_feature_layer", -1)
        mm_cfg["vision_feature_select_strategy"] = config_json.get(
            "vision_feature_select_strategy", "full"
        )

        # Placeholder string the user puts inline in the prompt — the
        # multimodal pipeline replaces this with the actual URL/path span
        # before tokenization, then re-expands to image tokens at the
        # VISION_START / VISION_END brackets.
        config.mm_related_params.special_tokens["default_mm_token"] = "]<]image[>["

        # ----- Text-side parsing -----
        # Delegate to the base class's single source of truth.
        text_cfg = config_json.get("text_config", config_json)
        MiniMaxM3._from_text_config(config, text_cfg)

    @staticmethod
    def get_weight_cls():
        # LLM weights live under language_model.* which MiniMaxM3Weight
        # already handles via its ``self.prefix = "language_model."`` setting.
        return MiniMaxM3Weight


# ----------------------------------------------------------------------
# Registration
#   - "minimax_m3_vl" : forced name
#   - HF arch         : MiniMaxM3SparseForConditionalGeneration (the VL
#                       container — text-only minimax_m3 also claims this
#                       arch; the auto-arch resolver picks the VL entry
#                       when vision_config is present).
# ----------------------------------------------------------------------
register_model(
    "minimax_m3_vl",
    MiniMaxM3_VL,
    ["MiniMaxM3SparseForConditionalGeneration"],
    [],
)
