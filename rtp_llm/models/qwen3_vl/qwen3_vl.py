import json
import os
from typing import Any, Dict, List

import torch
import torch.library as tl
from PIL import Image
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLVisionModel

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.multimodal.multimodal_mixin import (
    BaseMultiModalWeightInfo,
    BaseVitWeights,
    MultiModalMixin,
)
from rtp_llm.models.multimodal.multimodal_util import get_bytes_io_from_url
from rtp_llm.models.qwen2_5_vl.qwen2_5_vl import QWen2_5_VL, Qwen2_5_VLImageEmbedding
from rtp_llm.models.qwen2_vl.qwen2_vl_vit import MAX_PIXELS, MIN_PIXELS, smart_resize
from rtp_llm.models.qwen_v3 import QwenV3, QWenV3Weight
from rtp_llm.utils.base_model_datatypes import (
    MMPreprocessConfig,
    MMUrlType,
    MultimodalInput,
)

if not hasattr(tl, "wrap_triton"):

    def wrap_triton(fn):
        return fn

    tl.wrap_triton = wrap_triton


class Qwen3_VLImageEmbedding(Qwen2_5_VLImageEmbedding):
    def __init__(self, config: ModelConfig):
        self.mm_processor = AutoProcessor.from_pretrained(config.ckpt_path)
        config_hf = Qwen3VLConfig.from_pretrained(config.ckpt_path)
        self.visual = Qwen3VLVisionModel._from_config(config_hf.vision_config)
        self.spatial_merge_size = self.visual.spatial_merge_size
        self.data_type = config.compute_dtype

    @property
    def _device(self):
        return self.visual.device

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput],
        vit_config: VitConfig,
        processor,
    ):
        assert len(mm_inputs) == 1
        mm_input = mm_inputs[0]
        mm_type = mm_input.mm_type
        if mm_type == MMUrlType.DEFAULT or mm_type == MMUrlType.IMAGE:
            image = Image.open(get_bytes_io_from_url(mm_input.url))
            if mm_input.config.height != -1 and mm_input.config.width != -1:
                resized_height, resized_width = smart_resize(
                    mm_input.config.height,
                    mm_input.config.width,
                )
                image = image.resize((resized_width, resized_height))
            elif mm_input.config.max_pixels != -1 or mm_input.config.min_pixels != -1:
                width, height = image.size
                min_pixels = (
                    0
                    if mm_input.config.min_pixels == -1
                    else mm_input.config.min_pixels
                )
                max_pixels = (
                    0x7FFFFFFF
                    if mm_input.config.max_pixels == -1
                    else mm_input.config.max_pixels
                )
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                image = image.resize((resized_width, resized_height))
            res = processor.image_processor(image, return_tensors="pt")
            return res["pixel_values"], res["image_grid_thw"]
        elif mm_type == MMUrlType.VIDEO:
            res = processor.video_processor(mm_input.url, return_tensors="pt")
            return res["pixel_values_videos"], res["video_grid_thw"]
        else:
            raise Exception("unknown mm url type")

    def get_preprocess_params(self):
        return {
            "processor": self.mm_processor,
        }

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        pixel_values = data[0].to(self._device).to(self._data_type)
        grid_thw = data[1].to(self._device)
        embeds, deepstack_embeds = self.visual(pixel_values, grid_thw=grid_thw)
        split_sizes = (grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        embeds = torch.split(embeds, split_sizes)
        pos_id = self.get_position_ids(grid_thw)
        deepstack_embeds = torch.stack(deepstack_embeds).to(self._data_type)
        return embeds[0].to(self._data_type), pos_id, deepstack_embeds


class Qwen3VLVitWeight(BaseVitWeights):
    def _set_weight_prefix(self):
        self._ckpt_prefix = "model.visual."
        self._ft_prefix = "self.mm_part.visual."


class QWen3VLWeightInfo(QWenV3Weight, BaseMultiModalWeightInfo):
    def __init__(self, vit_weights: BaseVitWeights, **kwargs):
        QWenV3Weight.__init__(self, **kwargs)
        BaseMultiModalWeightInfo.__init__(self, vit_weights=vit_weights, **kwargs)

    def _get_weight_info(self):
        weights = self._get_hf_weight_info()
        weights = self._get_vit_info(weights)
        return weights


class QWen3_VL(QwenV3, MultiModalMixin):
    def _init_multimodal(self):
        self.mm_part = Qwen3_VLImageEmbedding(self.model_config)
        self.model_config.mm_related_params.vit_weights = Qwen3VLVitWeight(
            {"vit": self.mm_part.visual}
        )

    @classmethod
    def _get_mm_module(cls, config: ModelConfig):
        return Qwen3_VLImageEmbedding(config).visual

    @staticmethod
    def get_weight_cls():
        return QWen3VLWeightInfo

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = ModelConfig()

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
        QWen3_VL._from_config_json(config, config_json)
        return config

    @staticmethod
    def _from_config_json(config: ModelConfig, config_json: Dict[str, Any]):
        config.mm_related_params.special_tokens.update({"default_mm_token": "<img/>"})
        config.mm_model_config.mm_sep_tokens = [
            [config_json["vision_start_token_id"], config_json["vision_end_token_id"]]
        ]

        config_json = config_json["text_config"]
        config.inter_size = config_json["intermediate_size"]
        config.attn_config.head_num = config_json["num_attention_heads"]
        config.attn_config.kv_head_num = config_json.get("num_key_value_heads", config.attn_config.head_num)
        config.attn_config.size_per_head = (
            int(config_json.get("head_dim"))
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

        config.attn_config.rope_config.style = 7
        config.attn_config.rope_config.style = 7
        mrope_section = config_json["rope_scaling"].get("mrope_section", [16, 24, 24])
        config.attn_config.rope_config.index_factor = len(mrope_section)
        config.attn_config.rope_config.mrope_dim1 = mrope_section[0]
        config.attn_config.rope_config.mrope_dim2 = mrope_section[1]
        config.attn_config.rope_config.mrope_dim3 = mrope_section[2]
        config.mm_model_config.mm_position_ids_style = 2


register_model("qwen3_vl", QWen3_VL, ["Qwen3VLForConditionalGeneration"])
