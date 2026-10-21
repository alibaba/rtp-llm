"""MOSS-VL multimodal mixin: processor + vendored vision tower + weights.

The MOSS-VL contract differs from the Qwen-VL family: vision tokens never
enter the text stream. Each media contributes ONE image-token marker per frame
to input_ids (no per-patch expansion), and the merged vision features (with
one learnable separator per frame) flow to the LLM as cross-attention K/V
source. embedding()/batched_embedding() therefore return:

  - features: [rows_including_separators, out_hidden_size] per media
  - position ids: [num_frames, 3] int32 rows of (t=0, eh, ew) — the merged
    grid geometry per frame consumed by marker-mode position generation
  - extra input: the same geometry flattened as one opaque 1-D int64 tensor
    (deepstack-shaped slicing does not apply; see D2 in the RUN plan)
"""

import logging
from typing import Any, List, Tuple

import torch
from PIL import Image
from transformers import AutoImageProcessor

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.metrics.kmonitor_metric_reporter import GaugeMetrics
from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseMultiModalDeployWeightInfo,
    BaseMultiModalMixin,
    BaseVitWeights,
    VitParameters,
)
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MultiModalEmbeddingInterface,
)
from rtp_llm.multimodal.multimodal_mixins.moss_vl.moss_vl_vision import (
    MOSS_VL_DEFAULT_VISION_CONFIG,
    MossVLVisionConfig,
    MossVLVisionModel,
)
from rtp_llm.multimodal.multimodal_mixins.qwen2_5_vl.qwen2_5_vl_mixin import smart_resize
from rtp_llm.multimodal.multimodal_util import get_bytes_io_from_url
from rtp_llm.multimodal.vit_metrics import (
    record_vit_preprocess_value,
    vit_preprocess_timer,
)
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType
from rtp_llm.utils.flash_attn_utils import can_use_flash_attn
from rtp_llm.utils.model_weight import CkptWeightInfo, identity, sp_id
from rtp_llm.model_loader.weight_module import CustomAtomicWeight
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo

default_attn_impl = "sdpa"
try:
    if can_use_flash_attn():
        default_attn_impl = "flash_attention_2"
except Exception as e:
    logging.info(
        f"initialize flash_attn failed, exception {e}, using sdpa attention in moss_vl vit"
    )

# Reference preprocessor defaults (preprocessor_config.json): dynamic
# resolution with a 4096-pixel lower bound per image and a per-image cap.
MOSS_VL_MIN_PIXELS = 4096
MOSS_VL_MAX_PIXELS = 16777216


class MossVLImageEmbedding(MultiModalEmbeddingInterface):
    def __init__(self, mm_related_params: VitParameters):
        vision_config = mm_related_params.config.get(
            "vision_config", MOSS_VL_DEFAULT_VISION_CONFIG
        )
        self.mm_config = MossVLVisionConfig(vision_config)
        self.mm_processor = AutoImageProcessor.from_pretrained(
            mm_related_params.config["ckpt_path"], trust_remote_code=True
        )
        self.visual = MossVLVisionModel(self.mm_config, attn_implementation=default_attn_impl)
        self.spatial_merge_size = self.mm_config.spatial_merge_size
        self.patch_size = self.mm_config.patch_size
        self.vision_seq_pad_multiple = mm_related_params.config.get(
            "vision_seq_pad_multiple", 1
        )

    @property
    def _data_type(self):
        return self.visual.dtype

    @property
    def _device(self):
        return self.visual.device

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput],
        vit_config: VitConfig,
        processor,
        factor: int = 32,
    ):
        assert len(mm_inputs) == 1
        mm_input = mm_inputs[0]
        mm_type = mm_input.mm_type
        if mm_type != MMUrlType.DEFAULT and mm_type != MMUrlType.IMAGE:
            raise Exception("moss_vl video preprocessing is not wired yet")
        tags = {"model": "moss_vl", "mm_type": "image"}
        with vit_preprocess_timer(GaugeMetrics.VIT_IMAGE_FETCH_RT_US_METRIC, tags):
            image_data = get_bytes_io_from_url(
                mm_input.url,
                vit_config.download_headers,
                max_file_size_kb=vit_config.mm_image_max_file_size_kb,
            )
        with vit_preprocess_timer(GaugeMetrics.VIT_IMAGE_DECODE_RT_US_METRIC, tags):
            image = Image.open(image_data)
        if image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size
        min_pixels = MOSS_VL_MIN_PIXELS
        max_pixels = MOSS_VL_MAX_PIXELS
        if mm_input.mm_preprocess_config.min_pixels != -1:
            min_pixels = mm_input.mm_preprocess_config.min_pixels
        if mm_input.mm_preprocess_config.max_pixels != -1:
            max_pixels = mm_input.mm_preprocess_config.max_pixels
        resized_height, resized_width = smart_resize(
            height, width, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels
        )
        with vit_preprocess_timer(GaugeMetrics.VIT_IMAGE_RESIZE_RT_US_METRIC, tags):
            image = image.resize((resized_width, resized_height))
        record_vit_preprocess_value(
            GaugeMetrics.VIT_RESIZED_PIXEL_COUNT_METRIC,
            resized_width * resized_height,
            tags,
        )
        with vit_preprocess_timer(GaugeMetrics.VIT_IMAGE_PROCESSOR_RT_US_METRIC, tags):
            res = processor(image, return_tensors="pt")
        return res["pixel_values"], res["image_grid_thw"]

    def get_preprocess_params(self):
        return {
            "processor": self.mm_processor,
            "factor": self.spatial_merge_size * self.patch_size,
        }

    def _frame_geometry(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """[num_frames, 3] int32 rows of (t=0, eh, ew) per frame of one media."""
        merge = self.spatial_merge_size
        rows = []
        for t, h, w in grid_thw.tolist():
            rows.extend([[0, h // merge, w // merge]] * int(t))
        return torch.tensor(rows, dtype=torch.int32, device=self._device)

    def _geometry_extra(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Flat [num_frames, eh, ew] per media (3 int64 values)."""
        merge = self.spatial_merge_size
        t, h, w = grid_thw[0].tolist()
        return torch.tensor(
            [int(t), h // merge, w // merge], dtype=torch.int64, device=self._device
        )

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        pixel_values = data[0].to(self._device).to(self._data_type)
        grid_thw = data[1].to(self._device)
        merged = self.visual(pixel_values, grid_thw=grid_thw)
        sequences = self.visual.vision_sequence_per_media(
            merged, grid_thw, self.vision_seq_pad_multiple
        )
        pos_id = self._frame_geometry(grid_thw)
        # One opaque flat 1-D geometry tensor per media for the language model
        # desc; NOT deepstack-shaped, do not deepstack-slice.
        extra_input = self._geometry_extra(grid_thw)
        return sequences[0].to(self._data_type), pos_id, extra_input

    @torch.inference_mode()
    def batched_embedding(
        self, data_list: List[Any], mm_types: List[MMUrlType], **kwargs
    ):
        res_list = []
        pixel_values_list = []
        grid_thw_list = []
        for data in data_list:
            pixel_values_list.append(data[0])
            grid_thw_list.append(data[1])
        pixel_values = (
            torch.concat(pixel_values_list, dim=0).to(self._device).to(self._data_type)
        )
        grid_thw = torch.concat(grid_thw_list, dim=0).to(self._device)
        merged = self.visual(pixel_values, grid_thw=grid_thw)
        sequences = self.visual.vision_sequence_per_media(
            merged, grid_thw, self.vision_seq_pad_multiple
        )
        pos_ids = self._frame_geometry_per_media(grid_thw)
        for media_idx, (seq, pos_id) in enumerate(zip(sequences, pos_ids)):
            extra_input = torch.tensor(
                [
                    int(grid_thw[media_idx][0]),
                    int(grid_thw[media_idx][1]) // self.spatial_merge_size,
                    int(grid_thw[media_idx][2]) // self.spatial_merge_size,
                ],
                dtype=torch.int64,
                device=self._device,
            )
            res_list.append((seq.to(self._data_type), pos_id, extra_input))
        return res_list

    def _frame_geometry_per_media(self, grid_thw: torch.Tensor) -> List[torch.Tensor]:
        merge = self.spatial_merge_size
        result = []
        for t, h, w in grid_thw.tolist():
            frames = int(t)
            eh, ew = h // merge, w // merge
            rows = torch.tensor(
                [[0, eh, ew]] * frames, dtype=torch.int32, device=self._device
            )
            result.append(rows)
        return result


class MossVLVitWeight(BaseVitWeights):
    def _set_weight_prefix(self):
        # All vision weights live under model.visual.*, and the separator
        # token (an attribute of the vendored visual module) under
        # model.separator_token; the custom weight-info below fixes that one
        # non-uniform mapping.
        self._ckpt_prefix = "model.visual."
        self._ft_prefix = "self.mm_part.visual."


class MossVLMultiModalDeployWeightInfo(BaseMultiModalDeployWeightInfo):
    # separator_token sits at checkpoint key model.separator_token while the
    # module attribute path is visual.separator_token (see MossVLVitWeight).
    SEPARATOR_CKPT_NAME = "model.separator_token"
    SEPARATOR_WEIGHT_NAME = "separator_token"

    def get_weight_info(self):
        weights = []
        weight_names = self.vit_weights.weight_names
        ckpt_prefix = self.vit_weights.ckpt_prefix
        for w in weight_names:
            ckpt_name = (
                self.SEPARATOR_CKPT_NAME
                if w == self.SEPARATOR_WEIGHT_NAME
                else ckpt_prefix + w
            )
            weights.append(
                CustomAtomicWeight(w, [CkptWeightInfo(ckpt_name, identity)], identity)
            )
        return ModelWeightInfo(layer_weights=[], weights=weights)


class MossVLMixin(BaseMultiModalMixin):
    def _init_multimodal(self):
        self.mm_part = MossVLImageEmbedding(self.mm_related_params)
        self.mm_related_params.vit_weights = MossVLVitWeight(
            {"vit": self.mm_part.visual}
        )

    @classmethod
    def get_multimodal_mixin_weight_info(cls):
        return MossVLMultiModalDeployWeightInfo

    @classmethod
    def _get_mm_module(cls, mm_related_params: VitParameters, vit_config: VitConfig):
        return MossVLImageEmbedding(mm_related_params).visual


register_multimodal_mixin(["moss_vl"], MossVLMixin)
