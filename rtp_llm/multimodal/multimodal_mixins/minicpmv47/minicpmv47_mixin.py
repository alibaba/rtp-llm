"""MiniCPM-V-4.7 multimodal mixin: image preprocessing, vision tower
execution and canvas M-RoPE position ids.

Pipeline (per image url):
  1. ``preprocess_input``: PIL image -> MiniCPM 4.6 PIL image processor
     (thumbnail + up to ``max_slice_nums`` slices, NaViT packed patches).
  2. ``embedding``: tower + final merger (see minicpmv47_vit.py) -> one
     feature tensor per image (all crops concatenated, row-major).
  3. Canvas M-RoPE positions per image: relative (T=0, H, W) so the C++
     MROPE position-id generator (``PositionIdsGenerator``, style 2) can add
     the canvas origin and continue text/decode positions exactly like the
     HF reference ``get_rope_index``.

Weight layout: ``model.vision_tower.*`` -> ``mm_part.visual.*`` and
``model.merger.*`` -> ``mm_part.merger.*``.
"""
import json
import os
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseMultiModalMixin,
    BaseVitWeights,
    VitParameters,
)
from rtp_llm.multimodal.multimodal_mixins.minicpmv47.minicpmv47_processor import (
    MiniCPMV47ImageProcessor,
)
from rtp_llm.multimodal.multimodal_mixins.minicpmv47.minicpmv47_vit import (
    MiniCPMV47Merger,
    MiniCPMV47VisionConfig,
    MiniCPMV47VisionModel,
    minicpmv47_canvas_position_ids,
    minicpmv47_image_features,
)
from rtp_llm.multimodal.multimodal_util import get_bytes_io_from_url
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MultiModalEmbeddingInterface,
)
from rtp_llm.ops import MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType
from rtp_llm.utils.database import CkptDatabase
from rtp_llm.model_loader.model_weight_info import ModelDeployWeightInfo, ModelWeightInfo
from rtp_llm.model_loader.weight_module import CustomAtomicWeight
from rtp_llm.utils.model_weight import CkptWeightInfo, identity


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


class MiniCPMV47ImageEmbedding(MultiModalEmbeddingInterface):
    def __init__(self, mm_related_params: VitParameters):
        ckpt_path = mm_related_params.config["ckpt_path"]
        model_config = _load_json(os.path.join(ckpt_path, "config.json"))
        preprocessor_config_path = os.path.join(ckpt_path, "preprocessor_config.json")
        preprocessor_config = _load_json(preprocessor_config_path) if os.path.exists(preprocessor_config_path) else {}

        self.mm_config = model_config
        self.downsample_mode = model_config.get("downsample_mode", "16x")
        self.merge_kernel_size = tuple(model_config.get("merge_kernel_size", (2, 2)))
        self.merger_times = model_config.get("merger_times", 1)
        self.vision_config = MiniCPMV47VisionConfig.from_checkpoint_config(
            model_config.get("vision_config", {}), model_config
        )
        self.patch_size = self.vision_config.patch_size

        self.image_processor = MiniCPMV47ImageProcessor(
            image_mean=preprocessor_config.get("image_mean", [0.5, 0.5, 0.5]),
            image_std=preprocessor_config.get("image_std", [0.5, 0.5, 0.5]),
            max_slice_nums=preprocessor_config.get("max_slice_nums", 9),
            scale_resolution=preprocessor_config.get("scale_resolution", model_config.get("image_size", 448)),
            patch_size=self.patch_size,
            slice_mode=preprocessor_config.get("slice_mode", True),
        )

        self.visual = MiniCPMV47VisionModel(self.vision_config)
        self.merger = MiniCPMV47Merger(
            self.vision_config.hidden_size,
            model_config["text_config"]["hidden_size"],
            merge_kernel_size=self.merge_kernel_size,
            merger_times=self.merger_times,
        )

    @property
    def _data_type(self):
        return self.visual.embeddings.patch_embedding.weight.dtype

    @property
    def _device(self):
        return self.visual.embeddings.patch_embedding.weight.device

    def get_preprocess_params(self):
        return {"processor": self.image_processor, "factor": self.patch_size * 4}

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput],
        vit_config: VitConfig,
        processor,
        factor: int = 56,
    ):
        assert len(mm_inputs) == 1
        mm_input = mm_inputs[0]
        if mm_input.mm_type != MMUrlType.IMAGE and mm_input.mm_type != MMUrlType.DEFAULT:
            raise ValueError(f"MiniCPM-V-4.7 supports image input only, got mm_type {mm_input.mm_type}")
        image_data = get_bytes_io_from_url(
            mm_input.url,
            vit_config.download_headers,
            max_file_size_kb=vit_config.mm_image_max_file_size_kb,
        )
        image = Image.open(image_data).convert("RGB")
        res = processor(image)
        return (
            res["pixel_values"],
            res["target_sizes"],
            res["grids"],
            res["num_patches_per_image"],
        )

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        res = self._embed_images([data])
        features, pos_ids = res[0]
        return features, pos_ids

    @torch.inference_mode()
    def batched_embedding(self, data_list: List[Any], mm_types: List[MMUrlType], **kwargs):
        if not all(mm_type == MMUrlType.IMAGE for mm_type in mm_types):
            raise ValueError("MiniCPM-V-4.7 supports image input only")
        return self._embed_images(data_list)

    def _embed_images(self, data_list: List[Any]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        pixel_values_list = []
        target_sizes_list = []
        grids: List[list] = []
        num_patches_per_image: List[int] = []
        for data in data_list:
            pixel_values_list.append(data[0])
            target_sizes_list.append(data[1])
            grids.extend(data[2])
            num_patches_per_image.extend(data[3])
        # pixel_values arrive bf16 from the preprocess worker; for the
        # single-image fast path skip the full-size CPU torch.cat copy and
        # upload the crop tensor directly (values identical: torch.cat of a
        # one-element list is a pure copy).
        if len(pixel_values_list) == 1:
            pixel_values = pixel_values_list[0]
        else:
            pixel_values = torch.cat(pixel_values_list, dim=-1)
        pixel_values = pixel_values.to(self._device)
        if pixel_values.dtype != self._data_type:
            pixel_values = pixel_values.to(self._data_type)
        # target_sizes stays on the host: the tower consumes it as host crop
        # grids (single conversion, integer metadata host-built), which keeps
        # the vision forward free of GPU->CPU syncs.
        if len(target_sizes_list) == 1:
            target_sizes = target_sizes_list[0]
        else:
            target_sizes = torch.cat(target_sizes_list, dim=0)

        crop_features = minicpmv47_image_features(
            self.visual,
            self.merger,
            pixel_values,
            target_sizes,
            self.downsample_mode,
        )

        results = []
        crop_offset = 0
        for num_crops in num_patches_per_image:
            features = torch.cat(crop_features[crop_offset : crop_offset + num_crops], dim=0)
            crop_grids = target_sizes[crop_offset : crop_offset + num_crops].tolist()
            grid = grids[crop_offset] if num_crops > 1 else None
            pos_ids = minicpmv47_canvas_position_ids(
                crop_grids, grid, self.downsample_mode, device=self._device
            )
            results.append((features.to(self._data_type), pos_ids))
            crop_offset += num_crops
        return results


class MiniCPMV47VitWeight(BaseVitWeights):
    """Two ckpt prefixes: model.vision_tower.* and model.merger.*."""

    def __init__(self, vit_part: Dict[str, Any]):
        super().__init__(vit_part, with_prefix=False)

    def _set_weight_prefix(self):
        # Actual ckpt prefixes are resolved per module in MiniCPMV47DeployWeightInfo.
        self._ckpt_prefix = "model.vision_tower."
        self._ft_prefix = "self.mm_part."

    def module_ckpt_prefixes(self) -> Dict[str, str]:
        return {
            "visual": "model.vision_tower.",
            "merger": "model.merger.",
        }


class MiniCPMV47DeployWeightInfo(ModelDeployWeightInfo):
    def __init__(self, vit_weights: MiniCPMV47VitWeight, vit_config: VitConfig):
        self.vit_weights = vit_weights
        self.vit_config = vit_config

    def get_weight_info(self):
        # weight_names carry the module prefix (visual./merger.) for the
        # ft-side path resolution; strip it and re-attach the module's ckpt
        # prefix (model.vision_tower. / model.merger.).
        weights = []
        prefix_map = self.vit_weights.module_ckpt_prefixes()
        for w in self.vit_weights.weight_names:
            module_name, rest = w.split(".", 1)
            ckpt_prefix = prefix_map.get(module_name)
            if ckpt_prefix is None:
                raise ValueError(f"MiniCPMV47VitWeight: unknown mm module {module_name!r} in weight {w!r}")
            weights.append(
                CustomAtomicWeight(w, [CkptWeightInfo(ckpt_prefix + rest, identity)], identity)
            )
        return ModelWeightInfo(layer_weights=[], weights=weights)


class MiniCPMV47Mixin(BaseMultiModalMixin):
    mm_part: MiniCPMV47ImageEmbedding

    def _init_multimodal(self):
        self.mm_part = MiniCPMV47ImageEmbedding(self.mm_related_params)
        self.mm_related_params.vit_weights = MiniCPMV47VitWeight(
            {"visual": self.mm_part.visual, "merger": self.mm_part.merger}
        )

    @classmethod
    def get_multimodal_mixin_weight_info(cls) -> ModelDeployWeightInfo:
        return MiniCPMV47DeployWeightInfo

    @classmethod
    def _get_mm_module(cls, mm_related_params: VitParameters, vit_config: VitConfig):
        mm_part = MiniCPMV47ImageEmbedding(mm_related_params)
        return torch.nn.ModuleList([mm_part.visual, mm_part.merger])

    def _prepare_vit_weights(self, database: CkptDatabase) -> None:
        names = database.get_pretrain_tensor_names()
        if not any(name.startswith("model.vision_tower.") for name in names):
            raise ValueError(
                f"MiniCPMV47VitWeight: checkpoint has no model.vision_tower.* weights "
                f"({len(names)} keys scanned)"
            )
        if not any(name.startswith("model.merger.") for name in names):
            raise ValueError(
                f"MiniCPMV47VitWeight: checkpoint has no model.merger.* weights "
                f"({len(names)} keys scanned)"
            )


register_multimodal_mixin(["minicpmv4_7"], MiniCPMV47Mixin)
