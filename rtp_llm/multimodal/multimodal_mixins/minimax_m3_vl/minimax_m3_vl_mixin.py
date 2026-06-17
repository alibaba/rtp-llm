"""
MiniMax-M3 VL multimodal mixin.

The on-disk MiniMax-M3 VL checkpoint ships a bundled HuggingFace processor
(`MiniMaxVLProcessor` + `MiniMaxM3VLImageProcessor` + `MiniMaxM3VLVideoProcessor`)
and a bundled HF config (`MiniMaxM3VLConfig`).  We re-use both of those via
`AutoConfig.from_pretrained(..., trust_remote_code=True)` and
`AutoProcessor.from_pretrained(..., trust_remote_code=True)` so the token-id
contract here stays in lock-step with what the processor produces, and so
the same preprocessing path used by sglang / HF for this model is exercised
end-to-end inside rtp-llm.

The vision tower itself (`MiniMaxM3VLVisionTower`) is a custom module written
by the sibling file `minimax_m3_vl_vit.py`; this mixin only knows how to:
  1. build the tower from the HF config and run it on preprocessed pixels;
  2. surface the three top-level on-disk weight prefixes
     (`vision_tower.*`, `multi_modal_projector.*`, `patch_merge_mlp.*`) to
     rtp-llm's MultimodalMixin loader so they get materialized onto the GPU.

Video is intentionally NotImplementedError for v1 — sglang already exposes a
much heavier video path (frame sampling, temporal patch pruning, timestamp
prompt expansion) that we don't need until the LLM side wires it in.
"""

from typing import Any, List

import torch
from PIL import Image
from transformers import AutoConfig, AutoTokenizer

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseMultiModalDeployWeightInfo,
    BaseMultiModalMixin,
    BaseVitWeights,
)
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MultiModalEmbeddingInterface,
)
from rtp_llm.multimodal.multimodal_util import get_bytes_io_from_url
from rtp_llm.ops import MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType, VitParameters

from .image_processor import MiniMaxM3VLImageProcessor
from .minimax_m3_vl_vit import MiniMaxM3VLVisionTower, VisionConfig  # noqa: F401


class MiniMaxM3VLImageEmbedding(MultiModalEmbeddingInterface):
    """
    Wraps the MiniMax-M3 VL vision stack so the rtp-llm multimodal harness can
    treat it like any other image embedder: preprocess one URL into
    (pixel_values, grid_thw), then run it through `self.visual` to get the
    flattened patch features the LLM consumes.
    """

    def __init__(self, mm_related_params: VitParameters):
        ckpt_path = mm_related_params.config["ckpt_path"]

        self.hf_config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)
        self.mm_processor = MiniMaxM3VLImageProcessor.from_pretrained(ckpt_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            ckpt_path, trust_remote_code=True
        )

        # M3's HF config places image_token_index / video_token_index at the
        # top level (see configuration_minimax_m3_vl.py). Default to the values
        # baked into the released config to keep this robust against
        # config-subset checkpoints.
        self.image_token_index = getattr(self.hf_config, "image_token_index", 200025)
        self.video_token_index = getattr(self.hf_config, "video_token_index", 200026)

        # The base mixin instantiates this class inside a
        # `torch.set_default_dtype(compute_dtype)` context, BUT only when
        # invoked through BaseMultiModalMixin.__init__. eval_mm_model_size also
        # instantiates the embedding outside that context (purely on CPU /
        # meta) to count params, so cast explicitly to bf16 to keep both paths
        # consistent and to avoid loading fp32 visual weights by accident.
        self.visual = MiniMaxM3VLVisionTower(self.hf_config).to(torch.bfloat16)

    @property
    def _data_type(self) -> torch.dtype:
        return self.visual.dtype

    @property
    def _device(self):
        # Asking nn.Module.device directly only works if the model has
        # parameters AND they're all on one device; pulling the first
        # parameter is the same pattern HF's modeling utils use.
        return next(self.visual.parameters()).device

    def get_preprocess_params(self):
        # These get passed as kwargs into preprocess_input from the
        # mm_process_engine workers; staticmethod => no `self` access there.
        return {
            "processor": self.mm_processor,
            "image_token_index": self.image_token_index,
            "video_token_index": self.video_token_index,
        }

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput],
        vit_config: VitConfig,
        processor=None,
        image_token_index: int = 200025,
        video_token_index: int = 200026,
        **kwargs,
    ):
        # The rtp-llm preprocessing path feeds inputs one at a time even when
        # the request batches several images — the harness re-batches results
        # downstream via batched_embedding. Keep the single-input invariant.
        assert (
            len(mm_inputs) == 1
        ), f"MiniMaxM3VL expects exactly one mm input per call, got {len(mm_inputs)}"
        mm_input = mm_inputs[0]
        mm_type = mm_input.mm_type

        if mm_type == MMUrlType.DEFAULT or mm_type == MMUrlType.IMAGE:
            data = get_bytes_io_from_url(mm_input.url, vit_config.download_headers)
            image = Image.open(data).convert("RGB")
            # MiniMaxM3VLImageProcessor returns BatchFeature with the standard
            # Qwen-VL-style keys: pixel_values [N_patches, channels*ph*pw]
            # (typically 1176 = 3*14*14*2*2 after temporal/spatial folding)
            # plus image_grid_thw [num_images, 3] (T, H/patch, W/patch).
            # `processor` here IS the image processor (AutoImageProcessor),
            # not the multi-component AutoProcessor — so call it directly.
            # The bundled MiniMaxM3VLImageProcessor declares only 4 fields
            # in valid_kwargs (patch_size/temporal_patch_size/merge_size/
            # max_pixels), so BaseImageProcessorFast.preprocess only
            # auto-populates those from class attrs. All other params it
            # pops without defaults — we must forward them explicitly from
            # the loaded instance's class attrs.
            forwarded = {
                attr: getattr(processor, attr, None)
                for attr in (
                    "do_resize",
                    "size",
                    "default_to_square",
                    "resample",
                    "do_rescale",
                    "rescale_factor",
                    "do_normalize",
                    "image_mean",
                    "image_std",
                    "do_convert_rgb",
                    "disable_grouping",
                    "input_data_format",
                    "device",
                )
            }
            res = processor(images=[image], return_tensors="pt", **forwarded)
            return res["pixel_values"], res["image_grid_thw"]
        elif mm_type == MMUrlType.VIDEO:
            # v1 ships image-only. Video would need the temporal-patch
            # pruning + timestamp prompt rewriting that sglang's
            # MiniMaxM3VLProcessor handles; wiring that into rtp-llm's
            # prompt-side pipeline is a separate effort.
            raise NotImplementedError("MiniMaxM3VL video input is not supported in v1")
        else:
            raise ValueError(f"unknown MMUrlType for MiniMaxM3VL: {mm_type}")

    @torch.inference_mode()
    def embedding(self, data, mm_type=None, **kwargs):
        pixel_values, grid_thw = data
        device = self._device
        # pixel_values must be bf16 to match the vision tower weights; grid_thw
        # is an int32 index tensor and must NOT be cast to bf16 or the patch
        # arithmetic inside the tower overflows / loses precision.
        pixel_values = pixel_values.to(device).to(self._data_type)
        grid_thw = grid_thw.to(device)

        feats = self.visual(pixel_values, grid_thw)

        # MiniMax-M3 does NOT use MRope on the vision side — position ids for
        # vision tokens are just a flat 0..N-1 range (style 0). Returning
        # int32 keeps it compatible with the LLM's position-id dtype.
        position_ids = torch.arange(
            feats.shape[0], device=feats.device, dtype=torch.int32
        )
        return feats.to(self._data_type), position_ids

    @torch.inference_mode()
    def batched_embedding(
        self, data_list: List[Any], mm_types: List[MMUrlType], **kwargs
    ):
        # v1: fall back to the per-item loop in the base class.  A real batched
        # path would concat pixel_values along dim 0 and run a single visual
        # forward, but that requires the vision tower's grid_thw handling to
        # cope with multiple grids — defer until needed.
        return super().batched_embedding(data_list, mm_types, **kwargs)


class MiniMaxM3VLVitWeight(BaseVitWeights):
    """
    Names every weight loaded from disk for the MiniMax-M3 VL vision stack.

    On disk (top-level of the safetensors index) the keys are bare:
        vision_tower.vision_model.*, multi_modal_projector.*, patch_merge_mlp.*
    so `_ckpt_prefix` is empty.

    In the rtp-llm runtime the MiniMaxM3VLVisionTower composite is hung off
    `self.mm_part.visual` and its child attribute names (`vision_tower`,
    `multi_modal_projector`, `patch_merge_mlp`) are chosen to mirror the
    on-disk hierarchy exactly. So passing a single-entry dict
    `{"vit": self.mm_part.visual}` (no extra prefix needed) makes
    BaseVitWeights emit state_dict() keys whose full dotted path matches the
    on-disk key 1:1, and the loader resolves each one back to a live tensor
    by walking `self.mm_part.visual.<weight_name>`.
    """

    def _set_weight_prefix(self):
        self._ckpt_prefix = ""
        self._ft_prefix = "self.mm_part.visual."


class MiniMaxM3VLMixin(BaseMultiModalMixin):
    def _init_multimodal(self) -> None:
        self.mm_part = MiniMaxM3VLImageEmbedding(self.mm_related_params)

        # The live MiniMaxM3VLVisionTower hierarchy is structurally
        # isomorphic to the on-disk checkpoint hierarchy:
        #   visual.vision_tower.vision_model.*  /  visual.multi_modal_projector.*
        #   visual.patch_merge_mlp.*
        # so we register the composite tower as a single root and let
        # BaseVitWeights pull state_dict() keys verbatim — they already match
        # the on-disk top-level naming, and they round-trip back to the live
        # tensors via the ft_prefix "self.mm_part.visual.".
        self.mm_related_params.vit_weights = MiniMaxM3VLVitWeight(
            {"vit": self.mm_part.visual}
        )

    @classmethod
    def get_multimodal_mixin_weight_info(cls):
        return BaseMultiModalDeployWeightInfo

    @classmethod
    def _get_mm_module(cls, mm_related_params: VitParameters, vit_config: VitConfig):
        # Used only by eval_mm_model_size / eval_mm_model_param_count to
        # estimate device memory before the full mixin is instantiated.
        # Returning the vision tower (which contains all three trainable
        # sub-modules) is exactly the slice whose params we want to count.
        return MiniMaxM3VLImageEmbedding(mm_related_params).visual


register_multimodal_mixin(["minimax_m3_vl"], MiniMaxM3VLMixin)
