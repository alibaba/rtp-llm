"""
MiniMax-M3 VL multimodal mixin.

Handles both image and video inputs.  For images the HF processor produces
``]<]start of image[>[`` + N × ``]<]image[>[`` + ``]<]end of image[>[``;
for video each temporal group gets a timestamp prefix
``]<]X.X seconds[>[`` + the same bracket/token pattern.

This mixin reproduces the same token-count contract by:

1. Running the ViT + projector to get patch features (N rows).
2. Looking up the LLM word-embedding vectors for the bracket / timestamp
   tokens and concatenating them around the ViT features.
3. Returning one flat ``(total_tokens, hidden_dim)`` tensor so the C++
   ``expandTokenIds`` single-token mode replaces the 1 placeholder token
   with the correct number of embedding rows — no C++ changes needed.

The LLM word-embedding table is loaded once from the same checkpoint at
init time (kept on CPU, ~2.3 GB); fixed bracket embeddings are cached on
GPU.
"""

import json
import logging
import math
import os
from typing import Any, List, Optional

import torch
import torchvision
from PIL import Image
from transformers import AutoConfig, AutoTokenizer

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.mm_error_messages import MMErr, raise_mm
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

from .image_processor import (
    MiniMaxM3VLImageProcessor,
    compute_sampled_frame_indices,
    get_hw_multiple_of,
    smart_resize,
)
from .minimax_m3_vl_vit import MiniMaxM3VLVisionTower, VisionConfig  # noqa: F401

logger = logging.getLogger(__name__)


class MiniMaxM3VLImageEmbedding(MultiModalEmbeddingInterface):
    """
    Wraps the MiniMax-M3 VL vision stack.  Produces a single flat embedding
    tensor that includes bracket / timestamp word-embeddings around the ViT
    patch features so the C++ single-token expansion gets the right total
    token count.
    """

    # Token IDs for fixed bracket tokens (single-token special tokens).
    START_IMAGE_TOKEN_ID = 200029  # ]<]start of image[>[
    END_IMAGE_TOKEN_ID = 200030  # ]<]end of image[>[

    def __init__(self, mm_related_params: VitParameters):
        ckpt_path = mm_related_params.config["ckpt_path"]

        self.hf_config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)
        self.mm_processor = MiniMaxM3VLImageProcessor.from_pretrained(ckpt_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            ckpt_path, trust_remote_code=True
        )

        self.image_token_index = getattr(self.hf_config, "image_token_index", 200025)
        self.video_token_index = getattr(self.hf_config, "video_token_index", 200026)

        self.visual = MiniMaxM3VLVisionTower(self.hf_config).to(torch.bfloat16)

        # --- LLM word embedding (CPU, ~2.3 GB) ---
        self.word_embedding_weight = self._load_word_embedding(ckpt_path)

        # Pre-extract fixed bracket embeddings (will be moved to GPU lazily).
        self._start_emb = self.word_embedding_weight[self.START_IMAGE_TOKEN_ID].clone()
        self._end_emb = self.word_embedding_weight[self.END_IMAGE_TOKEN_ID].clone()
        self._bracket_embs_on_device = False

        # --- Video sampling defaults (from HF video processor config) ---
        self.video_fps = float(getattr(self.hf_config, "fps", 1.0))
        self.video_max_frames = int(getattr(self.hf_config, "max_frames", 768))
        self.video_min_frames = int(getattr(self.hf_config, "min_frames", 4))
        self.temporal_patch_size = self.mm_processor.temporal_patch_size
        self.merge_size = self.mm_processor.merge_size

    @staticmethod
    def _load_word_embedding(ckpt_path: str) -> torch.Tensor:
        from safetensors import safe_open

        index_path = os.path.join(ckpt_path, "model.safetensors.index.json")
        emb_key = "language_model.model.embed_tokens.weight"
        with open(index_path) as f:
            index = json.load(f)
        shard = index["weight_map"][emb_key]
        shard_path = os.path.join(ckpt_path, shard)
        with safe_open(shard_path, framework="pt", device="cpu") as sf:
            return sf.get_tensor(emb_key)

    def _ensure_bracket_embs_on_device(self):
        if not self._bracket_embs_on_device:
            device = self._device
            dtype = self._data_type
            self._start_emb = self._start_emb.to(device=device, dtype=dtype)
            self._end_emb = self._end_emb.to(device=device, dtype=dtype)
            self._bracket_embs_on_device = True

    @property
    def _data_type(self) -> torch.dtype:
        return self.visual.dtype

    @property
    def _device(self):
        return next(self.visual.parameters()).device

    def get_preprocess_params(self):
        return {
            "processor": self.mm_processor,
            "tokenizer": self.tokenizer,
            "image_token_index": self.image_token_index,
            "video_token_index": self.video_token_index,
            "video_fps": self.video_fps,
            "video_max_frames": self.video_max_frames,
            "video_min_frames": self.video_min_frames,
            "temporal_patch_size": self.temporal_patch_size,
            "merge_size": self.merge_size,
        }

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput],
        vit_config: VitConfig,
        processor=None,
        tokenizer=None,
        image_token_index: int = 200025,
        video_token_index: int = 200026,
        **kwargs,
    ):
        assert (
            len(mm_inputs) == 1
        ), f"MiniMaxM3VL expects exactly one mm input per call, got {len(mm_inputs)}"
        mm_input = mm_inputs[0]
        mm_type = mm_input.mm_type

        if mm_type == MMUrlType.DEFAULT or mm_type == MMUrlType.IMAGE:
            return MiniMaxM3VLImageEmbedding._preprocess_image(
                mm_input, vit_config, processor
            )
        elif mm_type == MMUrlType.VIDEO:
            return MiniMaxM3VLImageEmbedding._preprocess_video(
                mm_input, vit_config, processor, tokenizer, **kwargs
            )
        else:
            raise ValueError(f"unknown MMUrlType for MiniMaxM3VL: {mm_type}")

    @staticmethod
    def _preprocess_image(mm_input, vit_config, processor):
        """Download + decode only.  The resize/normalize/patch-fold transforms
        run later on GPU in ``embedding()`` — here we just decode to a raw uint8
        CHW tensor and precompute the (device-independent, cheap) target size
        from the per-request min/max pixels.

        Returns ``(raw_chw_uint8, (target_h, target_w), None)`` where the ``None``
        third element marks an image (no timestamps).
        """
        data = get_bytes_io_from_url(
            mm_input.url,
            vit_config.download_headers,
            max_file_size_kb=vit_config.mm_image_max_file_size_kb,
        )
        try:
            image = Image.open(data).convert("RGB")
        except Exception:
            raise_mm(MMErr.IMG_OPEN)
        raw = torchvision.transforms.functional.pil_to_tensor(image)  # uint8 [C,H,W]

        _, height, width = raw.shape
        factor = processor.patch_size * processor.merge_size
        min_pixels = processor.min_pixels
        max_pixels = processor.max_pixels
        pre_cfg = mm_input.mm_preprocess_config
        if getattr(pre_cfg, "max_pixels", -1) > 0:
            max_pixels = int(pre_cfg.max_pixels)
        if getattr(pre_cfg, "min_pixels", -1) > 0:
            min_pixels = int(pre_cfg.min_pixels)

        target_h, target_w = smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            min_image_dimension=vit_config.mm_image_min_dimension,
            max_image_aspect_ratio=vit_config.mm_image_max_aspect_ratio,
        )
        return raw, (target_h, target_w), None

    @staticmethod
    def _preprocess_video(mm_input, vit_config, processor, tokenizer, **kwargs):
        from decord import VideoReader, bridge

        bridge.set_bridge("torch")

        video_bytes = get_bytes_io_from_url(
            mm_input.url,
            vit_config.download_headers,
            max_file_size_kb=vit_config.mm_video_max_file_size_kb,
        )
        try:
            vr = VideoReader(video_bytes)
            total_frames = len(vr)
            video_fps = float(vr.get_avg_fps())
        except Exception:
            raise_mm(MMErr.VIDEO_INVALID)

        if total_frames <= 0 or video_fps <= 0 or not math.isfinite(video_fps):
            raise_mm(MMErr.VIDEO_INVALID)

        pre_cfg = mm_input.mm_preprocess_config
        target_fps = getattr(pre_cfg, "fps", 0)
        if not target_fps or target_fps <= 0:
            target_fps = kwargs.get("video_fps", 1.0)
        max_frames = getattr(pre_cfg, "max_frames", 0)
        if not max_frames or max_frames <= 0:
            max_frames = kwargs.get("video_max_frames", 768)
        configured_max_frames = vit_config.mm_video_max_frames
        if configured_max_frames <= 0:
            configured_max_frames = VitConfig.DEFAULT_MM_VIDEO_MAX_FRAMES
        max_frames = min(int(max_frames), int(configured_max_frames))

        indices = compute_sampled_frame_indices(
            total_frames, video_fps, target_fps, int(max_frames)
        )
        if not indices:
            raise_mm(MMErr.VIDEO_REQ.format("no video frames can be sampled"))

        # Decode the sampled frames only (the "load" step).  Resize/normalize/
        # patch-fold happen on GPU in embedding(); here we just keep raw uint8
        # frames + the cheap target size + timestamps.
        try:
            frames_tensor = vr.get_batch(indices)  # (N, H, W, C) uint8 torch (CPU)
        except Exception:
            raise_mm(MMErr.VIDEO_INVALID)

        patch_size = processor.patch_size
        merge_size = kwargs.get("merge_size", 2)
        temporal_patch_size = kwargs.get("temporal_patch_size", 2)
        factor = patch_size * merge_size

        num_frames, src_h, src_w, _ = frames_tensor.shape
        # sglang uses get_hw_multiple_of with frame_max_size (longest-edge cap)
        frame_max_size = getattr(processor, "max_pixels", None)
        if frame_max_size and isinstance(frame_max_size, int):
            # Convert pixel budget to longest-edge estimate for
            # get_hw_multiple_of.  Fall back to ceil-align only.
            edge = int(math.sqrt(frame_max_size))
            target_w, target_h = get_hw_multiple_of((src_w, src_h), factor, edge)
        else:
            target_w, target_h = get_hw_multiple_of((src_w, src_h), factor, None)

        # grid_t after temporal padding (matches the fold done in _gpu_fold).
        pad_n = (
            temporal_patch_size - num_frames % temporal_patch_size
        ) % temporal_patch_size
        grid_t = (num_frames + pad_n) // temporal_patch_size

        # Compute per-temporal-group timestamp token IDs.
        timestamp_token_ids: List[List[int]] = []
        for gi in range(grid_t):
            raw_frame_idx = indices[min(gi * temporal_patch_size, len(indices) - 1)]
            ts = raw_frame_idx / video_fps
            ts_text = f"]<]{ts:.1f} seconds[>["
            ts_ids = tokenizer.encode(ts_text, add_special_tokens=False)
            timestamp_token_ids.append(ts_ids)

        return frames_tensor, (target_h, target_w), timestamp_token_ids

    def _lookup_word_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Index into the CPU word-embedding table and move result to GPU."""
        return self.word_embedding_weight[token_ids].to(
            device=self._device, dtype=self._data_type
        )

    def _assemble_image(self, vit_feats: torch.Tensor) -> torch.Tensor:
        """Prepend start-of-image and append end-of-image embeddings."""
        return torch.cat(
            [
                self._start_emb.unsqueeze(0),
                vit_feats,
                self._end_emb.unsqueeze(0),
            ],
            dim=0,
        )

    def _assemble_video(
        self,
        vit_feats: torch.Tensor,
        grid_thw: torch.Tensor,
        timestamp_token_ids: List[List[int]],
        ts_embs_cache: Optional[torch.Tensor] = None,
        ts_offset: int = 0,
    ) -> tuple:
        """Interleave timestamp / bracket / ViT embeddings per temporal group.

        Returns ``(assembled_tensor, new_ts_offset)`` so the caller can track
        position within a shared ``ts_embs_cache`` across items.
        """
        grid_t = grid_thw[0][0].item()
        grid_h = grid_thw[0][1].item()
        grid_w = grid_thw[0][2].item()
        merge_length = self.merge_size**2
        frame_seqlen = (grid_h * grid_w) // merge_length

        chunks: List[torch.Tensor] = []
        for gi in range(grid_t):
            # Timestamp word embeddings
            ts_ids = timestamp_token_ids[gi]
            n_ts = len(ts_ids)
            if ts_embs_cache is not None:
                ts_emb = ts_embs_cache[ts_offset : ts_offset + n_ts]
                ts_offset += n_ts
            else:
                ts_emb = self._lookup_word_embeddings(
                    torch.tensor(ts_ids, dtype=torch.long)
                )

            # ViT features for this temporal group
            start = gi * frame_seqlen
            group_feats = vit_feats[start : start + frame_seqlen]

            chunks.extend(
                [
                    ts_emb,
                    self._start_emb.unsqueeze(0),
                    group_feats,
                    self._end_emb.unsqueeze(0),
                ]
            )

        return torch.cat(chunks, dim=0), ts_offset

    def _gpu_fold(self, frames_nchw: torch.Tensor, target_hw) -> tuple:
        """GPU resize + rescale + normalize + temporal-pad + patch-fold.

        Takes raw decoded frames ``[N, C, H, W]`` (N=1 for images) on any device
        and returns ``(pixel_values [total_patches, patch_dim], grid_thw [1,3])``.
        The fold math is identical to the previous CPU preprocess; only the device
        differs (torchvision bicubic runs on GPU when ``frames_nchw`` is on cuda).
        """
        device = self._device
        dtype = self._data_type
        p = self.mm_processor
        patch_size = p.patch_size
        merge_size = self.merge_size
        temporal_patch_size = self.temporal_patch_size
        target_h, target_w = target_hw

        frames = frames_nchw.to(device=device).float()
        frames = torchvision.transforms.functional.resize(
            frames,
            [target_h, target_w],
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
        )

        video = frames.unsqueeze(0)  # (1, T, C, H, W)
        video = video * p.rescale_factor
        mean = torch.tensor(p.image_mean, device=device).view(1, 1, 3, 1, 1)
        std = torch.tensor(p.image_std, device=device).view(1, 1, 3, 1, 1)
        video = (video - mean) / std

        T = video.shape[1]
        pad_n = (temporal_patch_size - T % temporal_patch_size) % temporal_patch_size
        if pad_n:
            tail = video[:, -1:].repeat(1, pad_n, 1, 1, 1)
            video = torch.cat([video, tail], dim=1)

        B, T_pad, channel, H, W = video.shape
        grid_t = T_pad // temporal_patch_size
        grid_h, grid_w = H // patch_size, W // patch_size

        patches = video.view(
            B,
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        flatten = patches.reshape(
            B,
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )
        pixel_values = flatten.squeeze(0).to(dtype)
        grid_thw = torch.tensor(
            [[grid_t, grid_h, grid_w]], dtype=torch.long, device=device
        )
        return pixel_values, grid_thw

    @staticmethod
    def _raw_to_nchw(raw: torch.Tensor, is_video: bool) -> torch.Tensor:
        """Normalize a raw decoded input to ``[N, C, H, W]``.

        Image raw is ``[C, H, W]`` (from ``pil_to_tensor``); video raw is
        ``[N, H, W, C]`` (from decord ``get_batch``).
        """
        if is_video:
            return raw.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return raw.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]

    @torch.inference_mode()
    def embedding(self, data, mm_type=None, **kwargs):
        raw, target_hw, timestamp_token_ids = data
        device = self._device
        dtype = self._data_type

        self._ensure_bracket_embs_on_device()

        is_video = timestamp_token_ids is not None
        frames_nchw = self._raw_to_nchw(raw, is_video)
        pixel_values, grid_thw = self._gpu_fold(frames_nchw, target_hw)
        vit_feats = self.visual(pixel_values, grid_thw).to(dtype)

        if not is_video:
            result = self._assemble_image(vit_feats)
        else:
            result, _ = self._assemble_video(vit_feats, grid_thw, timestamp_token_ids)

        position_ids = torch.arange(result.shape[0], device=device, dtype=torch.int32)
        return result, position_ids

    @torch.inference_mode()
    def batched_embedding(
        self, data_list: List[Any], mm_types: List[MMUrlType], **kwargs
    ):
        if len(data_list) <= 1:
            return super().batched_embedding(data_list, mm_types, **kwargs)

        device = self._device
        dtype = self._data_type
        self._ensure_bracket_embs_on_device()

        # --- 1. GPU transform (resize/normalize/fold) per item, then batch ViT ---
        all_pv: List[torch.Tensor] = []
        all_thw: List[torch.Tensor] = []
        grids: List[torch.Tensor] = []
        split_sizes: List[int] = []

        # self.visual applies the spatial 2x2 merge (patch_merge_mlp), so its
        # output has pv.shape[0] // merge_size**2 rows per item. Split the batched
        # ViT output by the POST-merge count, not the pre-merge patch count.
        merge_length = self.merge_size**2
        for raw, target_hw, ts_info in data_list:
            frames_nchw = self._raw_to_nchw(raw, ts_info is not None)
            pv, thw = self._gpu_fold(frames_nchw, target_hw)
            all_pv.append(pv)
            all_thw.append(thw)
            grids.append(thw)
            split_sizes.append(pv.shape[0] // merge_length)

        batched_pv = torch.cat(all_pv, dim=0)
        batched_thw = torch.cat(all_thw, dim=0)
        all_vit_feats = self.visual(batched_pv, batched_thw).to(dtype)
        per_item_feats = torch.split(all_vit_feats, split_sizes, dim=0)

        # --- 2. Batch word-embedding lookup for all timestamps ---
        all_ts_ids: List[int] = []
        for _, _, ts_info in data_list:
            if ts_info is not None:
                for ids in ts_info:
                    all_ts_ids.extend(ids)

        ts_embs_all: Optional[torch.Tensor] = None
        if all_ts_ids:
            ts_tensor = torch.tensor(all_ts_ids, dtype=torch.long)
            ts_embs_all = self.word_embedding_weight[ts_tensor].to(
                device=device, dtype=dtype
            )

        # --- 3. Assemble per-item results ---
        results: List[tuple] = []
        ts_offset = 0

        for i, (_, _, ts_info) in enumerate(data_list):
            vit_feats = per_item_feats[i]

            if ts_info is None:
                emb = self._assemble_image(vit_feats)
            else:
                emb, ts_offset = self._assemble_video(
                    vit_feats,
                    grids[i],
                    ts_info,
                    ts_embs_cache=ts_embs_all,
                    ts_offset=ts_offset,
                )

            pos = torch.arange(emb.shape[0], device=device, dtype=torch.int32)
            results.append((emb, pos))

        return results


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
