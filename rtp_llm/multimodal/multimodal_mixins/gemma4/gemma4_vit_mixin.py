"""Gemma4 multimodal mixin: vision tower weights, image embedding, registration."""

import json
import math
import os
import re
from typing import List

import torch
from PIL import Image
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import CustomAtomicWeight
from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseMultiModalDeployWeightInfo,
    BaseMultiModalMixin,
    BaseVitWeights,
    VitParameters,
)
from rtp_llm.multimodal.multimodal_mixins.gemma4.image_processing_gemma4 import (
    Gemma4ImageProcessor,
    Gemma4VideoProcessor,
)
from rtp_llm.multimodal.multimodal_mixins.gemma4.modeling_gemma4_vision import (
    VISION_CONFIG_DEFAULTS,
    Gemma4VisionModel,
)
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MultiModalEmbeddingInterface,
    get_bytes_io_from_url,
)
from rtp_llm.ops import MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType
from rtp_llm.utils.model_weight import CkptWeightInfo, identity
from tokenizers import Tokenizer

try:
    from decord import VideoReader, cpu
except ModuleNotFoundError:
    VideoReader = None
    cpu = None


class Gemma4VitWeight(BaseVitWeights):
    def _set_weight_prefix(self):
        self._ckpt_prefix = "model."
        self._ft_prefix = "self.mm_part.visual."


class Gemma4DeployWeightInfo(BaseMultiModalDeployWeightInfo):
    """Deploy weight info with the ckpt name mapping this model needs.

    The module builds nn.Linear layers from raw ckpt tensors (see
    modeling_gemma4_vision._linear), so its state_dict keys do not match the
    checkpoint's HF-style names: attention projections drop the `.linear.`
    infix and MLP projections additionally drop the `mlp.` infix. The default
    identity mapping would query 189 non-existent tensor names.
    """

    @staticmethod
    def _ckpt_name(w: str) -> str:
        parts = w.split(".")
        if parts[-1] == "weight" and parts[-2] in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ):
            return "model." + ".".join(parts[:-1] + ["linear", "weight"])
        if parts[-1] == "weight" and parts[-2] in (
            "gate_proj",
            "up_proj",
            "down_proj",
        ):
            return "model." + ".".join(
                parts[:-2] + ["mlp"] + parts[-2:-1] + ["linear", "weight"]
            )
        return "model." + w

    def get_weight_info(self) -> ModelWeightInfo:
        weights = []
        for w in self.vit_weights.weight_names:
            weights.append(
                CustomAtomicWeight(
                    w, [CkptWeightInfo(self._ckpt_name(w), identity)], identity
                )
            )
        return ModelWeightInfo(layer_weights=[], weights=weights)


class Gemma4PromptExpander:
    def __init__(self, ckpt_path: str):
        self.tokenizer = Tokenizer.from_file(os.path.join(ckpt_path, "tokenizer.json"))

    def expand_rendered_prompt(self, rendered_prompt, expansion_metadata):
        if not rendered_prompt or not expansion_metadata:
            return None
        replacements = []
        metadata_index = 0
        while metadata_index < len(expansion_metadata):
            metadata = expansion_metadata[metadata_index]
            kind = metadata["kind"]
            soft_tokens = int(metadata["soft_tokens"])
            if soft_tokens <= 0:
                raise ValueError("Gemma4 expansion requires positive soft_tokens")
            if kind == "image":
                if metadata["frame_number"] != 0 or metadata["frame_count"] != 1:
                    raise ValueError("invalid Gemma4 image expansion metadata")
                replacements.append(
                    (
                        "<|image|>",
                        "<|image>" + "<|image|>" * soft_tokens + "<image|>",
                    )
                )
                metadata_index += 1
                continue
            if kind != "video":
                raise ValueError(f"unsupported Gemma4 expansion kind: {kind}")
            frame_count = int(metadata["frame_count"])
            if frame_count <= 0:
                raise ValueError("Gemma4 video expansion requires positive frame_count")
            frames = expansion_metadata[metadata_index : metadata_index + frame_count]
            if len(frames) != frame_count or any(
                frame["kind"] != "video"
                or frame["frame_number"] != frame_number
                or frame["frame_count"] != frame_count
                or int(frame["soft_tokens"]) <= 0
                for frame_number, frame in enumerate(frames)
            ):
                raise ValueError("incomplete Gemma4 video expansion metadata")
            frame_replacements = []
            previous_frame_index = -1
            for frame in frames:
                fps = float(frame["fps"])
                frame_index = int(frame["frame_index"])
                if not math.isfinite(fps) or fps <= 0:
                    raise ValueError(
                        "Gemma4 video expansion requires positive finite FPS"
                    )
                if frame_index < 0 or frame_index < previous_frame_index:
                    raise ValueError("Gemma4 video frame indices must be ordered")
                previous_frame_index = frame_index
                seconds = frame_index / fps
                timestamp = f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
                frame_replacements.append(
                    timestamp
                    + " <|image>"
                    + "<|video|>" * int(frame["soft_tokens"])
                    + "<image|>"
                )
            replacements.append(("<|video|>", " ".join(frame_replacements)))
            metadata_index += frame_count

        replacement_iter = iter(replacements)

        def replace_placeholder(match):
            try:
                expected, replacement = next(replacement_iter)
            except StopIteration as error:
                raise ValueError(
                    "more Gemma4 placeholders than media inputs"
                ) from error
            if match.group(0) != expected:
                raise ValueError(
                    f"Gemma4 placeholder order mismatch: {match.group(0)} != {expected}"
                )
            return replacement

        expanded_prompt = re.sub(
            r"<\|image\|>|<\|video\|>", replace_placeholder, rendered_prompt
        )
        try:
            next(replacement_iter)
        except StopIteration:
            pass
        else:
            raise ValueError("fewer Gemma4 placeholders than media inputs")
        return self.tokenizer.encode(expanded_prompt, add_special_tokens=False).ids


class Gemma4ImageEmbedding(MultiModalEmbeddingInterface):
    def __init__(self, mm_related_params: VitParameters):
        self.mm_related_params = mm_related_params
        ckpt_path = mm_related_params.config["ckpt_path"]
        self.image_processor = Gemma4ImageProcessor.from_pretrained(ckpt_path)
        self.video_processor = Gemma4VideoProcessor.from_pretrained(ckpt_path)
        self.prompt_expander = Gemma4PromptExpander(ckpt_path)

        config = dict(VISION_CONFIG_DEFAULTS)
        config_path = os.path.join(ckpt_path, "config.json")
        with open(config_path) as f:
            config_json = json.load(f)
        vision_config = config_json.get("vision_config", {})
        for key in (
            "hidden_size",
            "head_dim",
            "num_hidden_layers",
            "intermediate_size",
            "patch_size",
            "pooling_kernel_size",
            "position_embedding_size",
            "rms_norm_eps",
            "standardize",
        ):
            if key in vision_config:
                config[key] = vision_config[key]
        rope_params = vision_config.get("rope_parameters", {})
        if "rope_theta" in rope_params:
            config["rope_theta"] = rope_params["rope_theta"]
        text_config = config_json.get("text_config", config_json)
        config["text_hidden_size"] = text_config["hidden_size"]

        self.visual = Gemma4VisionModel(config, weights=None).share_memory()
        self.output_dtype = self.visual.embed_vision.embedding_projection.weight.dtype

    @property
    def _data_type(self):
        return self.visual.embed_vision.embedding_projection.weight.dtype

    @property
    def _device(self):
        return self.visual.embed_vision.embedding_projection.weight.device

    @staticmethod
    def load_image(data, configs, **kwargs):
        return Image.open(data).convert("RGB")

    @staticmethod
    def _sample_video_indices(total_frames: int, source_fps: float, configs):
        target_fps = getattr(configs, "fps", -1)
        min_frames = getattr(configs, "min_frames", -1)
        max_frames = getattr(configs, "max_frames", -1)
        if target_fps is not None and target_fps > 0:
            frame_count = int(total_frames / source_fps * target_fps)
        else:
            frame_count = 32
        if min_frames is not None and min_frames > 0:
            frame_count = max(frame_count, min_frames)
        if max_frames is not None and max_frames > 0:
            frame_count = min(frame_count, max_frames)
        if frame_count <= 0 or frame_count > total_frames:
            raise ValueError(
                f"cannot sample {frame_count} frames from a {total_frames}-frame video"
            )
        return torch.arange(0, total_frames, total_frames / frame_count).int().tolist()

    @staticmethod
    def load_video(data, configs):
        if VideoReader is None:
            raise ImportError("decord is required for Gemma4 video processing")
        reader = VideoReader(data, ctx=cpu(0), num_threads=1)
        total_frames = len(reader)
        source_fps = float(reader.get_avg_fps())
        if source_fps <= 0:
            raise ValueError(f"video FPS must be positive, got {source_fps}")
        indices = Gemma4ImageEmbedding._sample_video_indices(
            total_frames, source_fps, configs
        )
        video = torch.tensor(reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2)
        del reader
        return video, {
            "fps": source_fps,
            "frame_indices": indices,
        }

    @staticmethod
    def preprocess_input(
        mm_inputs: List[MultimodalInput],
        vit_config: VitConfig,
        processor,
        video_processor,
    ):
        assert len(mm_inputs) == 1
        mm_input = mm_inputs[0]
        mm_type = mm_input.mm_type
        if mm_type == MMUrlType.DEFAULT or mm_type == MMUrlType.IMAGE:
            data = get_bytes_io_from_url(
                mm_input.url,
                vit_config.download_headers,
                max_file_size_kb=vit_config.mm_image_max_file_size_kb,
            )
            image = Gemma4ImageEmbedding.load_image(data, mm_input.mm_preprocess_config)
            res = processor(images=[image])
            return (
                res["pixel_values"],
                res["image_position_ids"],
                [
                    {
                        "kind": "image",
                        "frame_number": 0,
                        "frame_count": 1,
                        "soft_tokens": int(res["num_soft_tokens_per_image"][0]),
                    }
                ],
            )
        if mm_type == MMUrlType.VIDEO:
            data = get_bytes_io_from_url(
                mm_input.url,
                vit_config.download_headers,
                max_file_size_kb=vit_config.mm_video_max_file_size_kb,
            )
            video, metadata = Gemma4ImageEmbedding.load_video(
                data, mm_input.mm_preprocess_config
            )
            res = video_processor(videos=[video])
            soft_tokens = int(res["num_soft_tokens_per_video"][0])
            frame_indices = metadata["frame_indices"]
            expansion_metadata = [
                {
                    "kind": "video",
                    "fps": metadata["fps"],
                    "frame_index": frame_index,
                    "frame_number": frame_number,
                    "frame_count": len(frame_indices),
                    "soft_tokens": soft_tokens,
                }
                for frame_number, frame_index in enumerate(frame_indices)
            ]
            return (
                res["pixel_values_videos"],
                res["video_position_ids"],
                expansion_metadata,
            )
        raise ValueError(f"unknown mm url type: {mm_type}")

    def get_preprocess_params(self):
        return {
            "processor": self.image_processor,
            "video_processor": self.video_processor,
        }

    def expand_rendered_prompt(self, rendered_prompt, expansion_metadata):
        return self.prompt_expander.expand_rendered_prompt(
            rendered_prompt, expansion_metadata
        )

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        pixel_values = data[0].to(self._device)
        position_ids = data[1].to(self._device)
        expansion_metadata = data[2] if len(data) > 2 else None
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.flatten(0, 1)
            position_ids = position_ids.flatten(0, 1)
        embeddings = self.visual(pixel_values, position_ids).to(self.output_dtype)
        if expansion_metadata is None:
            return embeddings, None
        if expansion_metadata[0]["kind"] == "video":
            soft_tokens = expansion_metadata[0]["soft_tokens"]
            embeddings = list(embeddings.split(soft_tokens, dim=0))
            if len(embeddings) != len(expansion_metadata):
                raise ValueError("video embeddings do not match sampled frame count")
        return embeddings, None, None, expansion_metadata


class Gemma4Mixin(BaseMultiModalMixin):
    def _init_multimodal(self):
        self.mm_part = Gemma4ImageEmbedding(self.mm_related_params)
        self.mm_part.output_dtype = self.compute_dtype
        self.mm_related_params.vit_weights = Gemma4VitWeight(
            {"vit": self.mm_part.visual}
        )

    def load_mm_weight(self, ctype, device):
        super().load_mm_weight(ctype, device)
        self.mm_part.visual.to(dtype=torch.float32, device=device)

    @classmethod
    def get_multimodal_mixin_weight_info(cls) -> ModelDeployWeightInfo:
        return Gemma4DeployWeightInfo

    @classmethod
    def _get_mm_module(cls, mm_related_params: VitParameters, vit_config: VitConfig):
        return Gemma4ImageEmbedding(mm_related_params).visual

    @classmethod
    def create_prompt_expander(cls, mm_related_params: VitParameters):
        return Gemma4PromptExpander(mm_related_params.config["ckpt_path"])


register_multimodal_mixin(["gemma4"], Gemma4Mixin)
