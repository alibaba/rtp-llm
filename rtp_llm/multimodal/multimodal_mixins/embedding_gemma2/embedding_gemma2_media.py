"""Checkpoint-native EmbeddingGemma 2 audio/video around the shared vision stack."""

import json
import os
import re

import librosa
import numpy as np
import torch
from PIL import Image
from torch import nn

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import TokenizerFactory
from rtp_llm.models_py.modules.gemma4.core import gemma4_rms_norm
from rtp_llm.multimodal.multimodal_mixins.gemma4.feature_extraction_gemma4 import (
    Gemma4AudioFeatureExtractor,
)
from rtp_llm.multimodal.multimodal_mixins.gemma4.gemma4_vit_mixin import (
    Gemma4ImageEmbedding,
    VideoReader,
    cpu,
)
from rtp_llm.multimodal.multimodal_mixins.gemma4.image_processing_gemma4 import (
    Gemma4ImageProcessor,
)
from rtp_llm.multimodal.multimodal_mixins.gemma4.modeling_gemma4_audio import (
    Gemma4AudioTower,
)
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import get_bytes_io_from_url
from rtp_llm.utils.base_model_datatypes import MMUrlType


class EmbeddingGemma2PromptExpander:
    def __init__(self, ckpt_path):
        self.tokenizer = TokenizerFactory.create(
            ckpt_path, ckpt_path, "embedding_gemma2"
        )

    def expand_rendered_prompt(self, prompt, metadata):
        if not prompt or not metadata:
            return None
        groups = []
        index = 0
        while index < len(metadata):
            first = metadata[index]
            kind = first["kind"]
            count = first["frame_count"]
            frames = metadata[index : index + count]
            if (
                count <= 0
                or len(frames) != count
                or any(
                    frame["kind"] != kind
                    or frame["frame_number"] != number
                    or frame["frame_count"] != count
                    or frame["soft_tokens"] <= 0
                    for number, frame in enumerate(frames)
                )
            ):
                raise ValueError("incomplete EmbeddingGemma2 media expansion metadata")
            if kind not in ("image", "video", "audio"):
                raise ValueError(f"unsupported EmbeddingGemma2 media kind: {kind}")
            start, end = (
                ("<|audio>", "<audio|>")
                if kind == "audio"
                else ("<|image>", "<image|>")
            )
            replacement = "".join(
                start + f"<|{kind}|>" * frame["soft_tokens"] + end for frame in frames
            )
            groups.append((f"<|{kind}|>", replacement))
            index += count
        groups = iter(groups)

        def replace(match):
            try:
                expected, replacement = next(groups)
            except StopIteration as error:
                raise ValueError("more placeholders than media inputs") from error
            if match.group(0) != expected:
                raise ValueError(
                    "EmbeddingGemma2 media order does not match placeholders"
                )
            return replacement

        expanded = re.sub(r"<\|(?:image|video|audio)\|>", replace, prompt)
        if next(groups, None) is not None:
            raise ValueError("fewer placeholders than media inputs")
        return self.tokenizer.encode(expanded)


class EmbeddingGemma2VideoProcessor:
    def __init__(self, config):
        self.max_soft_tokens = config["max_soft_tokens"]
        self.fps = config["fps"]
        self.max_frames = config["max_frames"]
        self.image_processor = Gemma4ImageProcessor(
            patch_size=config["patch_size"],
            pooling_kernel_size=config["pooling_kernel_size"],
            max_soft_tokens=self.max_soft_tokens,
        )

    def sample_frames(self, total, source_fps, duration, fps=None, max_frames=None):
        fps = self.fps if fps is None else fps
        max_frames = self.max_frames if max_frames is None else max_frames
        step = source_fps / fps
        sampled = max(1, int(duration * fps))
        indices = np.array(
            [min(total - 1, int(i * step)) for i in range(sampled)], dtype=int
        )
        if len(indices) > max_frames:
            indices = indices[np.linspace(0, len(indices) - 1, max_frames, dtype=int)]
        return indices.tolist()

    def __call__(self, videos):
        results = []
        for video in videos:
            frames = [
                Image.fromarray(frame.permute(1, 2, 0).cpu().numpy(), "RGB")
                for frame in video
            ]
            results.append(self.image_processor(frames))
        return {
            "pixel_values_videos": torch.cat([r["pixel_values"] for r in results]),
            "video_position_ids": torch.cat([r["image_position_ids"] for r in results]),
            "num_frames_per_video": [len(video) for video in videos],
            "num_soft_tokens_per_video": [
                r["num_soft_tokens_per_image"][0] for r in results
            ],
        }


class EmbeddingGemma2AudioProjector(nn.Module):
    def __init__(self, config, hidden_size):
        super().__init__()
        self.eps = config["rms_norm_eps"]
        self.embedding_projection = nn.Linear(
            config["output_proj_dims"], hidden_size, bias=False
        )

    def forward(self, states):
        return self.embedding_projection(gemma4_rms_norm(states, None, self.eps))


class EmbeddingGemma2AudioEmbedding(nn.Module):
    def __init__(self, config, hidden_size, ckpt_path):
        super().__init__()
        self.feature_extractor = Gemma4AudioFeatureExtractor.from_pretrained(ckpt_path)
        self.audio_tower = Gemma4AudioTower(config)
        self.embed_audio = EmbeddingGemma2AudioProjector(config, hidden_size)

    def forward(self, data):
        device = self.audio_tower.output_proj.weight.device
        states, mask = self.audio_tower(data[0].to(device), data[1].to(device))
        projected = self.embed_audio(states)
        return (
            projected[mask],
            None,
            None,
            [
                {
                    "kind": "audio",
                    "frame_number": 0,
                    "frame_count": 1,
                    "soft_tokens": int(mask.sum()),
                }
            ],
        )


class EmbeddingGemma2MediaEmbedding(Gemma4ImageEmbedding):
    def __init__(self, params):
        super().__init__(params)
        ckpt_path = params.config["ckpt_path"]
        with open(os.path.join(ckpt_path, "config.json")) as reader:
            config = json.load(reader)
        with open(os.path.join(ckpt_path, "processor_config.json")) as reader:
            processor = json.load(reader)
        self.video_processor = EmbeddingGemma2VideoProcessor(
            processor["video_processor"]
        )
        self.prompt_expander = EmbeddingGemma2PromptExpander(ckpt_path)
        self.audio = EmbeddingGemma2AudioEmbedding(
            config["audio_config"], config["text_config"]["hidden_size"], ckpt_path
        )

    def get_preprocess_params(self):
        return {
            **super().get_preprocess_params(),
            "feature_extractor": self.audio.feature_extractor,
        }

    @staticmethod
    def preprocess_input(
        mm_inputs, vit_config, processor, video_processor, feature_extractor
    ):
        item = mm_inputs[0]
        if item.mm_type == MMUrlType.AUDIO:
            audio_bytes = get_bytes_io_from_url(
                item.url,
                vit_config.download_headers,
                max_file_size_kb=vit_config.mm_image_max_file_size_kb,
            )
            mono, _ = librosa.load(
                audio_bytes, sr=feature_extractor.sampling_rate, mono=True
            )
            if mono.size > 480000:
                raise FtRuntimeException(
                    ExceptionType.MM_LONG_PROMPT_ERROR,
                    "EmbeddingGemma2 audio exceeds the 30-second input budget",
                )
            data = feature_extractor([mono], return_tensors="pt", truncation=False)
            return (
                data["input_features"],
                data["input_features_mask"],
                None,
                {"kind": "audio"},
            )
        if item.mm_type == MMUrlType.VIDEO:
            if VideoReader is None:
                raise ImportError(
                    "decord is required for EmbeddingGemma2 video decoding"
                )
            data = get_bytes_io_from_url(
                item.url,
                vit_config.download_headers,
                max_file_size_kb=vit_config.mm_video_max_file_size_kb,
            )
            reader = VideoReader(data, ctx=cpu(0), num_threads=1)
            source_fps = float(reader.get_avg_fps())
            if source_fps <= 0:
                raise ValueError("video source FPS must be positive")
            options = item.mm_preprocess_config
            indices = video_processor.sample_frames(
                len(reader),
                source_fps,
                len(reader) / source_fps,
                fps=options.fps if options.fps > 0 else None,
                max_frames=options.max_frames if options.max_frames > 0 else None,
            )
            frames = torch.from_numpy(reader.get_batch(indices).asnumpy()).permute(
                0, 3, 1, 2
            )
            data = video_processor([frames])
            metadata = [
                {
                    "kind": "video",
                    "fps": source_fps,
                    "frame_index": frame_index,
                    "frame_number": number,
                    "frame_count": len(indices),
                    "soft_tokens": data["num_soft_tokens_per_video"][0],
                }
                for number, frame_index in enumerate(indices)
            ]
            return data["pixel_values_videos"], data["video_position_ids"], metadata
        return Gemma4ImageEmbedding.preprocess_input(
            mm_inputs, vit_config, processor, video_processor
        )

    def embedding(self, data, **kwargs):
        if kwargs.get("mm_type") == MMUrlType.AUDIO or (
            len(data) > 3 and data[3]["kind"] == "audio"
        ):
            return self.audio(data)
        return super().embedding(data, **kwargs)
