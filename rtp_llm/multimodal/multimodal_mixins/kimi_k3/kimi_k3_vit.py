"""Kimi-K3 MoonViT configuration and assembly."""

import json
import math
import os
import threading
from io import BytesIO
from typing import Any, List, Optional

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoTokenizer
from transformers.configuration_utils import PretrainedConfig

from rtp_llm.multimodal.mm_error_messages import raise_mm
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_image_processor import (
    KimiK3VisionProcessor,
)
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_moonvit import (
    MoonViT3dPretrainedModel,
)
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    ImageEmbeddingInterface,
)
from rtp_llm.multimodal.multimodal_util import MMUrlType, get_bytes_io_from_url

# Serialize MoonViT forwards; the vision tower is shared across the media
# preprocessing thread pool and is not safe to run concurrently.
mm_lock = threading.Lock()


class KimiK3VisionConfig(PretrainedConfig):
    """Vision-only subset of the K3 checkpoint configuration."""

    model_type = "kimi_k3_vision"

    def __init__(
        self,
        vt_hidden_size: int = 1024,
        vt_intermediate_size: int = 4096,
        vt_num_hidden_layers: int = 27,
        vt_num_attention_heads: int = 12,
        qkv_hidden_size: int = 1536,
        patch_size: int = 14,
        num_channels: int = 3,
        merge_kernel_size=(2, 2),
        merge_type: str = "sd2_tpool",
        mm_projector_type: str = "patchmergerv2",
        mm_hidden_size: int | None = None,
        text_hidden_size: int = 7168,
        projector_hidden_act: str = "gelu",
        projector_ln_eps: float = 1e-5,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        init_pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        pos_emb_interpolation_mode: str = "bilinear",
        video_attn_type: str = "spatial_temporal",
        norm_type: str = "rmsnorm",
        mlp_type: str = "mlp2",
        attn_bias: bool = False,
        linear_bias: bool = False,
        patch_embed_proj_bias: bool = False,
        rope_theta: float = 10000.0,
        max_pos_emb_height: int = 512,
        max_pos_emb_width: int = 512,
        **kwargs: Any,
    ) -> None:
        self.vt_hidden_size = vt_hidden_size
        self.vt_intermediate_size = vt_intermediate_size
        self.vt_num_hidden_layers = vt_num_hidden_layers
        self.vt_num_attention_heads = vt_num_attention_heads
        self.qkv_hidden_size = qkv_hidden_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.merge_kernel_size = (
            [merge_kernel_size, merge_kernel_size]
            if isinstance(merge_kernel_size, int)
            else list(merge_kernel_size)
        )
        self.merge_type = merge_type
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = (
            mm_hidden_size if mm_hidden_size is not None else vt_hidden_size
        )
        self.text_hidden_size = text_hidden_size
        self.projector_hidden_act = projector_hidden_act
        self.projector_ln_eps = projector_ln_eps
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.init_pos_emb_time = init_pos_emb_time
        self.pos_emb_type = pos_emb_type
        self.pos_emb_interpolation_mode = pos_emb_interpolation_mode
        self.video_attn_type = video_attn_type
        self.norm_type = norm_type
        self.mlp_type = mlp_type
        self.attn_bias = attn_bias
        self.linear_bias = linear_bias
        self.patch_embed_proj_bias = patch_embed_proj_bias
        self.rope_theta = rope_theta
        self.max_pos_emb_height = max_pos_emb_height
        self.max_pos_emb_width = max_pos_emb_width
        super().__init__(**kwargs)


class KimiK3PatchMergerMLPV2(nn.Module):
    """K3 patch merger matching mm_projector checkpoint names."""

    def __init__(self, config: KimiK3VisionConfig) -> None:
        super().__init__()
        merge_h, merge_w = config.merge_kernel_size
        self.hidden_size = config.mm_hidden_size * merge_h * merge_w
        if config.mm_projector_type != "patchmergerv2":
            raise NotImplementedError(
                f"mm_projector_type={config.mm_projector_type} not supported"
            )
        if config.projector_hidden_act != "gelu":
            raise ValueError(
                "KimiK3PatchMergerMLPV2 only supports projector_hidden_act='gelu', "
                f"got {config.projector_hidden_act!r}"
            )
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(self.hidden_size, config.text_hidden_size, bias=False),
        )
        self.post_norm = nn.RMSNorm(
            config.text_hidden_size, eps=config.projector_ln_eps
        )
        for module in self.proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(
                    module.weight, std=math.sqrt(2 / module.in_features)
                )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden = image_features.flatten(start_dim=-2)
        return self.post_norm(self.proj(hidden))


@torch.inference_mode()
def mm_projector_forward(
    mm_projector: KimiK3PatchMergerMLPV2,
    vision_outputs: List[torch.Tensor],
) -> List[torch.Tensor]:
    if len(vision_outputs) == 1:
        return [mm_projector(vision_outputs[0])]
    lengths = [output.shape[0] for output in vision_outputs]
    batched = torch.cat(vision_outputs, dim=0)
    projected = mm_projector(batched)
    return list(torch.split(projected, lengths, dim=0))


class KimiK3ImageEmbedding(ImageEmbeddingInterface):
    """K3 MoonViT and projector exposed through RTP-LLM's image interface."""

    def __init__(self, mm_related_params) -> None:
        config = mm_related_params.config or {}
        self.vision_config = KimiK3VisionConfig(
            **(config.get("vision_config", {}) or {})
        )
        self.vision_tower = MoonViT3dPretrainedModel(self.vision_config)
        self.mm_projector = KimiK3PatchMergerMLPV2(self.vision_config)
        self.image_processor = KimiK3VisionProcessor(config["media_proc_cfg"])
        self._ckpt_path = config.get("ckpt_path")
        self._tokenizer = None
        self._word_embedding_weight = None

    @property
    def _device(self):
        return self.vision_tower.patch_embed.proj.weight.device

    @property
    def _data_type(self):
        return self.vision_tower.patch_embed.proj.weight.dtype

    @staticmethod
    def preprocess_input(mm_inputs, vit_config, **kwargs):
        assert len(mm_inputs) == 1
        mm_input = mm_inputs[0]
        if mm_input.mm_type not in (MMUrlType.DEFAULT, MMUrlType.IMAGE):
            raise ValueError("Kimi-K3 only supports image multimodal inputs")
        if mm_input.tensor.numel() > 0:
            if mm_input.tensor.dtype != torch.uint8 or mm_input.tensor.ndim != 1:
                raise ValueError("Kimi-K3 image tensor must be a 1-D uint8 tensor")
            # Direct model RPC callers can supply bytes rather than a URL, so
            # enforce the same per-image cap on both input forms.
            if (
                vit_config.mm_image_max_file_size_kb > 0
                and mm_input.tensor.numel()
                > vit_config.mm_image_max_file_size_kb * 1024
            ):
                raise ValueError(
                    "Kimi K3 image bytes exceed the per-image limit: "
                    f"{mm_input.tensor.numel()} > {vit_config.mm_image_max_file_size_kb * 1024}"
                )
            # memoryview, not .tobytes(): BytesIO copies its initializer anyway.
            data = BytesIO(mm_input.tensor.detach().cpu().contiguous().numpy().data)
        else:
            data = get_bytes_io_from_url(
                mm_input.url,
                vit_config.download_headers,
                max_file_size_kb=vit_config.mm_image_max_file_size_kb,
            )
            # The URL cache can return a shared BytesIO to parallel requests.
            # Decode from a private cursor so one request cannot seek another.
            data = BytesIO(data.getbuffer())
        try:
            with Image.open(data) as image:
                if image.format in ("HEIF", "HEIC"):
                    return Image.frombytes(image.mode, image.size, image.tobytes())
                return image.copy()
        except Exception:
            raise_mm("Failed to open Kimi K3 image")

    @torch.inference_mode()
    def image_embedding(self, images: List[Image.Image]) -> List[torch.Tensor]:
        medias = [{"type": "image", "image": image} for image in images]
        processed = self.image_processor.preprocess(medias, return_tensors="pt")
        pixel_values = processed["pixel_values"]
        if self._device.type == "cuda":
            staged = torch.empty(
                pixel_values.shape,
                dtype=self._data_type,
                pin_memory=True,
            )
            staged.copy_(pixel_values)
            pixel_values = staged.to(device=self._device, non_blocking=True)
        else:
            pixel_values = pixel_values.to(device=self._device, dtype=self._data_type)
        # Shape metadata stays on CPU so Python consumers never synchronize CUDA.
        grid_thws = processed["grid_thws"]
        vision_outputs = self.vision_tower(pixel_values, grid_thws)
        return mm_projector_forward(self.mm_projector, vision_outputs)

    def _ensure_text_embeddings(self) -> None:
        if self._word_embedding_weight is not None:
            return
        if not self._ckpt_path:
            raise ValueError("Kimi-K3 ViT needs ckpt_path for image prompt embeddings")

        from safetensors import safe_open

        key = "language_model.model.embed_tokens.weight"
        index_path = os.path.join(self._ckpt_path, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            with open(index_path, encoding="utf-8") as file:
                weight_map = json.load(file)["weight_map"]
            shard_path = os.path.join(self._ckpt_path, weight_map[key])
        else:
            shard_path = os.path.join(self._ckpt_path, "model.safetensors")

        tokenizer = AutoTokenizer.from_pretrained(
            self._ckpt_path, trust_remote_code=True, verbose=False, use_fast=True
        )
        with safe_open(shard_path, framework="pt", device="cpu") as file:
            word_embedding_weight = file.get_tensor(key)
        if (
            word_embedding_weight.ndim != 2
            or word_embedding_weight.shape[1] != self.vision_config.text_hidden_size
        ):
            raise ValueError("Kimi-K3 ViT and text embedding hidden sizes disagree")
        self._tokenizer = tokenizer
        self._word_embedding_weight = word_embedding_weight

    def _assemble_image(
        self, image: Image.Image, vision_features: torch.Tensor
    ) -> torch.Tensor:
        """Reproduce the original full K3 image prompt in embedding space."""
        self._ensure_text_embeddings()
        prompt = self.image_processor.make_image_prompt(*image.size)
        token_ids = list(self._tokenizer.encode(prompt))
        pad_ids = list(self._tokenizer.encode("<|media_pad|>"))
        if len(pad_ids) != 1 or token_ids.count(pad_ids[0]) != 1:
            raise ValueError("Kimi-K3 image prompt must contain one media pad token")
        pad_index = token_ids.index(pad_ids[0])

        def text_embedding(ids: List[int]) -> torch.Tensor:
            if not ids:
                return vision_features.new_empty((0, vision_features.shape[1]))
            return self._word_embedding_weight[ids].to(
                device=vision_features.device, dtype=vision_features.dtype
            )

        return torch.cat(
            [
                text_embedding(token_ids[:pad_index]),
                vision_features,
                text_embedding(token_ids[pad_index + 1 :]),
            ],
            dim=0,
        ).contiguous()

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        """Single-image entry used by the multimodal processing engine."""
        with mm_lock:
            self._ensure_text_embeddings()
            vision_features = self.image_embedding([data])[0].to(self._data_type)
            features = self._assemble_image(data, vision_features)
        return features, None

    @torch.inference_mode()
    def batched_embedding(self, data_list, mm_types, **kwargs):
        """Batched entry: run the vision tower once over the whole batch."""
        del mm_types  # K3 only supports images; type is validated in preprocess.
        with mm_lock:
            self._ensure_text_embeddings()
            embeddings = [
                self._assemble_image(image, features.to(self._data_type))
                for image, features in zip(data_list, self.image_embedding(data_list))
            ]
        return [(embedding, None) for embedding in embeddings]
