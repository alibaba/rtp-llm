"""Kimi-K3 MoonViT configuration and assembly."""

import math
import threading
from io import BytesIO
from typing import Any, List

import torch
import torch.nn as nn
from PIL import Image
from transformers.configuration_utils import PretrainedConfig

from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    ImageEmbeddingInterface,
)
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_image_processor import (
    K3_MAX_IMAGE_FILE_SIZE_KB,
    K3_MAX_IMAGE_PIXELS,
    KimiK3VisionProcessor,
)
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_moonvit import (
    MoonViT3dPretrainedModel,
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
        self.image_processor = KimiK3VisionProcessor()

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
            # Third byte entry point: a direct model RPC call skips the renderer
            # preflight, so the shared per-image cap has to be enforced here too.
            if mm_input.tensor.numel() > K3_MAX_IMAGE_FILE_SIZE_KB * 1024:
                raise ValueError(
                    "Kimi K3 image bytes exceed the per-image limit: "
                    f"{mm_input.tensor.numel()} > {K3_MAX_IMAGE_FILE_SIZE_KB * 1024}"
                )
            # memoryview, not .tobytes(): BytesIO copies its initializer anyway.
            data = BytesIO(mm_input.tensor.detach().cpu().contiguous().numpy().data)
        else:
            data = get_bytes_io_from_url(
                mm_input.url,
                vit_config.download_headers,
                max_file_size_kb=K3_MAX_IMAGE_FILE_SIZE_KB,
            )
        with Image.open(data) as image:
            # image.copy() is the only full decode on any K3 path; PIL's own 89 MP
            # default would let a tiny header claim 256 MB of pixels.
            width, height = image.size
            if width * height > K3_MAX_IMAGE_PIXELS:
                raise ValueError(
                    "Kimi K3 image pixel count exceeds the per-image limit: "
                    f"{width}x{height} > {K3_MAX_IMAGE_PIXELS}"
                )
            return image.copy()

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
            pixel_values = pixel_values.to(
                device=self._device, dtype=self._data_type
            )
        # Shape metadata stays on CPU so Python consumers never synchronize CUDA.
        grid_thws = processed["grid_thws"]
        vision_outputs = self.vision_tower(pixel_values, grid_thws)
        return mm_projector_forward(self.mm_projector, vision_outputs)

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        """Single-image entry used by the multimodal processing engine."""
        with mm_lock:
            features = (
                self.image_embedding([data])[0].to(self._data_type).contiguous()
            )
        return features, None

    @torch.inference_mode()
    def batched_embedding(self, data_list, mm_types, **kwargs):
        """Batched entry: run the vision tower once over the whole batch."""
        del mm_types  # K3 only supports images; type is validated in preprocess.
        with mm_lock:
            embeddings = [
                embedding.to(self._data_type).contiguous()
                for embedding in self.image_embedding(data_list)
            ]
        return [(embedding, None) for embedding in embeddings]
