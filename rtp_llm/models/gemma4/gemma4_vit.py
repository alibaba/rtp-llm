"""Gemma4 Vision Encoder (SigLIP-based with RMSNorm, gated MLP, QK norm).

Architecture:
- Patch embedding: flatten + linear (not Conv2d)
- 2D position embedding table [2, max_patches, hidden_size]
- 27 transformer layers with RMSNorm, QK-norm, gated MLP (gate/up/down)
- Standardization via learned std_bias/std_scale
- Projector: AvgPool2d + RMSNorm + Linear (1152 → 5376)
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from rtp_llm.config.model_config import VitParameters
from rtp_llm.models.multimodal.multimodal_common import ImageEmbeddingInterface


class Gemma4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gemma-style: weight is (1 + w), not just w
        x_float = x.float()
        variance = x_float.pow(2).mean(-1, keepdim=True)
        x_normed = x_float * torch.rsqrt(variance + self.eps)
        return (x_normed * (1.0 + self.weight.float())).to(x.dtype)


class _LinearWrapper(nn.Module):
    """Wraps nn.Linear as .linear sub-module to match checkpoint naming convention.

    Checkpoint has e.g. 'q_proj.linear.weight', not 'q_proj.weight'.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class Gemma4VisionMLP(nn.Module):
    """Gated MLP: gate_proj * up_proj → down_proj"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = _LinearWrapper(hidden_size, intermediate_size)
        self.up_proj = _LinearWrapper(hidden_size, intermediate_size)
        self.down_proj = _LinearWrapper(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class Gemma4VisionAttention(nn.Module):
    """Multi-head attention with QK normalization."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = _LinearWrapper(hidden_size, num_heads * head_dim)
        self.k_proj = _LinearWrapper(hidden_size, num_heads * head_dim)
        self.v_proj = _LinearWrapper(hidden_size, num_heads * head_dim)
        self.o_proj = _LinearWrapper(num_heads * head_dim, hidden_size)
        self.q_norm = Gemma4RMSNorm(head_dim)
        self.k_norm = Gemma4RMSNorm(head_dim)
        self.scale = head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # QK normalization (applied per-head)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Scaled dot-product attention
        attn = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        attn = attn.transpose(1, 2).reshape(B, N, -1)
        return self.o_proj(attn)


class Gemma4VisionEncoderLayer(nn.Module):
    """Single vision encoder layer with 4 RMSNorms."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int,
                 intermediate_size: int, eps: float = 1e-6):
        super().__init__()
        self.input_layernorm = Gemma4RMSNorm(hidden_size, eps)
        self.self_attn = Gemma4VisionAttention(hidden_size, num_heads, head_dim)
        self.post_attention_layernorm = Gemma4RMSNorm(hidden_size, eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(hidden_size, eps)
        self.mlp = Gemma4VisionMLP(hidden_size, intermediate_size)
        self.post_feedforward_layernorm = Gemma4RMSNorm(hidden_size, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = self.post_attention_layernorm(x)
        x = residual + x

        # FFN block
        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x

        return x


class Gemma4PatchEmbedder(nn.Module):
    """Flatten-then-project patch embedding (not Conv2d).

    input_proj: Linear(3*patch_size*patch_size, hidden_size)
    position_embedding_table: [2, max_patches, hidden_size] (2D learned positions)
    """

    def __init__(self, hidden_size: int, patch_size: int, position_embedding_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.input_proj = nn.Linear(3 * patch_size * patch_size, hidden_size, bias=False)
        # 2D position embedding: [2, max_patches, hidden_size]
        # dim 0: height positions, dim 1: width positions
        self.position_embedding_table = nn.Parameter(
            torch.zeros(2, position_embedding_size, hidden_size)
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, C, H, W] — already standardized
        Returns:
            [B, num_patches, hidden_size]
        """
        B, C, H, W = pixel_values.shape
        pH = H // self.patch_size
        pW = W // self.patch_size

        # Flatten patches: [B, C, pH, patch, pW, patch] → [B, pH*pW, C*patch*patch]
        x = pixel_values.reshape(B, C, pH, self.patch_size, pW, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)  # [B, pH, pW, C, patch, patch]
        x = x.reshape(B, pH * pW, -1)  # [B, num_patches, C*patch_size*patch_size]

        # Project
        x = self.input_proj(x)

        # Add 2D position embeddings
        h_pos = self.position_embedding_table[0, :pH, :]  # [pH, hidden]
        w_pos = self.position_embedding_table[1, :pW, :]  # [pW, hidden]
        # Broadcast: each patch gets h_pos[row] + w_pos[col]
        pos = h_pos.unsqueeze(1) + w_pos.unsqueeze(0)  # [pH, pW, hidden]
        pos = pos.reshape(pH * pW, -1).unsqueeze(0)  # [1, num_patches, hidden]
        x = x + pos

        return x


class Gemma4VisionTower(nn.Module):
    """Complete vision tower: standardization → patch embed → encoder → pool → project."""

    def __init__(self, config: dict):
        super().__init__()
        hidden_size = config["hidden_size"]
        num_layers = config["num_hidden_layers"]
        num_heads = config["num_attention_heads"]
        head_dim = config.get("head_dim", hidden_size // num_heads)
        intermediate_size = config["intermediate_size"]
        patch_size = config["patch_size"]
        eps = config.get("rms_norm_eps", 1e-6)
        position_embedding_size = config.get("position_embedding_size", 10240)
        pooling_kernel_size = config.get("pooling_kernel_size", 3)
        default_output_length = config.get("default_output_length", 280)

        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.pooling_kernel_size = pooling_kernel_size
        self.default_output_length = default_output_length
        self.standardize = config.get("standardize", True)

        # Standardization params (learned)
        self.std_scale = nn.Parameter(torch.ones(hidden_size))
        self.std_bias = nn.Parameter(torch.zeros(hidden_size))

        # Patch embedder
        self.patch_embedder = Gemma4PatchEmbedder(hidden_size, patch_size, position_embedding_size)

        # Encoder
        self.encoder = nn.ModuleDict({
            "layers": nn.ModuleList([
                Gemma4VisionEncoderLayer(hidden_size, num_heads, head_dim, intermediate_size, eps)
                for _ in range(num_layers)
            ])
        })

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, C, H, W] normalized to [0, 1]
        Returns:
            [B, num_output_tokens, hidden_size] — pooled vision features
        """
        B, C, H, W = pixel_values.shape

        # Patch embedding
        x = self.patch_embedder(pixel_values)  # [B, num_patches, hidden]

        # Standardize patch embeddings
        if self.standardize:
            x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
            x = x * (1.0 + self.std_scale) + self.std_bias

        # Encoder layers
        for layer in self.encoder.layers:
            x = layer(x)

        # Pool: reshape to 2D grid and apply AvgPool2d
        pH = H // self.patch_size
        pW = W // self.patch_size
        x = x.reshape(B, pH, pW, self.hidden_size)
        x = x.permute(0, 3, 1, 2)  # [B, hidden, pH, pW]
        x = F.avg_pool2d(x, kernel_size=self.pooling_kernel_size, stride=self.pooling_kernel_size)
        x = x.permute(0, 2, 3, 1)  # [B, pH', pW', hidden]
        x = x.reshape(B, -1, self.hidden_size)  # [B, num_output_tokens, hidden]

        return x


class Gemma4MultiModalProjector(nn.Module):
    """Projects vision features to LLM hidden space.

    Checkpoint weight: model.embed_vision.embedding_projection.weight [text_hidden, vision_hidden]
    Note: Gemma4 checkpoint does NOT include a separate norm for the projector
    (unlike Gemma3 which has mm_soft_emb_norm). The projection is just a linear.
    """

    def __init__(self, vision_hidden_size: int, text_hidden_size: int):
        super().__init__()
        self.embedding_projection = nn.Linear(vision_hidden_size, text_hidden_size, bias=False)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: [B, num_tokens, vision_hidden_size]
        Returns:
            [B, num_tokens, text_hidden_size]
        """
        return self.embedding_projection(vision_features)


class Gemma4ImageEmbedding(ImageEmbeddingInterface):
    """Wraps vision tower + projector for RTP-LLM multimodal pipeline."""

    def __init__(self, mm_related_params: VitParameters, model_config=None):
        self.mm_related_params = mm_related_params
        vision_config = mm_related_params.config
        text_hidden_size = model_config.hidden_size if model_config else 5376

        self.vision_tower = Gemma4VisionTower(vision_config)
        # Named "embed_vision" to match checkpoint weight prefix: model.embed_vision.*
        self.embed_vision = Gemma4MultiModalProjector(
            vision_config["hidden_size"],
            text_hidden_size,
        )
        self.config = model_config
        self.patch_size = vision_config["patch_size"]

    @property
    def _data_type(self):
        return self.config.compute_dtype if self.config else torch.bfloat16

    @property
    def _device(self):
        return next(self.vision_tower.parameters()).device

    @torch.inference_mode()
    def image_embedding(self, images: List[Image.Image]) -> List[torch.Tensor]:
        """Process images and return projected features.

        Returns list of [num_tokens, text_hidden_size] tensors, one per image.
        """
        from torchvision import transforms

        results = []
        for img in images:
            # Ensure RGB
            img = img.convert("RGB")

            # Resize to multiple of patch_size
            w, h = img.size
            # Default: resize so that total patches fit within position_embedding_size
            new_h = max(self.patch_size, (h // self.patch_size) * self.patch_size)
            new_w = max(self.patch_size, (w // self.patch_size) * self.patch_size)
            # Cap at reasonable size (56 patches per side = 896 pixels)
            max_patches_per_side = 56
            if new_h > max_patches_per_side * self.patch_size:
                new_h = max_patches_per_side * self.patch_size
            if new_w > max_patches_per_side * self.patch_size:
                new_w = max_patches_per_side * self.patch_size

            transform = transforms.Compose([
                transforms.Resize((new_h, new_w)),
                transforms.ToTensor(),  # [0, 1]
            ])
            pixel_values = transform(img).unsqueeze(0)  # [1, 3, H, W]
            pixel_values = pixel_values.to(device=self._device, dtype=self._data_type)

            # Vision tower + projector
            vision_features = self.vision_tower(pixel_values)  # [1, N, vision_hidden]
            projected = self.embed_vision(vision_features)  # [1, N, text_hidden]
            results.append(projected.squeeze(0))

        return results
