"""MiniMax-M3 vision tower, multi-modal projector and patch-merge MLP.

Module names are chosen to match the HF checkpoint paths exactly:

  vision_tower.vision_model.embeddings.patch_embedding.weight   Conv3d [1280,3,2,14,14]
  vision_tower.vision_model.pre_layrnorm.weight/bias            LayerNorm [1280]
  vision_tower.vision_model.encoder.layers.{i}.layer_norm1/2    LayerNorm [1280]
  vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj Linear [1280,1280]
  vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj Linear [1280,1280]
  vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj Linear [1280,1280]
  vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj Linear [1280,1280]
  vision_tower.vision_model.encoder.layers.{i}.mlp.fc1          Linear [5120,1280]
  vision_tower.vision_model.encoder.layers.{i}.mlp.fc2          Linear [1280,5120]

  multi_modal_projector.linear_1  Linear [6144,1280]
  multi_modal_projector.linear_2  Linear [6144,6144]

  patch_merge_mlp.linear_1        Linear [6144,24576]
  patch_merge_mlp.linear_2        Linear [6144,6144]
"""

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.models_py.module_base import RtpModule

# ---------------------------------------------------------------------------
# Vision Encoder sub-modules
# ---------------------------------------------------------------------------


class MiniMaxVisionMLP(RtpModule):
    """Two-layer MLP used inside each encoder layer."""

    def __init__(
        self, hidden_size: int, intermediate_size: int, params_dtype: torch.dtype
    ):
        super().__init__()
        self.fc1 = nn.Linear(
            hidden_size, intermediate_size, bias=True, dtype=params_dtype
        )
        self.fc2 = nn.Linear(
            intermediate_size, hidden_size, bias=True, dtype=params_dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        return self.fc2(x)


class MiniMaxVisionAttention(RtpModule):
    """Multi-head self-attention with separate q/k/v/out projections."""

    def __init__(self, hidden_size: int, num_heads: int, params_dtype: torch.dtype):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True, dtype=params_dtype)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True, dtype=params_dtype)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True, dtype=params_dtype)
        self.out_proj = nn.Linear(
            hidden_size, hidden_size, bias=True, dtype=params_dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[0]
        q = (
            self.q_proj(x)
            .reshape(seq_len, self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        k = (
            self.k_proj(x)
            .reshape(seq_len, self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        v = (
            self.v_proj(x)
            .reshape(seq_len, self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        # q/k/v: [num_heads, seq_len, head_dim]
        q = q.unsqueeze(0)  # [1, num_heads, seq_len, head_dim]
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.squeeze(0).transpose(0, 1).reshape(seq_len, -1)
        return self.out_proj(out)


class MiniMaxVisionEncoderLayer(RtpModule):
    """Single encoder block: pre-LN attention + pre-LN MLP with residual."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
    ):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size, dtype=params_dtype)
        self.self_attn = MiniMaxVisionAttention(hidden_size, num_heads, params_dtype)
        self.layer_norm2 = nn.LayerNorm(hidden_size, dtype=params_dtype)
        self.mlp = MiniMaxVisionMLP(hidden_size, intermediate_size, params_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class MiniMaxVisionEncoder(RtpModule):
    """Stack of ``num_layers`` encoder layers."""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MiniMaxVisionEncoderLayer(
                    hidden_size, num_heads, intermediate_size, params_dtype
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Vision Embeddings
# ---------------------------------------------------------------------------


class MiniMaxVisionEmbeddings(RtpModule):
    """3-D patch embedding via Conv3d (no bias, matching checkpoint)."""

    def __init__(
        self,
        hidden_size: int,
        temporal_patch_size: int,
        patch_size: int,
        params_dtype: torch.dtype,
    ):
        super().__init__()
        self.patch_embedding = nn.Conv3d(
            in_channels=3,
            out_channels=hidden_size,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size),
            bias=False,
            dtype=params_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, T, H, W]  ->  [N, hidden_size, t, h, w]
        return self.patch_embedding(x)


# ---------------------------------------------------------------------------
# Vision Model (embeddings + pre-LN + encoder)
# ---------------------------------------------------------------------------


class MiniMaxVisionModel(RtpModule):
    """Inner vision model whose attribute tree mirrors the checkpoint exactly."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        temporal_patch_size: int,
        patch_size: int,
        layer_norm_eps: float,
        params_dtype: torch.dtype,
    ):
        super().__init__()
        self.embeddings = MiniMaxVisionEmbeddings(
            hidden_size=hidden_size,
            temporal_patch_size=temporal_patch_size,
            patch_size=patch_size,
            params_dtype=params_dtype,
        )
        # NOTE: "pre_layrnorm" is an intentional typo to match the HF checkpoint key.
        self.pre_layrnorm = nn.LayerNorm(
            hidden_size, eps=layer_norm_eps, dtype=params_dtype
        )
        self.encoder = MiniMaxVisionEncoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            params_dtype=params_dtype,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.embeddings.patch_embedding(pixel_values)  # [N, C, t, h, w]
        # Flatten spatial dims: [N, C, t*h*w] -> [N, t*h*w, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.pre_layrnorm(x)
        # Flatten batch into sequence: [N*T, hidden]
        x = x.reshape(-1, x.shape[-1])
        x = self.encoder(x)
        return x


# ---------------------------------------------------------------------------
# Top-level Vision Transformer
# ---------------------------------------------------------------------------


class MiniMaxVisionTransformer(RtpModule):
    """Wraps ``MiniMaxVisionModel`` and reads config from ``model_config``."""

    # Default CLIP-ViT-H/14 parameters used by MiniMax-M3.
    _DEFAULTS = {
        "hidden_size": 1280,
        "num_attention_heads": 16,
        "num_hidden_layers": 32,
        "intermediate_size": 5120,
        "patch_size": 14,
        "temporal_patch_size": 2,
        "layer_norm_eps": 1e-5,
    }

    def __init__(self, model_config: Any, load_config: Any):
        super().__init__()
        vcfg = self._get_vision_config(model_config)
        hidden_size = vcfg.get("hidden_size", self._DEFAULTS["hidden_size"])
        num_heads = vcfg.get(
            "num_attention_heads", self._DEFAULTS["num_attention_heads"]
        )
        num_layers = vcfg.get("num_hidden_layers", self._DEFAULTS["num_hidden_layers"])
        intermediate_size = vcfg.get(
            "intermediate_size", self._DEFAULTS["intermediate_size"]
        )
        patch_size = vcfg.get("patch_size", self._DEFAULTS["patch_size"])
        temporal_patch_size = vcfg.get(
            "temporal_patch_size", self._DEFAULTS["temporal_patch_size"]
        )
        layer_norm_eps = vcfg.get("layer_norm_eps", self._DEFAULTS["layer_norm_eps"])

        params_dtype = (
            getattr(load_config, "compute_dtype", torch.bfloat16) or torch.bfloat16
        )

        self.vision_model = MiniMaxVisionModel(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            temporal_patch_size=temporal_patch_size,
            patch_size=patch_size,
            layer_norm_eps=layer_norm_eps,
            params_dtype=params_dtype,
        )

    @staticmethod
    def _get_vision_config(model_config) -> dict:
        mm = getattr(model_config, "mm_related_params", None)
        if mm is not None and getattr(mm, "config", None):
            return mm.config
        if hasattr(model_config, "vision_config"):
            vc = model_config.vision_config
            return vc if isinstance(vc, dict) else vars(vc)
        if isinstance(model_config, dict):
            return model_config.get("vision_config", {})
        return {}

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_model(pixel_values)


# ---------------------------------------------------------------------------
# Multi-modal projector
# ---------------------------------------------------------------------------


class MiniMaxProjector(RtpModule):
    """Two-layer GELU projector: vision_hidden (1280) -> language_hidden (6144)."""

    def __init__(self, model_config: Any, load_config: Any):
        super().__init__()
        vcfg = MiniMaxVisionTransformer._get_vision_config(model_config)
        vision_hidden = vcfg.get(
            "hidden_size", MiniMaxVisionTransformer._DEFAULTS["hidden_size"]
        )
        # language hidden size – fall back to 6144
        lang_hidden = getattr(model_config, "hidden_size", 6144) or 6144
        params_dtype = (
            getattr(load_config, "compute_dtype", torch.bfloat16) or torch.bfloat16
        )
        self.linear_1 = nn.Linear(
            vision_hidden, lang_hidden, bias=True, dtype=params_dtype
        )
        self.linear_2 = nn.Linear(
            lang_hidden, lang_hidden, bias=True, dtype=params_dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.linear_2(x)
        return x


# ---------------------------------------------------------------------------
# Patch-merge MLP
# ---------------------------------------------------------------------------


class MiniMaxPatchMergeMLP(RtpModule):
    """Two-layer GELU MLP that merges spatial patch tokens before projection.

    Input dim 24576 = vision_hidden(1280) * spatial_merge_size^2 * temporal_factor
    (exact factor depends on the merging strategy; hard-coded to match checkpoint).
    """

    def __init__(self, model_config: Any, load_config: Any):
        super().__init__()
        lang_hidden = getattr(model_config, "hidden_size", 6144) or 6144
        # 24576 is fixed by the checkpoint (1280 * 4 * temporal? or spatial merge).
        merge_input_dim = 24576
        params_dtype = (
            getattr(load_config, "compute_dtype", torch.bfloat16) or torch.bfloat16
        )
        self.linear_1 = nn.Linear(
            merge_input_dim, lang_hidden, bias=True, dtype=params_dtype
        )
        self.linear_2 = nn.Linear(
            lang_hidden, lang_hidden, bias=True, dtype=params_dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.linear_2(x)
        return x
