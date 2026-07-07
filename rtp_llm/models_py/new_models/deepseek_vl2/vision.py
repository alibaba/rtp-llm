"""DeepSeek VL V2 vision tower for the new loader.

DeepSeek VL V2 的视觉塔由三部分组成：
  1. ``vision`` — SigLIP 视觉编码器（timm ``vit_so400m_patch14_siglip_384.webli``）
  2. ``projector`` — MlpProjector（将视觉特征投影到语言模型维度）
  3. 额外参数 ``image_newline`` / ``view_seperator``（2D tile）或 ``tile_indicators``（1D tile）

这三部分的权重在 HF ckpt 中分别以 ``vision.``、``projector.`` 前缀出现，
``image_newline`` / ``view_seperator`` / ``tile_indicators`` 则是无前缀的顶层参数。
``load_weights`` 接收去前缀后的权重名（由顶层容器的 WEIGHTS_MAPPER 路由），
按前缀分发到对应子模块的 ``load_state_dict``。

NOTE: 视觉塔属于待验证部分。权重加载用档 B（DUMP_WEIGHTS 指纹比对）核对，
forward 计算等价用档 C（推理等价）核对；若 ckpt 的 ``visual.*`` 命名与 HF state_dict
不完全一致，会在 load_state_dict 的 missing/unexpected 里暴露，按提示再修映射。
"""

import logging
from typing import Any, Dict

import torch
import torch.nn as nn

from rtp_llm.models_py.module_base import RtpModule

logger = logging.getLogger(__name__)


class DeepSeekVLV2VisionTransformer(RtpModule):
    """Vision tower for DeepSeek VL V2.

    Wraps the SigLIP vision encoder (timm), MlpProjector, and tile-format
    parameters (image_newline / view_seperator / tile_indicators) in a single
    RtpModule so the top-level container can route all non-language weights
    to ``self.visual``.
    """

    def __init__(self, model_config: Any, load_config: Any):
        super().__init__()
        from rtp_llm.multimodal.multimodal_mixins.deepseek_vl2.deepseek_vl2_vit import (
            MlpProjector,
            MlpProjectorConfig,
            VisionEncoderConfig,
        )

        # --- Resolve ckpt path ---
        ckpt_path = getattr(model_config, "ckpt_path", None)
        if ckpt_path is None and hasattr(model_config, "mm_related_params"):
            ckpt_path = getattr(model_config.mm_related_params, "config", {}).get(
                "ckpt_path"
            )
        if ckpt_path is None:
            ckpt_path = getattr(load_config, "model_path", None)

        # --- Read vision / projector config from the top-level config.json ---
        import json
        import os

        top_config: Dict[str, Any] = {}
        if ckpt_path:
            config_file = os.path.join(ckpt_path, "config.json")
            if os.path.exists(config_file):
                with open(config_file) as f:
                    top_config = json.loads(f.read())

        vision_config_dict = top_config.get("vision_config", {})
        projector_config_dict = top_config.get("projector_config", {})

        # Also check mm_related_params (old loader sets these)
        if not vision_config_dict and hasattr(model_config, "mm_related_params"):
            vision_config_dict = model_config.mm_related_params.config.get(
                "vision_config", {}
            )
        if not projector_config_dict and hasattr(model_config, "mm_related_params"):
            projector_config_dict = model_config.mm_related_params.config.get(
                "projector_config", {}
            )

        vision_config = VisionEncoderConfig(**vision_config_dict)
        projector_config = MlpProjectorConfig(**projector_config_dict)

        params_dtype = getattr(load_config, "compute_dtype", torch.float16)

        # --- Build SigLIP vision encoder (timm) ---
        import timm

        with torch.device("cpu"):
            torch.set_default_dtype(params_dtype)
            try:
                self.vision = timm.create_model(
                    "vit_so400m_patch14_siglip_384.webli",
                    pretrained=False,
                    num_classes=0,
                    dynamic_img_size=True,
                    dynamic_img_pad=True,
                )
                self.vision = self.vision.to(params_dtype)

                # --- Build MlpProjector ---
                self.projector = MlpProjector(projector_config)
                self.projector = self.projector.to(params_dtype)
            finally:
                torch.set_default_dtype(torch.float32)

        # --- Build tile-format parameters ---
        # These match the DeepSeekVLV2ImageEmbedding init in the old mixin.
        tile_tag = top_config.get("tile_tag", "2D")
        if hasattr(model_config, "mm_related_params"):
            tile_tag = model_config.mm_related_params.config.get("tile_tag", tile_tag)

        n_embed = projector_config.n_embed
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))

        self.tile_tag = tile_tag
        if tile_tag == "2D":
            self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
            self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)
        elif tile_tag == "1D":
            candidate_resolutions = top_config.get("candidate_resolutions", [])
            if hasattr(model_config, "mm_related_params"):
                candidate_resolutions = model_config.mm_related_params.config.get(
                    "candidate_resolutions", candidate_resolutions
                )
            if not candidate_resolutions:
                raise ValueError(
                    "len(candidate_resolutions) should be larger than 0 for 1D tile_tag"
                )
            tile_variants_num = len(candidate_resolutions)
            self.tile_indicators = nn.Parameter(
                torch.randn(size=(tile_variants_num + 1, n_embed)) * embed_std
            )
        else:
            raise ValueError(f"tile tag should be either 1D or 2D, but got {tile_tag}")

    def load_weights(self, weights: Any):
        """Dispatch weights to vision, projector, and standalone parameters.

        Receives weight names with the top-level ``visual.`` prefix already
        stripped by the container's WEIGHTS_MAPPER + _groupby_prefix. The
        remaining names are:
          - ``vision.*``      → timm SigLIP model
          - ``projector.*``   → MlpProjector
          - ``image_newline`` → standalone nn.Parameter
          - ``view_seperator`` → standalone nn.Parameter
          - ``tile_indicators`` → standalone nn.Parameter
        """
        if isinstance(weights, dict):
            state: Dict[str, torch.Tensor] = dict(weights.items())
        else:
            state = {name: tensor for name, tensor in weights}

        target_dtype = next(self.vision.parameters()).dtype

        # --- Split by prefix ---
        vision_state: Dict[str, torch.Tensor] = {}
        projector_state: Dict[str, torch.Tensor] = {}
        own_params: Dict[str, torch.Tensor] = {}

        for name, tensor in list(state.items()):
            # Convert dtype for floating-point tensors
            if tensor.is_floating_point():
                tensor = tensor.to(target_dtype)

            if name.startswith("vision."):
                vision_state[name[len("vision.") :]] = tensor
            elif name.startswith("projector."):
                projector_state[name[len("projector.") :]] = tensor
            else:
                own_params[name] = tensor

        # --- Load vision encoder ---
        if vision_state:
            missing, unexpected = self.vision.load_state_dict(
                vision_state, strict=False
            )
            if missing:
                logger.warning(
                    "[DeepSeekVLV2Vision] %d missing vision weight(s): %s%s",
                    len(missing),
                    missing[:10],
                    f" (+{len(missing) - 10} more)" if len(missing) > 10 else "",
                )
            if unexpected:
                logger.warning(
                    "[DeepSeekVLV2Vision] %d unexpected vision weight(s): %s%s",
                    len(unexpected),
                    unexpected[:10],
                    f" (+{len(unexpected) - 10} more)" if len(unexpected) > 10 else "",
                )

        # --- Load projector ---
        if projector_state:
            missing, unexpected = self.projector.load_state_dict(
                projector_state, strict=False
            )
            if missing:
                logger.warning(
                    "[DeepSeekVLV2Vision] %d missing projector weight(s): %s%s",
                    len(missing),
                    missing[:10],
                    f" (+{len(missing) - 10} more)" if len(missing) > 10 else "",
                )
            if unexpected:
                logger.warning(
                    "[DeepSeekVLV2Vision] %d unexpected projector weight(s): %s%s",
                    len(unexpected),
                    unexpected[:10],
                    f" (+{len(unexpected) - 10} more)" if len(unexpected) > 10 else "",
                )

        # --- Load standalone parameters ---
        for name, tensor in own_params.items():
            if hasattr(self, name):
                param = getattr(self, name)
                if isinstance(param, nn.Parameter):
                    param.data.copy_(tensor)
                else:
                    logger.warning(
                        "[DeepSeekVLV2Vision] %s is not an nn.Parameter, skipping",
                        name,
                    )
            else:
                logger.warning(
                    "[DeepSeekVLV2Vision] standalone weight %s has no matching attribute",
                    name,
                )


__all__ = ["DeepSeekVLV2VisionTransformer"]
