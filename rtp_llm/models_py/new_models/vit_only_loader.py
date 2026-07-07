"""新 loader 风格的「只加载 vit」入口，供独立 vit server 使用。

独立 vit server（``VIT_SEPARATION=ROLE/REMOTE``）进程里没有 py_model，无法像 LOCAL 模式
那样复用 ``py_model.visual``（见 [[newloader-vl-vision-pipeline]]）。本模块用新 loader 的加载
方式（迭代 safetensors → ``load_weights``，**无 CkptDatabase / 无 ModelDeployWeightInfo**）
单独把 vit 建好并加载，再作为 ``injected_vit_module`` 注入 mm 流水线，使 vision 在独立 server
里也走新 loader 的加载哲学。

只建 vit、不建语言模型——这正是独立 vit server 存在的意义（不在 vit 进程里持有 LLM）。
非新 loader、或该 model_type 未注册时返回 None → mm 流水线回退到旧 CkptDatabase 加载方式。
"""

import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# model_type -> (vit_wrapper_cls, ckpt_visual_prefix)
#   vit_wrapper_cls: __init__(model_config, load_config)，提供 .load_weights(iter)
#       与 .vit（mm_part 期望的 self.visual：带 (pixel_values, grid_thw) 接口和
#       .spatial_merge_size / .patch_size 的 HF 视觉塔）。
_VIT_BUILDERS: Dict[str, Tuple[Callable[..., Any], str]] = {}


def register_new_loader_vit(model_types, wrapper_cls, ckpt_visual_prefix: str) -> None:
    if isinstance(model_types, str):
        model_types = [model_types]
    for mt in model_types:
        _VIT_BUILDERS[mt] = (wrapper_cls, ckpt_visual_prefix)


def _new_loader_active(model_config: Any) -> bool:
    # 与 base_model._use_new_loader 同口径：环境变量优先，其次 model_config 标志。
    if os.environ.get("USE_NEW_LOADER", "0") == "1":
        return True
    return bool(getattr(model_config, "use_new_loader", False))


def build_new_loader_vit(
    model_config: Any, device: str = "cuda"
) -> Optional[torch.nn.Module]:
    """新 loader 方式建并加载 vit，返回 mm_part 期望的 vit 模块。

    非新 loader / 该 model_type 未注册 / 找不到 ckpt → 返回 None（回退旧加载方式）。
    """
    if not _new_loader_active(model_config):
        return None
    model_type = getattr(model_config, "model_type", None)
    entry = _VIT_BUILDERS.get(model_type)
    if entry is None:
        logger.info(
            "[vit_only_loader] model_type=%s 未注册新 loader vit builder，回退旧加载方式",
            model_type,
        )
        return None
    wrapper_cls, prefix = entry

    from rtp_llm.models_py.model_loader import (
        LoadConfig,
        _discover_ckpt_files,
        _get_all_weights,
    )

    ckpt_path = model_config.ckpt_path
    compute_dtype = getattr(model_config, "compute_dtype", torch.float16)
    load_config = LoadConfig(
        compute_dtype=compute_dtype, device=device, model_path=ckpt_path
    )

    ckpt_files = _discover_ckpt_files(ckpt_path)
    if not ckpt_files:
        logger.warning(
            "[vit_only_loader] 在 %s 未发现 ckpt 文件，回退旧加载方式", ckpt_path
        )
        return None

    wrapper = wrapper_cls(model_config=model_config, load_config=load_config)

    def _visual_iter():
        # 从 ckpt 的 model.visual.<hf_key> 剥成裸 HF key（wrapper.load_weights →
        # vit.load_state_dict 直接匹配）。非 visual 的权重（语言模型）全部跳过。
        n = len(prefix)
        for name, tensor in _get_all_weights(ckpt_files, device="cpu"):
            if name.startswith(prefix):
                yield name[n:], tensor

    wrapper.load_weights(_visual_iter())
    wrapper.to(device)
    logger.info(
        "[vit_only_loader] new-loader vit loaded "
        "(model_type=%s, prefix=%s, device=%s)",
        model_type,
        prefix,
        device,
    )
    return getattr(wrapper, "vit", wrapper)


def _register_builtin_builders() -> None:
    # qwen3_vl / qwen3_vl_moe 共用同一个 HF 视觉塔 wrapper、同样的 ckpt 前缀。
    from rtp_llm.models_py.new_models.qwen3_vl.vision import Qwen3VLVisionTransformer

    register_new_loader_vit(
        ["qwen3_vl", "qwen3_vl_moe"], Qwen3VLVisionTransformer, "model.visual."
    )

    # DeepSeek VL V2 视觉塔: SigLIP (timm) + MlpProjector + tile-format 参数。
    # ckpt 前缀为 ""（无前缀）—— vision.* / projector.* / image_newline 等
    # 都是顶层键。wrapper.load_weights 内部按前缀分发到对应子模块。
    from rtp_llm.models_py.new_models.deepseek_vl2.vision import (
        DeepSeekVLV2VisionTransformer,
    )

    register_new_loader_vit(["deepseek_vl_v2"], DeepSeekVLV2VisionTransformer, "")


try:
    _register_builtin_builders()
except Exception as e:  # pragma: no cover - 注册失败不应阻塞非 VL 启动
    logger.warning("[vit_only_loader] 注册内置 vit builder 失败: %s", e)
