import importlib
from typing import Any, Dict

from .base_multimodal_mixin import (
    BaseMultiModalDeployWeightInfo,
    BaseMultiModalMixin,
    BaseVitWeights,
)

_CLASS_TO_MODULE: Dict[str, str] = {
    "ChatGlmV4VisionMixin": ".chatglm4v.chatglm4v_mixin",
    "DeepSeekVLV2Mixin": ".deepseek_vl2.deepseek_vl2_mixin",
    "KimiK25Mixin": ".kimi_k25.kimi_k25_mixin",
    "LlavaMixin": ".llava.llava_mixin",
    "MiniMaxM3VLMixin": ".minimax_m3_vl.minimax_m3_vl_mixin",
    "Qwen2_5_VLMixin": ".qwen2_5_vl.qwen2_5_vl_mixin",
    "Qwen2_AudioMixin": ".qwen2_audio.qwen2_audio_mixin",
    "Qwen2_VLMixin": ".qwen2_vl.qwen2_vl_mixin",
    "Qwen3_5MoeMixin": ".qwen3_5_moe.qwen3_5_moe_mixin",
    "Qwen3_VLMixin": ".qwen3_vl_mixin",
    "QwenVLMixin": ".qwen_vl.qwen_vl_mixin",
}

__all__ = [
    "BaseMultiModalDeployWeightInfo",
    "BaseMultiModalMixin",
    "BaseVitWeights",
    *_CLASS_TO_MODULE,
]


def __getattr__(name: str) -> Any:
    module_path = _CLASS_TO_MODULE.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path, __name__), name)
    globals()[name] = value
    return value
