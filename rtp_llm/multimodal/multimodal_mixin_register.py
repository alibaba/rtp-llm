from typing import Any, Dict, List, Type, Union

from rtp_llm.utils.import_util import (
    LazyModuleRegistry,
    import_optional_internal_source_entrypoint,
)

_multimodal_mixin_factory: Dict[str, Type[Any]] = {}
_lazy_multimodal_mixin_registry = LazyModuleRegistry("multimodal mixin")
_internal_multimodal_mixins_loaded = False


def register_multimodal_mixin(name: Union[str, List[str]], multimodal_mixin: Any):
    global _multimodal_mixin_factory
    if isinstance(name, List):
        for n in name:
            register_multimodal_mixin(n, multimodal_mixin)
    else:
        if (
            name in _multimodal_mixin_factory
            and _multimodal_mixin_factory[name] != multimodal_mixin
        ):
            raise Exception(
                f"try register model {name} with type {_multimodal_mixin_factory[name]} and {multimodal_mixin}, confict!"
            )
        _multimodal_mixin_factory[name] = multimodal_mixin


def register_lazy_multimodal_mixin(
    name: Union[str, List[str]], module_path: str
) -> None:
    _lazy_multimodal_mixin_registry.register(name, module_path)


def get_multimodal_mixin_cls(name: str) -> Type[Any]:
    global _internal_multimodal_mixins_loaded

    module_path = _lazy_multimodal_mixin_registry.get_module_path(name)
    if module_path is not None:
        _lazy_multimodal_mixin_registry.import_module(module_path)
    elif not _internal_multimodal_mixins_loaded:
        import_optional_internal_source_entrypoint("multimodal_mixins.internal_init")
        _internal_multimodal_mixins_loaded = True

    if name not in _multimodal_mixin_factory:
        raise ValueError(f"Multimodal mixin {name} not found")
    return _multimodal_mixin_factory[name]


register_lazy_multimodal_mixin(
    "chatglm4v",
    "rtp_llm.multimodal.multimodal_mixins.chatglm4v.chatglm4v_mixin",
)
register_lazy_multimodal_mixin(
    "deepseek_vl_v2",
    "rtp_llm.multimodal.multimodal_mixins.deepseek_vl2.deepseek_vl2_mixin",
)
register_lazy_multimodal_mixin(
    "kimi_k25", "rtp_llm.multimodal.multimodal_mixins.kimi_k25.kimi_k25_mixin"
)
register_lazy_multimodal_mixin(
    "llava", "rtp_llm.multimodal.multimodal_mixins.llava.llava_mixin"
)
register_lazy_multimodal_mixin(
    "minimax_m3_vl",
    "rtp_llm.multimodal.multimodal_mixins.minimax_m3_vl.minimax_m3_vl_mixin",
)
register_lazy_multimodal_mixin(
    "qwen2_5_vl",
    "rtp_llm.multimodal.multimodal_mixins.qwen2_5_vl.qwen2_5_vl_mixin",
)
register_lazy_multimodal_mixin(
    "qwen_v2_audio",
    "rtp_llm.multimodal.multimodal_mixins.qwen2_audio.qwen2_audio_mixin",
)
register_lazy_multimodal_mixin(
    "qwen2_vl",
    "rtp_llm.multimodal.multimodal_mixins.qwen2_vl.qwen2_vl_mixin",
)
register_lazy_multimodal_mixin(
    ["qwen35_moe", "qwen35_dense", "qwen35_moe_mtp"],
    "rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_mixin",
)
register_lazy_multimodal_mixin(
    ["qwen3_vl", "qwen3_vl_moe"],
    "rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin",
)
register_lazy_multimodal_mixin(
    "qwen_vl", "rtp_llm.multimodal.multimodal_mixins.qwen_vl.qwen_vl_mixin"
)
