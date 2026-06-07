import threading
from typing import Any, Dict, Iterable, Optional, Type

from rtp_llm.utils.import_util import (
    LazyModuleRegistry,
    import_optional_internal_source_entrypoint,
)

_renderer_factory: Dict[str, Type[Any]] = {}
_renderer_registry_lock = threading.RLock()
_lazy_renderer_registry = LazyModuleRegistry("renderer")
_renderer_type_to_module = _lazy_renderer_registry.name_to_module
_lazy_renderer_modules = _lazy_renderer_registry.module_paths
_loaded_renderer_modules = _lazy_renderer_registry.loaded_modules
_internal_lazy_renderers_registered = False
_internal_legacy_renderers_loaded = False


def register_renderer(name: str, renderer_type: Any):
    global _renderer_factory
    with _renderer_registry_lock:
        if name in _renderer_factory and _renderer_factory[name] != renderer_type:
            raise Exception(
                f"try register renderer {name} with type {_renderer_factory[name]} and {renderer_type}, confict!"
            )
        _renderer_factory[name] = renderer_type


def register_lazy_renderer(names: Iterable[str], module_path: str) -> None:
    _lazy_renderer_registry.register(names, module_path)


def _import_renderer_module(module_path: str) -> None:
    _lazy_renderer_registry.import_module(module_path)


def _load_internal_lazy_renderers() -> None:
    global _internal_lazy_renderers_registered
    with _renderer_registry_lock:
        if _internal_lazy_renderers_registered:
            return
        import_optional_internal_source_entrypoint("openai_renderers.lazy_register")
        _internal_lazy_renderers_registered = True


def _load_internal_legacy_renderers() -> None:
    global _internal_legacy_renderers_loaded
    with _renderer_registry_lock:
        if _internal_legacy_renderers_loaded:
            return
        import_optional_internal_source_entrypoint("openai_renderers.internal_init")
        _internal_legacy_renderers_loaded = True


def _should_try_internal_legacy_renderer(renderer_type: str) -> bool:
    from rtp_llm.model_factory_register import get_lazy_model_module_path

    model_module_path = get_lazy_model_module_path(renderer_type)
    return model_module_path is None or model_module_path.startswith("internal_source.")


def ensure_renderer_registered(renderer_type: Optional[str]) -> bool:
    if not renderer_type:
        return False
    _load_internal_lazy_renderers()
    with _renderer_registry_lock:
        if renderer_type in _renderer_factory:
            return True
    module_path = _lazy_renderer_registry.get_module_path(renderer_type)
    if not module_path:
        if not _should_try_internal_legacy_renderer(renderer_type):
            return False
        _load_internal_legacy_renderers()
        with _renderer_registry_lock:
            if renderer_type in _renderer_factory:
                return True
        module_path = _lazy_renderer_registry.get_module_path(renderer_type)
        if not module_path:
            return False
    _import_renderer_module(module_path)
    with _renderer_registry_lock:
        return renderer_type in _renderer_factory


def ensure_all_renderers_registered() -> None:
    _load_internal_lazy_renderers()
    _lazy_renderer_registry.import_all_modules()
    _load_internal_legacy_renderers()


def _register_builtin_lazy_renderers() -> None:
    register_lazy_renderer(
        ["chatglm4", "chatglm4v"], "rtp_llm.openai.renderers.chatglm4_renderer"
    )
    register_lazy_renderer(
        ["glm4_moe", "glm_5"], "rtp_llm.openai.renderers.chatglm45_renderer"
    )
    register_lazy_renderer(["glm47_moe"], "rtp_llm.openai.renderers.chatglm47_renderer")
    register_lazy_renderer(
        ["deepseek_vl_v2"], "rtp_llm.openai.renderers.deepseek_vl2_renderer"
    )
    register_lazy_renderer(
        ["deepseek_v31"], "rtp_llm.openai.renderers.deepseekv31_renderer"
    )
    register_lazy_renderer(
        ["deepseek_v32"], "rtp_llm.openai.renderers.deepseekv32_renderer"
    )
    register_lazy_renderer(
        ["deepseek_v4"], "rtp_llm.openai.renderers.deepseekv4_renderer"
    )
    register_lazy_renderer(["internvl"], "rtp_llm.openai.renderers.internvl_renderer")
    register_lazy_renderer(["kimi_k2"], "rtp_llm.openai.renderers.kimik2_renderer")
    register_lazy_renderer(["llava"], "rtp_llm.openai.renderers.llava_renderer")
    register_lazy_renderer(["minicpmv"], "rtp_llm.openai.renderers.minicpmv_renderer")
    register_lazy_renderer(
        ["qwen3_coder_moe", "qwen35_moe", "qwen35_dense"],
        "rtp_llm.openai.renderers.qwen3_code_renderer",
    )
    register_lazy_renderer(
        ["qwen_agent"], "rtp_llm.openai.renderers.qwen_agent_renderer"
    )
    register_lazy_renderer(
        ["qwen_agent_tool"], "rtp_llm.openai.renderers.qwen_agent_tool_renderer"
    )
    register_lazy_renderer(
        ["qwen", "qwen_7b", "qwen_13b", "qwen_1b8", "qwen_2", "qwen_2_moe"],
        "rtp_llm.openai.renderers.qwen_renderer",
    )
    register_lazy_renderer(
        ["qwen_tool", "qwen_3_tool", "qwen_3", "qwen_3_moe", "qwen3_next"],
        "rtp_llm.openai.renderers.qwen_reasoning_tool_renderer",
    )
    register_lazy_renderer(
        ["qwen_v2_audio"], "rtp_llm.openai.renderers.qwen_v2_audio_renderer"
    )
    register_lazy_renderer(
        ["qwen_vl", "qwen_vl_1b8", "qwen2_vl", "qwen2_5_vl", "qwen3_vl_moe"],
        "rtp_llm.openai.renderers.qwen_vl_renderer",
    )


_register_builtin_lazy_renderers()
_load_internal_lazy_renderers()
