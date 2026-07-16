import platform
import threading
from typing import Any, Dict, Iterable, List, Optional, Type, Union

from rtp_llm.utils.import_util import (
    LazyModuleRegistry,
    import_optional_internal_source_entrypoint,
)

_tokenizer_factory: Dict[str, Type[Any]] = {}
_tokenizer_registry_lock = threading.RLock()
_lazy_tokenizer_registry = LazyModuleRegistry("tokenizer")
_tokenizer_type_to_module = _lazy_tokenizer_registry.name_to_module
_lazy_tokenizer_modules = _lazy_tokenizer_registry.module_paths
_loaded_tokenizer_modules = _lazy_tokenizer_registry.loaded_modules
_internal_lazy_tokenizers_registered = False
_internal_legacy_tokenizers_loaded = False


def register_tokenizer(name: Union[str, List[str]], tokenizer: Any):
    global _tokenizer_factory
    if isinstance(name, list):
        for n in name:
            register_tokenizer(n, tokenizer)
    else:
        with _tokenizer_registry_lock:
            if name in _tokenizer_factory and _tokenizer_factory[name] != tokenizer:
                raise Exception(
                    f"try register model {name} with type {_tokenizer_factory[name]} and {tokenizer}, confict!"
                )
            _tokenizer_factory[name] = tokenizer


def register_lazy_tokenizer(
    name: Union[str, Iterable[str]],
    module_path: str,
) -> None:
    _lazy_tokenizer_registry.register(name, module_path)


def _import_tokenizer_module(module_path: str) -> None:
    _lazy_tokenizer_registry.import_module(module_path)


def _load_internal_lazy_tokenizers() -> None:
    global _internal_lazy_tokenizers_registered
    with _tokenizer_registry_lock:
        if _internal_lazy_tokenizers_registered:
            return
        import_optional_internal_source_entrypoint("tokenizers.lazy_register")
        _internal_lazy_tokenizers_registered = True


def _load_internal_legacy_tokenizers() -> None:
    global _internal_legacy_tokenizers_loaded
    with _tokenizer_registry_lock:
        if _internal_legacy_tokenizers_loaded:
            return
        import_optional_internal_source_entrypoint("tokenizers.internal_init")
        _internal_legacy_tokenizers_loaded = True


def _should_try_internal_legacy_tokenizer(model_type: str) -> bool:
    from rtp_llm.model_factory_register import get_lazy_model_module_path

    model_module_path = get_lazy_model_module_path(model_type)
    return model_module_path is None or model_module_path.startswith("internal_source.")


def ensure_tokenizer_registered(model_type: Optional[str]) -> bool:
    if not model_type:
        return False
    _load_internal_lazy_tokenizers()
    with _tokenizer_registry_lock:
        if model_type in _tokenizer_factory:
            return True
    module_path = _lazy_tokenizer_registry.get_module_path(model_type)
    if not module_path:
        if not _should_try_internal_legacy_tokenizer(model_type):
            return False
        _load_internal_legacy_tokenizers()
        with _tokenizer_registry_lock:
            if model_type in _tokenizer_factory:
                return True
        module_path = _lazy_tokenizer_registry.get_module_path(model_type)
        if not module_path:
            return False
    _import_tokenizer_module(module_path)
    with _tokenizer_registry_lock:
        return model_type in _tokenizer_factory


def ensure_all_tokenizers_registered() -> None:
    _load_internal_lazy_tokenizers()
    _lazy_tokenizer_registry.import_all_modules()
    _load_internal_legacy_tokenizers()


def _register_builtin_lazy_tokenizers() -> None:
    register_lazy_tokenizer(
        ["bert", "roberta", "vision_bert"],
        "rtp_llm.frontend.tokenizer_factory.tokenizers.bert_tokenizer",
    )
    register_lazy_tokenizer(
        [
            "chatglm2",
            "chat_glm_2",
            "chatglm3",
            "chat_glm_3",
            "chatglm4",
            "chatglm4v",
            "glm_5",
        ],
        "rtp_llm.frontend.tokenizer_factory.tokenizers.chatglm_tokenizer",
    )
    register_lazy_tokenizer(
        "deepseek_vl_v2",
        "rtp_llm.frontend.tokenizer_factory.tokenizers.deepseek_vl2_tokenizer",
    )
    register_lazy_tokenizer(
        [
            "llama",
            "internlm",
            "internlm2",
            "xverse",
            "aquila",
            "mistral",
            "baichuan",
            "baichuan2",
            "gemma",
            "cohere",
        ],
        "rtp_llm.frontend.tokenizer_factory.tokenizers.llama_tokenizer",
    )
    register_lazy_tokenizer(
        "llava", "rtp_llm.frontend.tokenizer_factory.tokenizers.llava_tokenizer"
    )
    register_lazy_tokenizer(
        ["qwen", "qwen_7b", "qwen_13b", "qwen_1b8"],
        "rtp_llm.frontend.tokenizer_factory.tokenizers.qwen_tokenizer",
    )
    register_lazy_tokenizer(
        [
            "qwen_2",
            "qwen_agent",
            "qwen_2_embedding",
            "qwen_tool",
            "qwen_2-mtp",
            "qwen_3",
            "qwen_3_tool",
        ],
        "rtp_llm.frontend.tokenizer_factory.tokenizers.qwen_tokenizer",
    )

    # Preserve the old ARM behavior: these modules were not imported on aarch64.
    if platform.processor() != "aarch64":
        register_lazy_tokenizer(
            "internvl",
            "rtp_llm.frontend.tokenizer_factory.tokenizers.internvl_tokenizer",
        )
        register_lazy_tokenizer(
            "minicpmv_embedding",
            "rtp_llm.frontend.tokenizer_factory.tokenizers.minicpmv_embedding_tokenizer",
        )


_register_builtin_lazy_tokenizers()
_load_internal_lazy_tokenizers()
