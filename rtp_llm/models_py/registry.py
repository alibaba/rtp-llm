import importlib
import inspect
import logging
from typing import Dict, Tuple, Type

import torch.nn as nn

logger = logging.getLogger(__name__)

# 仅声明模型名称到其对应的导入路径和类名
_LAZY_MODEL_REGISTRY: Dict[str, Tuple[str, str]] = {
    # DeepSeek / GLM
    "deepseek_v32": (
        "rtp_llm.models_py.new_models.deepseek_v3",
        "DeepSeekV32ForCausalLM",
    ),
    "deepseek2": (
        "rtp_llm.models_py.new_models.deepseek_v3",
        "DeepSeekV32ForCausalLM",
    ),
    "deepseek3": (
        "rtp_llm.models_py.new_models.deepseek_v3",
        "DeepSeekV32ForCausalLM",
    ),
    "glm_5": ("rtp_llm.models_py.new_models.deepseek_v3", "DeepSeekV32ForCausalLM"),
    "chatglm4": ("rtp_llm.models_py.new_models.glm", "ChatGLMForCausalLM"),
    "chatglm2": ("rtp_llm.models_py.new_models.glm", "ChatGLMForCausalLM"),
    "chat_glm_2": ("rtp_llm.models_py.new_models.glm", "ChatGLMForCausalLM"),
    "chatglm3": ("rtp_llm.models_py.new_models.glm", "ChatGLMForCausalLM"),
    "chat_glm_3": ("rtp_llm.models_py.new_models.glm", "ChatGLMForCausalLM"),
    "glm4_moe": ("rtp_llm.models_py.new_models.glm", "Glm4MoeForCausalLM"),
    # Qwen 语言模型
    "qwen_2": ("rtp_llm.models_py.new_models.qwen2_vl.language", "Qwen2ForCausalLM"),
    "qwen_2_embedding": (
        "rtp_llm.models_py.new_models.qwen2_vl.language",
        "Qwen2ForCausalLM",
    ),
    "qwen_tool": (
        "rtp_llm.models_py.new_models.qwen2_vl.language",
        "Qwen2ForCausalLM",
    ),
    "qwen_3": ("rtp_llm.models_py.new_models.qwen3", "Qwen3ForCausalLM"),
    "qwen_3_tool": ("rtp_llm.models_py.new_models.qwen3", "Qwen3ForCausalLM"),
    "qwen_3_moe": ("rtp_llm.models_py.new_models.qwen3_moe", "Qwen3MoeForCausalLM"),
    "qwen3_coder_moe": (
        "rtp_llm.models_py.new_models.qwen3_moe",
        "Qwen3MoeForCausalLM",
    ),
    "qwen35_moe": (
        "rtp_llm.models_py.new_models.qwen3_next",
        "Qwen35MoeForCausalLM",
    ),
    "qwen35_dense": (
        "rtp_llm.models_py.new_models.qwen3_next",
        "Qwen35DenseForCausalLM",
    ),
    "qwen3_next": (
        "rtp_llm.models_py.new_models.qwen3_next",
        "Qwen3NextForCausalLM",
    ),
    "qwen_3_moe_eagle3": (
        "rtp_llm.models_py.new_models.qwen3_next",
        "Qwen3MoeEagle3ForCausalLM",
    ),
    # Qwen 多模态
    "qwen2_vl": (
        "rtp_llm.models_py.new_models.qwen2_vl",
        "Qwen2VLForConditionalGeneration",
    ),
    "qwen3_vl": (
        "rtp_llm.models_py.new_models.qwen3_vl",
        "Qwen3VLForConditionalGeneration",
    ),
    "qwen3_vl_moe": (
        "rtp_llm.models_py.new_models.qwen3_vl_moe",
        "Qwen3VLMoeForConditionalGeneration",
    ),
    "kimi_linear": (
        "rtp_llm.models_py.new_models.kimi_linear",
        "KimiLinearForCausalLM",
    ),
    # MiniMax
    "minimax_m3_vl": (
        "rtp_llm.models_py.new_models.minimax_m3.model",
        "MiniMaxM3VLForConditionalGeneration",
    ),
    # MTP
    "qwen_2-mtp": ("rtp_llm.models_py.new_models.qwen2_mtp", "Qwen2MTPForCausalLM"),
    "qwen35_moe_mtp": (
        "rtp_llm.models_py.new_models.qwen3_next",
        "Qwen35MoeMTPForCausalLM",
    ),
    "qwen3_next_mtp": (
        "rtp_llm.models_py.new_models.qwen3_next",
        "Qwen3NextMTPForCausalLM",
    ),
    "deepseek-v3-mtp": (
        "rtp_llm.models_py.new_models.deepseek_v3_mtp",
        "DeepSeekV32MTPForCausalLM",
    ),
    "llama": (
        "rtp_llm.models_py.new_models.llama",
        "LlamaForCausalLM",
    ),
    "bert": (
        "rtp_llm.models_py.new_models.bert",
        "BertForEmbedding",
    ),
    "roberta": (
        "rtp_llm.models_py.new_models.bert",
        "RobertaForEmbedding",
    ),
}


class LazyRegistryDict(dict):
    """A dictionary that lazily imports and registers models on demand.

    This ensures we remain 100% backward compatible with external code that
    directly accesses or iterates over MODEL_REGISTRY, while preventing
    eager imports of all model dependencies during package initialization.
    """

    def __getitem__(self, key):
        if not dict.__contains__(self, key) and key in _LAZY_MODEL_REGISTRY:
            # Trigger lazy load
            cls = get_model_class(key)
            dict.__setitem__(self, key, cls)
        return super().__getitem__(key)

    def __contains__(self, key):
        return key in _LAZY_MODEL_REGISTRY or super().__contains__(key)

    def keys(self):
        return _LAZY_MODEL_REGISTRY.keys()

    def __iter__(self):
        return iter(_LAZY_MODEL_REGISTRY.keys())

    def __len__(self):
        return len(_LAZY_MODEL_REGISTRY)

    def items(self):
        for k in _LAZY_MODEL_REGISTRY.keys():
            yield k, self[k]


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = LazyRegistryDict()


def _safe_class_file(cls: Type[nn.Module]) -> str:
    try:
        return inspect.getfile(cls)
    except TypeError:
        return "<unknown>"


def register_model(model_type: str):
    def decorator(cls):
        caller = inspect.stack()[1]
        cls_file = _safe_class_file(cls)
        if model_type in MODEL_REGISTRY and model_type not in _LAZY_MODEL_REGISTRY:
            existing_cls = MODEL_REGISTRY[model_type]
            logger.warning(
                "NewModelRegistry duplicate registration: model_type=%s "
                "new_cls=%s new_module=%s new_file=%s caller=%s:%s "
                "existing_cls=%s existing_module=%s existing_file=%s",
                model_type,
                cls.__qualname__,
                cls.__module__,
                cls_file,
                caller.filename,
                caller.lineno,
                existing_cls.__qualname__,
                existing_cls.__module__,
                _safe_class_file(existing_cls),
            )
            raise ValueError(
                f"Model type '{model_type}' is already registered by "
                f"{MODEL_REGISTRY[model_type].__name__}. Cannot register {cls.__name__}."
            )
        logger.info(
            "NewModelRegistry register_model: model_type=%s cls=%s module=%s "
            "file=%s caller=%s:%s",
            model_type,
            cls.__qualname__,
            cls.__module__,
            cls_file,
            caller.filename,
            caller.lineno,
        )
        # Directly set the item in base dict to cache it
        dict.__setitem__(MODEL_REGISTRY, model_type, cls)
        return cls

    return decorator


def get_model_class(model_type: str) -> Type[nn.Module]:
    # Check physical cache first (registered classes)
    if not dict.__contains__(MODEL_REGISTRY, model_type):
        if model_type in _LAZY_MODEL_REGISTRY:
            module_path, class_name = _LAZY_MODEL_REGISTRY[model_type]
            try:
                mod = importlib.import_module(module_path)
                cls = getattr(mod, class_name)
                dict.__setitem__(MODEL_REGISTRY, model_type, cls)
            except Exception as e:
                logger.error(
                    "NewModelRegistry failed to lazy load model_type=%s from %s.%s: %s",
                    model_type,
                    module_path,
                    class_name,
                    e,
                )
                raise ImportError(
                    f"Failed to dynamically load model '{model_type}' from {module_path}.{class_name}: {e}"
                ) from e
        else:
            raise KeyError(
                f"Model type '{model_type}' not found in registry. "
                f"Available types: {list(_LAZY_MODEL_REGISTRY.keys())}"
            )
    return dict.__getitem__(MODEL_REGISTRY, model_type)


def list_models():
    return list(_LAZY_MODEL_REGISTRY.keys())
