import importlib
import inspect
import logging
from typing import Dict, Tuple, Type

import torch.nn as nn

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}
_LAZY_MODEL_REGISTRY: Dict[str, Tuple[str, str]] = {}


def _validate_model_type(model_type: str) -> str:
    if not isinstance(model_type, str) or not model_type.strip():
        raise ValueError("model_type must be a non-empty string")
    return model_type.strip()


def _class_location(cls: Type[nn.Module]) -> str:
    try:
        return inspect.getfile(cls)
    except (TypeError, OSError):
        return "<unknown>"


def register_lazy_model(model_type: str, module_path: str, class_name: str) -> None:
    model_type = _validate_model_type(model_type)
    if not module_path or not class_name:
        raise ValueError("module_path and class_name must be non-empty")
    if model_type in MODEL_REGISTRY:
        raise ValueError(
            f"Model type {model_type!r} is already registered by "
            f"{MODEL_REGISTRY[model_type].__name__}"
        )
    existing = _LAZY_MODEL_REGISTRY.get(model_type)
    candidate = (module_path, class_name)
    if existing is not None and existing != candidate:
        raise ValueError(
            f"Model type {model_type!r} already maps to {existing[0]}.{existing[1]}"
        )
    _LAZY_MODEL_REGISTRY[model_type] = candidate


def register_model(model_type: str):
    model_type = _validate_model_type(model_type)

    def decorator(cls: Type[nn.Module]):
        if not isinstance(cls, type) or not issubclass(cls, nn.Module):
            raise TypeError(f"Registered model must inherit nn.Module, got {cls!r}")
        existing = MODEL_REGISTRY.get(model_type)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Model type {model_type!r} is already registered by "
                f"{existing.__module__}.{existing.__qualname__}"
            )
        lazy = _LAZY_MODEL_REGISTRY.get(model_type)
        if lazy is not None and lazy != (cls.__module__, cls.__name__):
            raise ValueError(
                f"Model type {model_type!r} already maps to {lazy[0]}.{lazy[1]}"
            )
        MODEL_REGISTRY[model_type] = cls
        logger.info(
            "Registered newloader model %s -> %s.%s (%s)",
            model_type,
            cls.__module__,
            cls.__qualname__,
            _class_location(cls),
        )
        return cls

    return decorator


def get_model_class(model_type: str) -> Type[nn.Module]:
    model_type = _validate_model_type(model_type)
    cached = MODEL_REGISTRY.get(model_type)
    if cached is not None:
        return cached

    target = _LAZY_MODEL_REGISTRY.get(model_type)
    if target is None:
        raise KeyError(
            f"Model type {model_type!r} is not registered. Available: {list_models()}"
        )
    module_path, class_name = target
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            f"Failed to load model {model_type!r} from {module_path}.{class_name}: {exc}"
        ) from exc
    if not isinstance(cls, type) or not issubclass(cls, nn.Module):
        raise TypeError(
            f"Lazy model {module_path}.{class_name} must be an nn.Module subclass"
        )
    MODEL_REGISTRY[model_type] = cls
    return cls


def list_models():
    return sorted(set(MODEL_REGISTRY) | set(_LAZY_MODEL_REGISTRY))


def is_model_registered(model_type: str) -> bool:
    """Return whether ``model_type`` has an explicit NewLoader implementation."""
    model_type = _validate_model_type(model_type)
    return model_type in MODEL_REGISTRY or model_type in _LAZY_MODEL_REGISTRY


# Additional owners are registered only after their main-compatible port lands.
register_lazy_model(
    "bert",
    "rtp_llm.models_py.new_models.bert.language",
    "BertEncoderOwner",
)
register_lazy_model(
    "deepseek_v41",
    "rtp_llm.models_py.new_models.deepseek_v41.language",
    "DeepSeekV41ForCausalLM",
)
register_lazy_model(
    "deepseek_v41_dspark",
    "rtp_llm.models_py.new_models.deepseek_v41_dspark.language",
    "DeepSeekV41DSparkForCausalLM",
)

for _name in ("qwen_3_moe", "qwen3_coder_moe"):
    register_lazy_model(
        _name, "rtp_llm.models_py.new_models.qwen3_moe.language", "Qwen3MoeForCausalLM"
    )
for _name in ("qwen_3", "qwen_3_tool"):
    register_lazy_model(_name, "rtp_llm.models_py.new_models.qwen3", "Qwen3ForCausalLM")
register_lazy_model(
    "glm4_moe",
    "rtp_llm.models_py.new_models.glm4_moe.language",
    "Glm4MoeForCausalLM",
)
register_lazy_model(
    "llama",
    "rtp_llm.models_py.new_models.qwen2.llama",
    "LlamaForCausalLM",
)
register_lazy_model(
    "qwen_2",
    "rtp_llm.models_py.new_models.qwen2",
    "Qwen2ForCausalLM",
)
register_lazy_model(
    "qwen_7b", "rtp_llm.models_py.new_models.qwen1.language", "Qwen1ForCausalLM"
)
# qwen_tool changes tokenization/rendering and request output selection, not
# the language-model parameters. Keep its complete causal LM projection.
for _name in ("qwen_tool", "qwen_agent"):
    register_lazy_model(_name, "rtp_llm.models_py.new_models.qwen2", "Qwen2ForCausalLM")
register_lazy_model(
    "qwen_2_embedding",
    "rtp_llm.models_py.new_models.qwen2.language",
    "Qwen2EmbeddingBackbone",
)
register_lazy_model(
    "qwen_2-mtp", "rtp_llm.models_py.new_models.qwen2_mtp", "Qwen2MTPForCausalLM"
)

# Distinct service types share main's vision tower, not a text-only runtime.
for _vl_type, _vl_owner in (
    ("qwen2_vl", "Qwen2VLForCausalLM"),
    ("qwen2_5_vl", "Qwen2VLForCausalLM"),
    ("qwen3_vl", "Qwen3VLForCausalLM"),
    ("qwen3_vl_moe", "Qwen3VLMoeForCausalLM"),
):
    register_lazy_model(
        _vl_type, "rtp_llm.models_py.new_models.qwen3_vl.language", _vl_owner
    )

# The owner selects direct Q, compressed Q or the sparse indexer from the
# parsed attention config; it does not replace a sparse model with dense MLA.
for _model_type in (
    "deepseek2",
    "glm4_moe_lite",
    "deepseek3",
    "deepseek_v32",
    "glm_5",
    "deepseek_v31",
    "kimi_k2",
):
    register_lazy_model(
        _model_type,
        "rtp_llm.models_py.new_models.deepseek_v3.language",
        "DeepSeekV32ForCausalLM",
    )

register_lazy_model(
    "deepseek-v3-mtp",
    "rtp_llm.models_py.new_models.deepseek_v3_mtp.language",
    "DeepSeekV32MTPForCausalLM",
)

register_lazy_model(
    "kimi_linear",
    "rtp_llm.models_py.new_models.kimi_linear.language",
    "KimiLinearForCausalLM",
)

register_lazy_model(
    "kimi_k25",
    "rtp_llm.models_py.new_models.kimi_k25.language",
    "KimiK25ForCausalLM",
)
register_lazy_model(
    "kimi_k25_vision",
    "rtp_llm.models_py.new_models.kimi_k25.vision",
    "KimiK25ForVisionEmbedding",
)

# Use the service model_type, not the Hugging Face architecture string. Draft
# models need different checkpoint prefixes and cannot alias the target class.
for _model_type, _owner in (
    ("qwen3_next", "Qwen3NextForCausalLM"),
    ("qwen35_dense", "Qwen3NextForCausalLM"),
    ("qwen35_moe", "Qwen3NextForCausalLM"),
    ("qwen3_next_mtp", "Qwen3NextMTPForCausalLM"),
    ("qwen35_dense_mtp", "Qwen35DenseMTPForCausalLM"),
    ("qwen35_moe_mtp", "Qwen35MoeMTPForCausalLM"),
):
    register_lazy_model(
        _model_type, "rtp_llm.models_py.new_models.qwen3_next.language", _owner
    )

register_lazy_model(
    "deepseek_v4",
    "rtp_llm.models_py.new_models.deepseek_v4.language",
    "DeepSeekV4ForCausalLM",
)
register_lazy_model(
    "deepseek_v4_dspark",
    "rtp_llm.models_py.new_models.deepseek_v4_dspark.language",
    "DeepSeekV4DSparkForCausalLM",
)
register_lazy_model(
    "deepseek_v4_mtp",
    "rtp_llm.models_py.new_models.deepseek_v4.mtp",
    "DeepSeekV4MtpForCausalLM",
)

register_lazy_model(
    "roberta",
    "rtp_llm.models_py.new_models.roberta.language",
    "RobertaEncoderOwner",
)

register_lazy_model(
    "cosyvoice_qwen",
    "rtp_llm.models_py.new_models.cosyvoice_qwen.language",
    "CosyVoiceQwenForCausalLM",
)

register_lazy_model(
    "jina_bert_code",
    "rtp_llm.models_py.new_models.jina_bert.runtime",
    "JinaBertForEmbedding",
)

register_lazy_model(
    "megatron_bert",
    "rtp_llm.models_py.new_models.megatron_bert.language",
    "MegatronBertForEmbedding",
)

register_lazy_model(
    "qwen_3_dspark",
    "rtp_llm.models_py.new_models.qwen3_dspark.language",
    "Qwen3DSparkForCausalLM",
)

register_lazy_model(
    "qwen_3_moe_eagle3",
    "rtp_llm.models_py.new_models.qwen3_moe_eagle3.service",
    "Qwen3MoeEagle3ForCausalLM",
)

register_lazy_model(
    "qwen4_exp",
    "rtp_llm.models_py.new_models.qwen4_exp.language",
    "Qwen4ForCausalLM",
)

register_lazy_model(
    "qwen4_exp_mtp",
    "rtp_llm.models_py.new_models.qwen4_exp_mtp.draft_service",
    "Qwen4ExpMTPForCausalLM",
)
