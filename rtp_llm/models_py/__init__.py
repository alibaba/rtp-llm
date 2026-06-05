from rtp_llm.models_py.model_loader import LoadConfig, NewModelLoader
from rtp_llm.models_py.module_base import rtp_module
from rtp_llm.models_py.new_models.qwen2_vl import Qwen2VLForConditionalGeneration
from rtp_llm.models_py.new_models.qwen2_vl.language import Qwen2ForCausalLM
from rtp_llm.models_py.new_models.qwen3 import Qwen3ForCausalLM
from rtp_llm.models_py.registry import MODEL_REGISTRY, get_model_class, register_model

register_model("qwen2_vl")(Qwen2VLForConditionalGeneration)
register_model("qwen_2")(Qwen2ForCausalLM)
register_model("qwen_3")(Qwen3ForCausalLM)
register_model("qwen_3_tool")(Qwen3ForCausalLM)

__all__ = [
    "MODEL_REGISTRY",
    "register_model",
    "get_model_class",
    "NewModelLoader",
    "LoadConfig",
    "rtp_module",
    "Qwen2VLForConditionalGeneration",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
]
