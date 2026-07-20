import importlib
from typing import Any, Dict

from rtp_llm.model_factory_register import ensure_all_models_registered

_CLASS_TO_MODULE: Dict[str, str] = {
    "BaseModel": "rtp_llm.models.base_model",
    "Bert": "rtp_llm.models.bert",
    "Bloom": "rtp_llm.models.bloom",
    "ChatGlmV2": "rtp_llm.models.chat_glm_v2",
    "ChatGlmV3": "rtp_llm.models.chat_glm_v3",
    "ChatGlmV4": "rtp_llm.models.chat_glm_v4",
    "ChatGlmV4Vision": "rtp_llm.models.chat_glm_v4_vision",
    "CosyVoiceQwen": "rtp_llm.models.cosyvoice_qwen",
    "DeepSeekV2": "rtp_llm.models.deepseek_v2",
    "DeepSeekV3Mtp": "rtp_llm.models.deepseek_v2",
    "DeepSeekVLV2": "rtp_llm.models.deepseek_vl2.deepseek_vl2",
    "Falcon": "rtp_llm.models.falcon",
    "GPTNeox": "rtp_llm.models.gpt_neox",
    "Glm4Moe": "rtp_llm.models.glm4_moe",
    "Glm4MoeLite": "rtp_llm.models.glm4_moe_lite",
    "JinaBert": "rtp_llm.models.jina_bert.jina_bert",
    "KimiK25": "rtp_llm.models.kimi_k25.kimi_k25",
    "KimiLinear": "rtp_llm.models.kimi_linear.kimi_linear",
    "Llama": "rtp_llm.models.llama",
    "Baichuan": "rtp_llm.models.llama",
    "Llava": "rtp_llm.models.llava",
    "MegatronBert": "rtp_llm.models.megatron_bert",
    "Mixtral": "rtp_llm.models.mixtral",
    "Mpt": "rtp_llm.models.mpt",
    "Phi": "rtp_llm.models.phi",
    "QWen_1B8": "rtp_llm.models.qwen",
    "QWen_7B": "rtp_llm.models.qwen",
    "QWen_13B": "rtp_llm.models.qwen",
    "QWen_VL": "rtp_llm.models.qwen_vl",
    "QWen2_VL": "rtp_llm.models.qwen2_vl",
    "QWenV2": "rtp_llm.models.qwen_v2",
    "QWenV2Audio": "rtp_llm.models.qwen_v2_audio",
    "Qwen2Moe": "rtp_llm.models.qwen_v2_moe",
    "Qwen3Moe": "rtp_llm.models.qwen_v3_moe",
    "Qwen3Next": "rtp_llm.models.qwen3_next.qwen3_next",
    "Qwen3NextMTP": "rtp_llm.models.qwen3_next.qwen3_next_mtp",
    "QWen3_VL": "rtp_llm.models.qwen3_vl",
    "QWen3_VL_MOE": "rtp_llm.models.qwen3_vl_moe",
    "QwenV3": "rtp_llm.models.qwen_v3",
    "StarCoder": "rtp_llm.models.starcoder",
    "StarCoder2": "rtp_llm.models.starcoder2",
}

__all__ = sorted(_CLASS_TO_MODULE) + ["load_all_models"]


def load_all_models() -> None:
    ensure_all_models_registered()


def __getattr__(name: str) -> Any:
    module_path = _CLASS_TO_MODULE.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value
