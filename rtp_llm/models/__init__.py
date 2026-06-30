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
    "DeepSeekV4": "rtp_llm.models.deepseek_v4",
    "DeepSeekV4Mtp": "rtp_llm.models.deepseek_v4",
    "DeepSeekVLV2": "rtp_llm.models.deepseek_vl2.deepseek_vl2",
    "Falcon": "rtp_llm.models.falcon",
    "GPTNeox": "rtp_llm.models.gpt_neox",
    "Glm4Moe": "rtp_llm.models.glm4_moe",
    "InternVL": "rtp_llm.models.internvl",
    "JinaBert": "rtp_llm.models.jina_bert.jina_bert",
    "Llama": "rtp_llm.models.llama",
    "Baichuan": "rtp_llm.models.llama",
    "Llava": "rtp_llm.models.llava",
    "MegatronBert": "rtp_llm.models.megatron_bert",
    "MiniCPMV": "rtp_llm.models.minicpmv.minicpmv",
    "MiniCPMVEmbedding": "rtp_llm.models.minicpmv_embedding.minicpmv_embedding",
    "Mixtral": "rtp_llm.models.mixtral",
    "Mpt": "rtp_llm.models.mpt",
    "Phi": "rtp_llm.models.phi",
    "QWen_1B8": "rtp_llm.models.qwen",
    "QWen_7B": "rtp_llm.models.qwen",
    "QWen_13B": "rtp_llm.models.qwen",
    "QWen_VL": "rtp_llm.models.qwen_vl",
    "QWen2_5_VL": "rtp_llm.models.qwen2_5_vl.qwen2_5_vl",
    "QWen2_VL": "rtp_llm.models.qwen2_vl.qwen2_vl",
    "QWenV2": "rtp_llm.models.qwen_v2",
    "QWenV2Audio": "rtp_llm.models.qwen_v2_audio.qwen_v2_audio",
    "Qwen2Moe": "rtp_llm.models.qwen_v2_moe",
    "Qwen3Moe": "rtp_llm.models.qwen_v3_moe",
    "Qwen3Next": "rtp_llm.models.qwen3_next.qwen3_next",
    "Qwen3NextMTP": "rtp_llm.models.qwen3_next.qwen3_next_mtp",
    "Qwen3_VL_MOE": "rtp_llm.models.qwen3_vl_moe.qwen3_vl_moe",
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


import os
import logging
import importlib
from rtp_llm.utils.import_util import load_module

def _get_csv_env_values(env_name: str):
    raw_value = os.getenv(env_name, "")
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]

def _load_external_models_from_env() -> None:
    # atom.plugin.rtp_llm.models
    external_modules = _get_csv_env_values("RTP_LLM_EXTERNAL_MODEL_PACKAGES")
    for module_name in external_modules:
        logging.info("loading external model module: %s", module_name)
        try:
            importlib.import_module(module_name)
        except Exception as e:
            raise RuntimeError(
                f"failed to import external model module [{module_name}]: {e}"
            ) from e

    external_files = _get_csv_env_values("RTP_LLM_EXTERNAL_MODEL_FILES")
    for module_file in external_files:
        logging.info("loading external model file: %s", module_file)
        try:
            load_module(module_file)
        except Exception as e:
            raise RuntimeError(
                f"failed to import external model file [{module_file}]: {e}"
            ) from e

_load_external_models_from_env()
