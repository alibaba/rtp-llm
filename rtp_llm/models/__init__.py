import platform

from .base_model import BaseModel
from .bloom import Bloom
from .chat_glm_v2 import ChatGlmV2
from .chat_glm_v3 import ChatGlmV3
from .chat_glm_v4 import ChatGlmV4
from .cosyvoice_qwen import CosyVoiceQwen
from .deepseek_v2 import DeepSeekV2, DeepSeekV3Mtp
from .deepseek_vl2.deepseek_vl2 import DeepSeekVLV2
from .falcon import Falcon
from .gpt_neox import GPTNeox
from .llama import Baichuan, Llama
from .mpt import Mpt
from .phi import Phi
from .qwen import QWen_1B8, QWen_7B, QWen_13B
from .qwen_v2 import QWenV2
from .qwen_v3 import QwenV3
from .starcoder import StarCoder
from .starcoder2 import StarCoder2

if platform.processor() != "aarch64":
    from .chat_glm_v4_vision import ChatGlmV4Vision
    from .kimi_k25.kimi_k25 import KimiK25
    from .llava import Llava
    from .qwen_vl import QWen_VL
    from .qwen2_vl import QWen2_VL
    from .qwen_v2_audio import QWenV2Audio
    from .qwen3_vl import QWen3_VL
    from .qwen3_vl_moe import QWen3_VL_MOE

from rtp_llm.utils.import_util import has_internal_source

from .bert import Bert
from .glm4_moe import Glm4Moe
from .glm4_moe_lite import Glm4MoeLite
from .jina_bert.jina_bert import JinaBert
from .kimi_linear.kimi_linear import KimiLinear
from .megatron_bert import MegatronBert
from .mixtral import Mixtral
from .qwen3_next.qwen3_next import Qwen3Next
from .qwen3_next.qwen3_next_mtp import Qwen3NextMTP
from .qwen_v2_moe import Qwen2Moe
from .qwen_v3_moe import Qwen3Moe

if has_internal_source():
    import internal_source.rtp_llm.models.internal_init

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