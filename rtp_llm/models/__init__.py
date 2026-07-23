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
    # REBASE CONFLICT CONTEXT(e2e00e570): source branch eagerly imported
    # `Glm5Mtp` from deepseek_v2; new base uses lazy model imports. Keep the
    # lazy table and add the GLM5 MTP symbol here instead of restoring eager imports.
    "Glm5Mtp": "rtp_llm.models.deepseek_v2",
    "DeepSeekV4": "rtp_llm.models.deepseek_v4",
    "DeepSeekV4Mtp": "rtp_llm.models.deepseek_v4",
    "DeepSeekVLV2": "rtp_llm.models.deepseek_vl2.deepseek_vl2",
    "Falcon": "rtp_llm.models.falcon",
    "GPTNeox": "rtp_llm.models.gpt_neox",
    "Glm4Moe": "rtp_llm.models.glm4_moe",
    "InternVL": "rtp_llm.models.internvl",
    "JinaBert": "rtp_llm.models.jina_bert.jina_bert",
    "Llama": "rtp_llm.models.llama",
    "MiniMaxM3Eagle1": "rtp_llm.models.minimax_m3_eagle1",
    "MiniMaxM3Eagle3": "rtp_llm.models.minimax_m3_eagle3",
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

from .bert import Bert
from .glm4_moe import Glm4Moe
from .jina_bert.jina_bert import JinaBert
from .megatron_bert import MegatronBert
from .minimax_m3 import MiniMaxM3
from .minimax_m3_vl import MiniMaxM3_VL
from .mixtral import Mixtral
from .qwen3_next.qwen3_next import Qwen3Next
from .qwen3_next.qwen3_next_mtp import Qwen3NextMTP
from .qwen_v2_moe import Qwen2Moe
from .qwen_v3_moe import Qwen3Moe


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
