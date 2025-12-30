import platform

from .base_model import BaseModel
from .bloom import Bloom
from .chat_glm_v2 import ChatGlmV2
from .chat_glm_v3 import ChatGlmV3
from .chat_glm_v4 import ChatGlmV4
from .cosyvoice_qwen import CosyVoiceQwen
from .deepseek_v2 import DeepSeekV2, DeepSeekV3Mtp
from .falcon import Falcon
from .gpt_neox import GPTNeox
from .llama import Baichuan, Llama
from .mpt import Mpt
from .phi import Phi
from .qwen import QWen_1B8, QWen_7B, QWen_13B
from .qwen_v2 import QWenV2
from .qwen_v3 import QwenV3
from .sgpt_bloom import SGPTBloom
from .sgpt_bloom_vector import SGPTBloomVector
from .starcoder import StarCoder
from .starcoder2 import StarCoder2

if platform.processor() != "aarch64":
    from .chat_glm_v4_vision import ChatGlmV4Vision
    from .llava import Llava
    from .qwen_vl import QWen_VL
    from .qwen2_vl.qwen2_vl import QWen2_VL
    from .qwen2_5_vl.qwen2_5_vl import QWen2_5_VL
    from .qwen3_vl_moe.qwen3_vl_moe import QWen3_VL_MOE
    from .qwen_v2_audio.qwen_v2_audio import QWenV2Audio
    from .internvl import InternVL
    from .minicpmv.minicpmv import MiniCPMV
    from .minicpmv_embedding.minicpmv_embedding import MiniCPMVEmbedding

import logging

from .bert import Bert
from .glm4_moe import Glm4Moe
from .jina_bert.jina_bert import JinaBert
from .megatron_bert import MegatronBert
from .mixtral import Mixtral
from .qwen3_next.qwen3_next import Qwen3Next
from .qwen_v2_moe import Qwen2Moe
from .qwen_v3_moe import Qwen3Moe

try:
    from internal_source.rtp_llm.models import internal_init
except ImportError as e:
    logging.info(f"import internal source failed, error: {str(e)}")
    pass
