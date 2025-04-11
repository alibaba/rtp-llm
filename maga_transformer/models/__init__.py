from .base_model import BaseModel
from .gpt_neox import GPTNeox
from .llama import Llama, Baichuan
from .sgpt_bloom import SGPTBloom
from .sgpt_bloom_vector import SGPTBloomVector
from .starcoder import StarCoder
from .starcoder2 import StarCoder2
from .bloom import Bloom
from .chat_glm_v2 import ChatGlmV2
from .chat_glm_v3 import ChatGlmV3
from .chat_glm_v4 import ChatGlmV4
from .qwen import QWen_7B, QWen_13B, QWen_1B8
from .qwen_v2 import QWenV2
from .falcon import Falcon
from .mpt import Mpt
from .phi import Phi
from .deepseek_v2 import DeepSeekV2, DeepSeekV3Mtp
from .cosyvoice_qwen import CosyVoiceQwen

import platform
if platform.processor() != 'aarch64':
    from .chat_glm_v4_vision import ChatGlmV4Vision
    from .llava import Llava
    from .qwen_vl import QWen_VL
    from .qwen2_vl.qwen2_vl import QWen2_VL
    from .cogvlm2 import CogVLM2
    from .qwen_v2_audio.qwen_v2_audio import QWenV2Audio
    from .internvl import InternVL
    from .minicpmv.minicpmv import MiniCPMV
    from .minicpmv_embedding.minicpmv_embedding import MiniCPMVEmbedding

from .mixtral import Mixtral
from .bert import Bert
from .jina_bert.jina_bert import JinaBert
from .megatron_bert import MegatronBert
from .qwen_v2_moe import Qwen2Moe

import logging
try:
    from internal_source.maga_transformer.models import internal_init
except ImportError as e:
    logging.info(f"import internal source failed, error: {str(e)}")
    pass
