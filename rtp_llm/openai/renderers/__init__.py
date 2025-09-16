import logging

from .chatglm4_renderer import ChatGlm4Renderer
from .chatglm45_renderer import ChatGlm45Renderer
from .deepseekv31_renderer import DeepseekV31Renderer
from .kimik2_renderer import KimiK2Renderer
from .llava_renderer import LlavaRenderer
from .qwen3_code_renderer import Qwen3CoderRenderer
from .qwen_agent_renderer import QwenAgentRenderer
from .qwen_agent_tool_renderer import QwenAgentToolRenderer
from .qwen_reasoning_tool_renderer import QwenReasoningToolRenderer
from .qwen_renderer import QwenRenderer
from .qwen_v2_audio_renderer import QwenV2AudioRenderer
from .qwen_vl_renderer import QwenVLRenderer

try:
    from internal_source.rtp_llm.openai_renderers import internal_init
except ImportError as e:
    logging.info(f"no internal_source found, error: {str(e)}")
    pass
