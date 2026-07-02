from rtp_llm.utils.import_util import has_internal_source

from .chatglm4_renderer import ChatGlm4Renderer
from .chatglm45_renderer import ChatGlm45Renderer
from .chatglm47_renderer import ChatGlm47Renderer
from .deepseek_vl2_renderer import DeepSeekVLV2Renderer
from .deepseekv31_renderer import DeepseekV31Renderer
from .deepseekv32_renderer import DeepseekV32Renderer
from .kimi_k25_renderer import KimiK25Renderer
from .kimik2_renderer import KimiK2Renderer
from .llava_renderer import LlavaRenderer
from .qwen3_code_renderer import Qwen3CoderRenderer
from .qwen35_renderer import Qwen35Renderer
from .qwen_agent_renderer import QwenAgentRenderer
from .qwen_agent_tool_renderer import QwenAgentToolRenderer
from .qwen_reasoning_tool_renderer import QwenReasoningToolRenderer
from .qwen_renderer import QwenRenderer
from .qwen_v2_audio_renderer import QwenV2AudioRenderer
from .qwen_vl_renderer import QwenVLRenderer

if has_internal_source():
    # Phase-25 namespace merge: rtp_llm.openai_renderers resolves to
    # internal_source/rtp_llm/openai_renderers via the extended __path__.
    from rtp_llm.openai_renderers import internal_init  # noqa: F401
