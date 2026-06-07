import importlib
from typing import Any, Dict

from rtp_llm.openai.renderer_factory_register import ensure_all_renderers_registered

_CLASS_TO_MODULE: Dict[str, str] = {
    "ChatGlm4Renderer": "rtp_llm.openai.renderers.chatglm4_renderer",
    "ChatGlm45Renderer": "rtp_llm.openai.renderers.chatglm45_renderer",
    "ChatGlm47Renderer": "rtp_llm.openai.renderers.chatglm47_renderer",
    "DeepSeekVLV2Renderer": "rtp_llm.openai.renderers.deepseek_vl2_renderer",
    "DeepseekV31Renderer": "rtp_llm.openai.renderers.deepseekv31_renderer",
    "DeepseekV32Renderer": "rtp_llm.openai.renderers.deepseekv32_renderer",
    "InternVLRenderer": "rtp_llm.openai.renderers.internvl_renderer",
    "KimiK2Renderer": "rtp_llm.openai.renderers.kimik2_renderer",
    "KimiK25Renderer": "rtp_llm.openai.renderers.kimi_k25_renderer",
    "LlavaRenderer": "rtp_llm.openai.renderers.llava_renderer",
    "MiniCPMVRenderer": "rtp_llm.openai.renderers.minicpmv_renderer",
    "Qwen3CoderRenderer": "rtp_llm.openai.renderers.qwen3_code_renderer",
    "QwenAgentRenderer": "rtp_llm.openai.renderers.qwen_agent_renderer",
    "QwenAgentToolRenderer": "rtp_llm.openai.renderers.qwen_agent_tool_renderer",
    "QwenReasoningToolRenderer": "rtp_llm.openai.renderers.qwen_reasoning_tool_renderer",
    "QwenRenderer": "rtp_llm.openai.renderers.qwen_renderer",
    "QwenV2AudioRenderer": "rtp_llm.openai.renderers.qwen_v2_audio_renderer",
    "QwenVLRenderer": "rtp_llm.openai.renderers.qwen_vl_renderer",
}

__all__ = sorted(_CLASS_TO_MODULE) + ["load_all_renderers"]


def load_all_renderers() -> None:
    ensure_all_renderers_registered()


def __getattr__(name: str) -> Any:
    module_path = _CLASS_TO_MODULE.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value
