from typing import Any

from rtp_llm.config.generate_config import RequestFormat

_GLM5_RAW_CHAT_MODELS = {"glm_5"}
_GLM5_PREFIX = "[gMASK]<sop>"
_GLM5_ASSISTANT_SUFFIX = "<|assistant|><think></think>"
_GLM5_CHAT_MARKERS = ("<|user|>", "<|assistant|>", "<|system|>", "<|observation|>")


def maybe_wrap_glm5_raw_prompt(model_type: str, text: Any, request_format: str) -> Any:
    if model_type not in _GLM5_RAW_CHAT_MODELS:
        return text
    if request_format != RequestFormat.RAW or not isinstance(text, str):
        return text
    if not text.strip():
        return text
    if any(marker in text for marker in _GLM5_CHAT_MARKERS):
        return text if text.startswith(_GLM5_PREFIX) else f"{_GLM5_PREFIX}{text}"
    return f"{_GLM5_PREFIX}<|user|>{text}{_GLM5_ASSISTANT_SUFFIX}"
