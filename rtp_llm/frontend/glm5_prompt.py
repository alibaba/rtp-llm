from typing import Any

from rtp_llm.config.generate_config import RequestFormat

_GLM5_RAW_CHAT_MODELS = {"glm_5"}
_GLM5_CHAT_MARKERS = ("<|user|>", "<|assistant|>", "<|system|>", "<|observation|>")


def maybe_wrap_glm5_raw_prompt(model_type: str, text: Any, request_format: str) -> Any:
    if model_type not in _GLM5_RAW_CHAT_MODELS:
        return text
    if request_format != RequestFormat.RAW or not isinstance(text, str):
        return text
    if not text.strip() or any(marker in text for marker in _GLM5_CHAT_MARKERS):
        return text
    return f"<|user|>{text}\n<|assistant|><think></think>"
