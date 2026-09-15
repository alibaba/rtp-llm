import importlib
import logging
import os

from transformers.utils import is_flash_attn_2_available

from rtp_llm.utils.flash_attn_utils import can_use_flash_attn


def get_vit_attn_implementation():
    """Select the Hugging Face Qwen3-VL vision attention backend at model load."""
    requested = os.environ.get("VIT_ATTN_IMPLEMENTATION", "auto")
    supported = ("auto", "sdpa", "eager", "flash_attention_2")
    if requested not in supported:
        raise ValueError(
            f"Invalid VIT_ATTN_IMPLEMENTATION={requested!r}; expected one of {supported}"
        )

    selected = requested
    reason = "explicit_config"
    if requested in ("auto", "flash_attention_2"):
        error = None
        try:
            if not can_use_flash_attn():
                reason = "unsupported_device"
            elif not is_flash_attn_2_available():
                reason = "flash_attn_unavailable"
            else:
                # Package metadata alone cannot detect a broken native extension.
                flash_attn = importlib.import_module("flash_attn")
                if not callable(flash_attn.flash_attn_varlen_func):
                    raise ImportError("flash_attn_varlen_func is not callable")
                reason = "flash_attn_available"
        except Exception as exc:
            error = exc
            reason = "flash_attn_check_failed"
            logging.info("Qwen3-VL ViT FlashAttention2 check failed: %s", exc)

        selected = "flash_attention_2" if reason == "flash_attn_available" else "sdpa"
        if requested == "flash_attention_2" and selected != requested:
            raise RuntimeError(
                f"VIT_ATTN_IMPLEMENTATION=flash_attention_2 cannot be used: {reason}. "
                "Use VIT_ATTN_IMPLEMENTATION=auto or sdpa to allow loading without FA2."
            ) from error

    logging.info(
        "Qwen3-VL ViT attention: requested=%s selected=%s reason=%s",
        requested,
        selected,
        reason,
    )
    return selected
