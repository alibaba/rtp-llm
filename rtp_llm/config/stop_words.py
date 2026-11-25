"""Resolve model-wide stop sequences for Dash SC and grammar compilation."""

import json
import logging
from typing import Any, List, Optional


def primary_eos_id(tokenizer: Any, model_config: Any) -> Optional[int]:
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = model_config.special_tokens.eos_token_id
    if isinstance(eos, (list, tuple)):
        return eos[0] if eos else None
    return eos


def checkpoint_eos_ids(tokenizer: Any) -> List[int]:
    ids = tokenizer.config_json.get("eos_token_id", [])
    return [ids] if isinstance(ids, int) else list(ids or [])


def resolve_stop_word_ids(
    tokenizer: Any, model_config: Any, misc_config: Any, vit_config: Any
) -> List[List[int]]:
    """Apply model/renderer defaults and the same startup stop policy as HTTP."""
    from rtp_llm.config.py_config_modules import GenerateEnvConfig, RenderConfig

    config = model_config.generate_env_config or GenerateEnvConfig()
    special = model_config.special_tokens
    try:
        startup_ids = json.loads(config.stop_words_list or "[]")
        startup_strings = json.loads(config.stop_words_str or "[]")
    except (TypeError, ValueError):
        logging.warning("Invalid startup stop words", exc_info=True)
        return []

    sequences: List[List[int]] = []
    strings: List[str] = []
    if not config.force_stop_words:
        from rtp_llm.openai.renderer_factory import ChatRendererFactory
        from rtp_llm.openai.renderers.custom_renderer import RendererParams

        sequences.extend(special.stop_words_id_list)
        strings.extend(special.stop_words_str_list)
        primary_eos = primary_eos_id(tokenizer, model_config)
        sequences.extend([i] for i in checkpoint_eos_ids(tokenizer) if i != primary_eos)
        params = RendererParams(
            model_type=model_config.model_type,
            max_seq_len=model_config.max_seq_len,
            eos_token_id=primary_eos,
            stop_word_ids_list=list(special.stop_words_id_list),
            template_type=model_config.template_type,
            ckpt_path=model_config.ckpt_path,
        )
        try:
            renderer = ChatRendererFactory.get_renderer(
                tokenizer,
                params,
                config,
                model_config.render_config or RenderConfig(),
                model_config.ckpt_path,
                misc_config,
                vit_config,
            )
            sequences.extend(renderer.get_all_extra_stop_word_ids_list())
        except Exception:
            logging.warning("Could not resolve renderer stop words", exc_info=True)

    sequences.extend(startup_ids)
    strings.extend(startup_strings)
    for word in strings:
        try:
            sequences.append(tokenizer.encode(word))
        except Exception:
            logging.warning("Could not tokenize stop word %r", word, exc_info=True)

    # Keep multi-token stops intact, even when their decoded text has a shorter encoding.
    return [list(ids) for ids in dict.fromkeys(tuple(s) for s in sequences if s)]
