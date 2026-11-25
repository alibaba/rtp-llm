from __future__ import annotations

import logging
from typing import Any

from rtp_llm.config.grammar_tokenizer_info import grammar_stop_token_ids
from rtp_llm.config.stop_words import resolve_stop_word_ids


def bootstrap_grammar_config(engine_config: Any, model: Any) -> None:
    grammar_config = engine_config.grammar_config
    if (grammar_config.grammar_backend or "").strip().lower() in ("", "none"):
        grammar_config.tokenizer_info_json = ""
        return

    tokenizer = model.tokenizer.tokenizer
    try:
        vocab = tokenizer.get_vocab()
        vocab_size = max(
            int(model.model_config.vocab_size),
            (max(vocab.values()) + 1) if vocab else 0,
        )
        stop_words = resolve_stop_word_ids(
            model.tokenizer,
            model.model_config,
            engine_config.misc_config,
            model.vit_config,
        )
        stop_token_ids = grammar_stop_token_ids(
            model.tokenizer,
            model.model_config,
            stop_words,
            grammar_config.override_stop_tokens,
        )

        from rtp_llm.ops import build_xgrammar_tokenizer_info_json

        grammar_config.tokenizer_info_json = build_xgrammar_tokenizer_info_json(
            vocab,
            tokenizer.backend_tokenizer.to_str(),
            vocab_size,
            stop_token_ids,
        )
        logging.info(
            "xgrammar bootstrap: vocab_size=%d tokenizer_info_json=%dB stop_token_ids=%s",
            vocab_size,
            len(grammar_config.tokenizer_info_json),
            stop_token_ids,
        )
    except Exception as e:
        logging.warning("xgrammar bootstrap failed (%s); grammar disabled", e)
        grammar_config.tokenizer_info_json = ""
