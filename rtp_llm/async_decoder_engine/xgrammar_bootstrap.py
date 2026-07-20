from __future__ import annotations

import logging
from typing import Any, List


def _collect_stop_token_ids(model: Any) -> List[int]:
    ids: set[int] = set()
    special_tokens = model.model_config.special_tokens
    eos_token_id = getattr(special_tokens, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        ids.add(eos_token_id)
    for seq in getattr(special_tokens, "stop_words_id_list", None) or []:
        if isinstance(seq, (list, tuple)) and len(seq) == 1:
            ids.add(int(seq[0]))
    return sorted(ids)


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
        stop_token_ids = _collect_stop_token_ids(model)

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
