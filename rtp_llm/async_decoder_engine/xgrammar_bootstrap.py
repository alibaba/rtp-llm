from __future__ import annotations

import logging
from typing import Any, List

_XGRAMMAR_BYTE_LEVEL_VOCAB = 2


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

        backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
        if backend_tokenizer is not None and callable(
            getattr(backend_tokenizer, "to_str", None)
        ):
            from rtp_llm.ops import build_xgrammar_tokenizer_info_json

            grammar_config.tokenizer_info_json = build_xgrammar_tokenizer_info_json(
                vocab,
                backend_tokenizer.to_str(),
                vocab_size,
                stop_token_ids,
            )
            tokenizer_kind = "huggingface_fast"
        else:
            byte_decoder = getattr(tokenizer, "byte_decoder", None)
            tokenizer_model = getattr(tokenizer, "model", None)
            if not isinstance(byte_decoder, dict) or not callable(
                getattr(tokenizer_model, "decode_single_token_bytes", None)
            ):
                raise TypeError(
                    "tokenizer exposes neither a Hugging Face backend_tokenizer "
                    "nor a tiktoken byte-level vocabulary"
                )

            from rtp_llm.ops import build_xgrammar_tokenizer_info_json_from_vocab

            grammar_config.tokenizer_info_json = (
                build_xgrammar_tokenizer_info_json_from_vocab(
                    vocab,
                    _XGRAMMAR_BYTE_LEVEL_VOCAB,
                    vocab_size,
                    stop_token_ids,
                    False,
                )
            )
            tokenizer_kind = "tiktoken_byte_level"
        logging.info(
            "xgrammar bootstrap: tokenizer=%s vocab_size=%d "
            "tokenizer_info_json=%dB stop_token_ids=%s",
            tokenizer_kind,
            vocab_size,
            len(grammar_config.tokenizer_info_json),
            stop_token_ids,
        )
    except Exception as e:
        logging.warning("xgrammar bootstrap failed (%s); grammar disabled", e)
        grammar_config.tokenizer_info_json = ""
