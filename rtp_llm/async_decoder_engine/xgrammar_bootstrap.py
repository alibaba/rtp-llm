"""Engine-init xgrammar bootstrap: pokes the HF tokenizer once and stuffs
``tokenizer_info_json`` / ``think_end_id`` onto ``engine_config.grammar_config``
so the C++ engine can build XGrammarBackendCpp with no further pybind hop.
"""
from __future__ import annotations

import logging
from typing import Any, List


def _resolve_think_end_id(rtype: str, tokenizer: Any) -> int:
    try:
        from rtp_llm.openai.renderers.sglang_helpers.reasoning_parser import (
            ReasoningParser,
        )

        token = ReasoningParser(model_type=rtype).detector.think_end_token
        ids = tokenizer.encode(token, add_special_tokens=False)
    except Exception as e:  # noqa: BLE001
        logging.warning("reasoner_grammar: think_end_id resolve failed (%s); reasoner disabled", e)
        return -1
    if len(ids) != 1:
        logging.warning(
            "reasoner_grammar: think_end_token=%s encoded to %d tokens; reasoner disabled",
            token,
            len(ids),
        )
        return -1
    return int(ids[0])


def _collect_stop_token_ids(model: Any) -> List[int]:
    """Union of every token id the model treats as a legitimate stop, so
    xgrammar accepts ALL of them at grammar-terminal states.
    tokenizer.eos_token_id alone misses extras from generation_config
    (Qwen3 151643, Llama-3 128009, ...); special_tokens has already merged
    hf_config + generation_config + tokenizer + env overrides at
    model_config build time, so it is the single authoritative source.
    """
    ids: set[int] = set[int]()
    st = model.model_config.special_tokens
    if isinstance(st.eos_token_id, int):
        ids.add(st.eos_token_id)
    for seq in st.stop_words_id_list or []:
        # Multi-token sequences can't be registered as xgrammar stop tokens.
        if isinstance(seq, (list, tuple)) and len(seq) == 1:
            ids.add(int(seq[0]))
    return sorted(ids)


def bootstrap_grammar_config(engine_config: Any, model: Any) -> None:
    gc = engine_config.grammar_config
    if (gc.grammar_backend or "").strip().lower() in ("", "none"):
        gc.tokenizer_info_json = ""
        return

    tokenizer = model.tokenizer.tokenizer
    try:
        vocab = tokenizer.get_vocab()
        total = max(int(model.model_config.vocab_size),
                    (max(vocab.values()) + 1) if vocab else 0)
        stop_token_ids = _collect_stop_token_ids(model)

        from rtp_llm.ops import build_xgrammar_tokenizer_info_json

        gc.tokenizer_info_json = build_xgrammar_tokenizer_info_json(
            vocab, tokenizer.backend_tokenizer.to_str(), total, stop_token_ids
        )
    except Exception as e:  # noqa: BLE001
        logging.warning("xgrammar bootstrap failed (%s); grammar disabled", e)
        gc.tokenizer_info_json = ""
        return

    if gc.reasoning_parser:
        gc.think_end_id = _resolve_think_end_id(gc.reasoning_parser, tokenizer)

    logging.info(
        "grammar bootstrap: vocab=%d, json=%dB, think_end_id=%d, stop_tokens=%s",
        total,
        len(gc.tokenizer_info_json),
        gc.think_end_id,
        stop_token_ids,
    )
