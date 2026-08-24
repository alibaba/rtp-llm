from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

_XGRAMMAR_BYTE_LEVEL_VOCAB = 2


def _clear_pretokenized_chat_constraints(grammar_config: Any) -> None:
    grammar_config.reasoning_prompt_tail_token_ids = []
    grammar_config.response_prompt_tail_token_ids = []
    grammar_config.reasoning_structural_tag = ""
    grammar_config.response_structural_tag = ""
    grammar_config.reasoning_completion_boundary_token_ids = []
    grammar_config.response_completion_boundary_token_ids = []
    grammar_config.completion_think_close_token_ids = []
    grammar_config.completion_response_open_token_ids = []
    grammar_config.completion_response_close_token_ids = []
    grammar_config.completion_tools_open_token_ids = []
    grammar_config.completion_tools_close_token_ids = []
    grammar_config.completion_whitespace_token_ids = []


def _encode_prompt_tail(tokenizer: Any, text: str) -> List[int]:
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(text)
    return [int(token_id) for token_id in token_ids]


def _collect_whitespace_token_ids(tokenizer: Any) -> List[int]:
    whitespace_ids: set[int] = set()
    for text in (" ", "\t", "\n", "\r", "\r\n", "\n\n"):
        token_ids = _encode_prompt_tail(tokenizer, text)
        if len(token_ids) == 1:
            whitespace_ids.add(token_ids[0])

    decode_bytes = getattr(
        getattr(tokenizer, "model", None), "decode_single_token_bytes", None
    )
    if not callable(decode_bytes):
        return sorted(whitespace_ids)

    for token_id in set(tokenizer.get_vocab().values()):
        try:
            text = decode_bytes(int(token_id)).decode("utf-8")
        except (UnicodeDecodeError, KeyError, ValueError):
            continue
        if text and text.isspace():
            whitespace_ids.add(int(token_id))
    return sorted(whitespace_ids)


def _encode_completion_guard(tokenizer: Any, value: Any) -> Dict[str, List[int]]:
    if not isinstance(value, dict):
        return {}
    encoded: Dict[str, List[int]] = {}
    for name in (
        "think_close",
        "response_open",
        "response_close",
        "tools_open",
        "tools_close",
        "message_close",
    ):
        text = value.get(name, "")
        if not isinstance(text, str):
            raise TypeError(f"completion_guard.{name} must be a string")
        encoded[name] = _encode_prompt_tail(tokenizer, text)
    return encoded


def _bootstrap_pretokenized_chat_constraints(
    grammar_config: Any, model: Any, tokenizer: Any
) -> None:
    """Load optional renderer-declared defaults for already-tokenized chat RPCs."""
    from rtp_llm.openai.renderer_factory_register import get_renderer_class

    model_type = getattr(model.model_config, "model_type", None)
    renderer_class = get_renderer_class(model_type)
    provider = getattr(renderer_class, "pretokenized_chat_constraints", None)
    if not callable(provider):
        return

    constraints: Dict[str, Dict[str, Any]] = provider()
    reasoning = constraints.get("reasoning", {})
    response = constraints.get("response", {})
    reasoning_tail = _encode_prompt_tail(tokenizer, reasoning["prompt_tail"])
    response_tail = _encode_prompt_tail(tokenizer, response["prompt_tail"])
    if not reasoning_tail or not response_tail or reasoning_tail == response_tail:
        raise ValueError(
            "invalid pretokenized chat prompt tails: "
            f"reasoning={reasoning_tail} response={response_tail}"
        )

    grammar_config.reasoning_prompt_tail_token_ids = reasoning_tail
    grammar_config.response_prompt_tail_token_ids = response_tail
    reasoning_tag = reasoning.get("structural_tag")
    response_tag = response.get("structural_tag")
    grammar_config.reasoning_structural_tag = (
        json.dumps(reasoning_tag, ensure_ascii=False, separators=(",", ":"))
        if reasoning_tag is not None
        else ""
    )
    grammar_config.response_structural_tag = (
        json.dumps(response_tag, ensure_ascii=False, separators=(",", ":"))
        if response_tag is not None
        else ""
    )
    reasoning_boundary = _encode_prompt_tail(
        tokenizer, reasoning.get("completion_boundary", "")
    )
    response_boundary = _encode_prompt_tail(
        tokenizer, response.get("completion_boundary", "")
    )
    reasoning_guard = _encode_completion_guard(
        tokenizer, reasoning.get("completion_guard")
    )
    response_guard = _encode_completion_guard(
        tokenizer, response.get("completion_guard")
    )
    if reasoning_guard or response_guard:
        if not reasoning_guard or not response_guard:
            raise ValueError("both pretokenized phases must declare completion_guard")
        for name in ("response_close", "tools_open", "tools_close", "message_close"):
            if reasoning_guard[name] != response_guard[name]:
                raise ValueError(f"completion_guard.{name} differs between phases")
        if not reasoning_guard["think_close"] or not reasoning_guard["response_open"]:
            raise ValueError(
                "reasoning completion_guard has no think/response transition"
            )
        if response_guard["think_close"] or response_guard["response_open"]:
            raise ValueError(
                "response completion_guard must start inside response body"
            )
        if (
            not reasoning_guard["response_close"]
            or not reasoning_guard["message_close"]
        ):
            raise ValueError("completion_guard has no response/message close")
        if bool(reasoning_guard["tools_open"]) != bool(reasoning_guard["tools_close"]):
            raise ValueError(
                "completion_guard tools open/close must be configured together"
            )
        reasoning_boundary = reasoning_guard["message_close"]
        response_boundary = response_guard["message_close"]
        grammar_config.completion_think_close_token_ids = reasoning_guard["think_close"]
        grammar_config.completion_response_open_token_ids = reasoning_guard[
            "response_open"
        ]
        grammar_config.completion_response_close_token_ids = reasoning_guard[
            "response_close"
        ]
        grammar_config.completion_tools_open_token_ids = reasoning_guard["tools_open"]
        grammar_config.completion_tools_close_token_ids = reasoning_guard["tools_close"]
        grammar_config.completion_whitespace_token_ids = _collect_whitespace_token_ids(
            tokenizer
        )
    if not (grammar_config.reasoning_structural_tag or reasoning_boundary):
        raise ValueError("reasoning pretokenized constraint has no tag or boundary")
    if not (grammar_config.response_structural_tag or response_boundary):
        raise ValueError("response pretokenized constraint has no tag or boundary")
    grammar_config.reasoning_completion_boundary_token_ids = reasoning_boundary
    grammar_config.response_completion_boundary_token_ids = response_boundary
    logging.info(
        "xgrammar pretokenized chat defaults: renderer=%s reasoning_tail=%s "
        "response_tail=%s reasoning_tag=%dB response_tag=%dB "
        "reasoning_boundary=%s response_boundary=%s",
        renderer_class.__name__,
        reasoning_tail,
        response_tail,
        len(grammar_config.reasoning_structural_tag),
        len(grammar_config.response_structural_tag),
        reasoning_boundary,
        response_boundary,
    )


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
    _clear_pretokenized_chat_constraints(grammar_config)
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
        try:
            _bootstrap_pretokenized_chat_constraints(grammar_config, model, tokenizer)
        except Exception as e:
            logging.warning(
                "xgrammar pretokenized chat defaults unavailable (%s); "
                "request-provided grammar remains enabled",
                e,
            )
            _clear_pretokenized_chat_constraints(grammar_config)
    except Exception as e:
        logging.warning("xgrammar bootstrap failed (%s); grammar disabled", e)
        grammar_config.tokenizer_info_json = ""
        _clear_pretokenized_chat_constraints(grammar_config)
