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


def _resolve_excluded_token_ids(model: Any) -> List[int]:
    """per-model: 把模型声明的 thinking 期拦截 token(字符串)编码成 token id。

    只收编码成**单个 token** 的(否则无法做单 token rewrite),其余告警跳过。marker 多为
    全角 special token,直接原文 encode。
    """
    get_strs = getattr(model, "get_think_excluded_token_strs", None)
    if not callable(get_strs):
        return []
    strs = get_strs() or []
    if not strs:
        return []
    tokenizer = getattr(getattr(model, "tokenizer", None), "tokenizer", None)
    if tokenizer is None:
        return []
    ids: List[int] = []
    for s in strs:
        try:
            encoded = tokenizer.encode(s, add_special_tokens=False)
        except Exception as e:
            logging.warning("excluded token %r encode failed (%s); skipped", s, e)
            continue
        if len(encoded) == 1:
            ids.append(int(encoded[0]))
        else:
            logging.warning(
                "excluded token %r did not encode to a single token (got %s); skipped",
                s,
                encoded,
            )
    return ids


def _strict_thinking_enabled(model: Any) -> bool:
    """服务启动开关:GenerateEnvConfig.enable_strict_thinking。一个服务只加载一个模型,
    该开关决定这个模型要不要 strict thinking。"""
    env_cfg = getattr(getattr(model, "model_config", None), "generate_env_config", None)
    return bool(getattr(env_cfg, "enable_strict_thinking", False))


def bootstrap_grammar_config(engine_config: Any, model: Any) -> None:
    grammar_config = engine_config.grammar_config
    # thinking 期 intercepted token 由服务启动开关 enable_strict_thinking 控制,与 grammar
    # 后端独立(后端关掉也生效);默认关 → 空 → 不 rewrite。
    strict_thinking = _strict_thinking_enabled(model)
    grammar_config.excluded_token_ids = (
        _resolve_excluded_token_ids(model) if strict_thinking else []
    )
    logging.info(
        "strict_thinking=%s excluded_token_ids=%s",
        strict_thinking,
        list(grammar_config.excluded_token_ids),
    )
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
