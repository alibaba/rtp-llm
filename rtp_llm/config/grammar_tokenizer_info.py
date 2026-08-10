from typing import Any, List, Sequence


def build_grammar_tokenizer_info_json(
    tokenizer: Any,
    *,
    model_vocab_size: int,
    stop_token_ids: Sequence[int],
) -> str:
    """Serialize tokenizer metadata through xgrammar's public Python API."""
    import xgrammar as xgr

    normalized_stop_ids = [int(token_id) for token_id in stop_token_ids]
    if not normalized_stop_ids:
        raise ValueError("stop_token_ids cannot be empty")
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer,
        vocab_size=int(model_vocab_size or 0) or None,
        stop_token_ids=normalized_stop_ids,
    )
    return tokenizer_info.serialize_json()


def build_model_grammar_tokenizer_info_json(
    tokenizer: Any,
    model_config: Any,
) -> str:
    """Build admission metadata from the tokenizer already loaded by Dash-SC."""
    real_tokenizer = tokenizer.get_real_tokenizer()
    if real_tokenizer is None:
        return ""

    stop_token_ids: List[int] = []

    def add_id(token_id: Any) -> None:
        if isinstance(token_id, (list, tuple)):
            token_id = token_id[0] if token_id else None
        if token_id is None:
            return
        token_id = int(token_id)
        if token_id >= 0 and token_id not in stop_token_ids:
            stop_token_ids.append(token_id)

    special_tokens = model_config.special_tokens
    add_id(getattr(tokenizer, "eos_token_id", None))
    if not stop_token_ids:
        add_id(special_tokens.eos_token_id)
    for token_ids in special_tokens.stop_words_id_list:
        if len(token_ids) == 1:
            add_id(token_ids[0])

    return build_grammar_tokenizer_info_json(
        real_tokenizer,
        model_vocab_size=int(model_config.vocab_size or 0),
        stop_token_ids=sorted(stop_token_ids),
    )
