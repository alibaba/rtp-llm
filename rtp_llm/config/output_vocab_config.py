import json
import os
from collections.abc import Iterable, Mapping
from typing import Any, Optional

OUTPUT_TOKENS_FILENAME = "output_tokens.json"


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer, got {value!r}")
    return value


def _flatten_output_tokens(raw_tokens: Any) -> tuple[list[Any], str]:
    if not isinstance(raw_tokens, list) or not raw_tokens:
        raise ValueError("output_tokens.json must be a non-empty JSON array")

    nested_items = [isinstance(item, list) for item in raw_tokens]
    if any(nested_items) and not all(nested_items):
        raise ValueError(
            "output_tokens.json cannot mix token values and organizational groups"
        )

    if all(nested_items):
        values = []
        for group_index, group in enumerate(raw_tokens):
            if not group:
                raise ValueError(
                    f"output_tokens.json group {group_index} must not be empty"
                )
            if any(isinstance(item, list) for item in group):
                raise ValueError(
                    "output_tokens.json supports only one organizational group level"
                )
            values.extend(group)
    else:
        values = list(raw_tokens)

    if all(isinstance(value, str) for value in values):
        return values, "text"
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return values, "id"
    raise ValueError(
        "output_tokens.json must contain only token strings or only canonical "
        "integer token IDs"
    )


def _resolve_text_token_ids(
    token_texts: list[str], tokenizer_vocab: Optional[Mapping[str, int]]
) -> list[int]:
    if not isinstance(tokenizer_vocab, Mapping):
        raise ValueError(
            "token-text output_tokens.json requires tokenizer.get_vocab() to "
            "return an exact token-to-ID mapping"
        )

    token_ids = []
    unknown_tokens = []
    for token in token_texts:
        if token not in tokenizer_vocab:
            unknown_tokens.append(token)
            continue
        token_ids.append(
            _require_int(
                tokenizer_vocab[token], f"tokenizer vocabulary ID for {token!r}"
            )
        )

    if unknown_tokens:
        examples = ", ".join(repr(token) for token in unknown_tokens[:10])
        suffix = "" if len(unknown_tokens) <= 10 else ", ..."
        raise ValueError(
            f"{len(unknown_tokens)} token entries are absent from the exact tokenizer "
            f"vocabulary: {examples}{suffix}"
        )
    return token_ids


def _normalize_output_vocab_ids(
    token_ids: Iterable[int],
    extra_token_ids: Iterable[int],
    model_vocab_size: int,
    input_vocab_size: Optional[int],
) -> list[int]:
    if model_vocab_size <= 0:
        raise ValueError(f"model vocab_size must be positive, got {model_vocab_size}")

    effective_input_vocab_size = (
        input_vocab_size
        if input_vocab_size is not None and input_vocab_size > 0
        else model_vocab_size
    )
    normalized_ids = []
    for index, value in enumerate(token_ids):
        normalized_ids.append(_require_int(value, f"output token ID at index {index}"))
    for index, value in enumerate(extra_token_ids):
        normalized_ids.append(_require_int(value, f"extra token ID at index {index}"))

    output_vocab_ids = sorted(set(normalized_ids))
    for token_id in output_vocab_ids:
        if token_id < 0 or token_id >= model_vocab_size:
            raise ValueError(
                f"output token ID {token_id} is outside [0, {model_vocab_size})"
            )
        if token_id >= effective_input_vocab_size:
            raise ValueError(
                f"output token ID {token_id} is not covered by input embedding size "
                f"{effective_input_vocab_size}"
            )

    if not output_vocab_ids:
        raise ValueError("output vocabulary must not be empty")
    if len(output_vocab_ids) >= model_vocab_size:
        raise ValueError(
            "output vocabulary must be a proper subset of the model vocabulary, "
            f"got K={len(output_vocab_ids)}, V={model_vocab_size}"
        )
    return output_vocab_ids


def parse_output_tokens(
    raw_tokens: Any,
    model_vocab_size: int,
    input_vocab_size: Optional[int] = None,
    tokenizer_vocab: Optional[Mapping[str, int]] = None,
    extra_token_ids: Iterable[int] = (),
) -> list[int]:
    """Resolve one unsegmented token manifest to sorted canonical token IDs."""
    values, source_kind = _flatten_output_tokens(raw_tokens)
    token_ids = (
        _resolve_text_token_ids(values, tokenizer_vocab)
        if source_kind == "text"
        else values
    )
    return _normalize_output_vocab_ids(
        token_ids,
        extra_token_ids,
        model_vocab_size,
        input_vocab_size,
    )


def load_output_vocab_ids(
    checkpoint_path: str,
    model_vocab_size: int,
    input_vocab_size: Optional[int] = None,
    tokenizer: Optional[Any] = None,
    extra_token_ids: Iterable[int] = (),
) -> list[int]:
    config_path = os.path.join(checkpoint_path, OUTPUT_TOKENS_FILENAME)
    try:
        with open(config_path, "r", encoding="utf-8") as reader:
            raw_tokens = json.load(reader)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"failed to read output token manifest {config_path}: {error}"
        ) from error

    values, source_kind = _flatten_output_tokens(raw_tokens)
    tokenizer_vocab = None
    if source_kind == "text":
        get_vocab = getattr(tokenizer, "get_vocab", None)
        if not callable(get_vocab):
            raise ValueError(
                "token-text output_tokens.json requires an exact tokenizer "
                "get_vocab() implementation"
            )
        try:
            tokenizer_vocab = get_vocab()
        except Exception as error:
            raise ValueError(
                "failed to read the exact tokenizer vocabulary required by "
                "output_tokens.json"
            ) from error

    token_ids = (
        _resolve_text_token_ids(values, tokenizer_vocab)
        if source_kind == "text"
        else values
    )
    return _normalize_output_vocab_ids(
        token_ids,
        extra_token_ids,
        model_vocab_size,
        input_vocab_size,
    )
