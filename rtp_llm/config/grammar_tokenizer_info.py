import json
from typing import Any, Dict, Iterator, List, Sequence, Tuple


def _build_encoded_vocab(vocab: Dict[str, int], model_vocab_size: int) -> Tuple[List[str], int]:
    if not vocab:
        raise ValueError("tokenizer vocab is empty")
    if model_vocab_size < 0:
        raise ValueError(f"negative model_vocab_size {model_vocab_size}")

    max_id = -1
    for token_id in vocab.values():
        token_id = int(token_id)
        if token_id < 0:
            raise ValueError(f"negative token id {token_id}")
        max_id = max(max_id, token_id)

    tokenizer_vocab_size = max(len(vocab), max_id + 1)
    vocab_size = max(int(model_vocab_size or 0), tokenizer_vocab_size)
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")

    encoded_vocab = [""] * vocab_size
    for token, token_id in vocab.items():
        encoded_vocab[int(token_id)] = token
    return encoded_vocab, vocab_size


def _get_hf_tokenizer_json(tokenizer: Any) -> str:
    try:
        to_str = tokenizer.backend_tokenizer.to_str
    except AttributeError as error:
        raise ValueError(
            f"tokenizer {type(tokenizer)} does not expose backend_tokenizer.to_str()"
        ) from error
    if not callable(to_str):
        raise ValueError(
            f"tokenizer {type(tokenizer)} does not expose backend_tokenizer.to_str()"
        )
    return to_str()


def _is_fast_tokenizer(tokenizer: Any) -> bool:
    try:
        from transformers import PreTrainedTokenizerFast
    except ImportError:
        return False
    return isinstance(tokenizer, PreTrainedTokenizerFast)


def _is_tiktoken_tokenizer(tokenizer: Any) -> bool:
    return (
        _has_tiktoken_encoding(tokenizer)
        or _has_tiktoken_vocab_file(tokenizer)
        or _has_mergeable_ranks(tokenizer)
    )


def _has_tiktoken_encoding(tokenizer: Any) -> bool:
    try:
        inner_tokenizer = tokenizer.tokenizer
    except AttributeError:
        return False
    inner_type = type(inner_tokenizer)
    return (
        inner_type.__name__ == "Encoding"
        and inner_type.__module__.split(".", 1)[0] == "tiktoken"
    )


def _has_tiktoken_vocab_file(tokenizer: Any) -> bool:
    try:
        vocab_file = tokenizer.vocab_files_names["vocab_file"]
        return "tiktoken" in vocab_file
    except (AttributeError, KeyError, TypeError):
        return False


def _has_mergeable_ranks(tokenizer: Any) -> bool:
    try:
        tokenizer.mergeable_ranks
    except AttributeError:
        return False
    return True


def _is_byte_level_tokenizer(tokenizer: Any) -> bool:
    new_ids = tokenizer.encode(r" ")
    if len(new_ids) < 1:
        return False
    new_tokens = tokenizer.convert_ids_to_tokens(new_ids)
    return new_tokens[0] == "\u0120"


def _is_sentencepiece_tokenizer(tokenizer: Any) -> bool:
    return any(
        _has_sentencepiece_api(candidate)
        for candidate in _iter_sentencepiece_candidates(tokenizer)
    )


def _iter_sentencepiece_candidates(tokenizer: Any) -> Iterator[Any]:
    try:
        yield tokenizer.sp_model
    except AttributeError:
        pass

    try:
        yield tokenizer.tokenizer.sp_model
    except AttributeError:
        pass

    try:
        yield tokenizer.tok
    except AttributeError:
        pass


def _has_sentencepiece_api(candidate: Any) -> bool:
    if candidate is None:
        return False
    try:
        piece_to_id = candidate.PieceToId
        id_to_piece = candidate.IdToPiece
        vocab_size = candidate.vocab_size
    except AttributeError:
        return False
    return callable(piece_to_id) and callable(id_to_piece) and callable(vocab_size)


def build_grammar_tokenizer_info_json(
    tokenizer: Any,
    *,
    model_vocab_size: int,
    stop_token_ids: Sequence[int],
) -> str:
    from rtp_llm.ops import serialize_grammar_tokenizer_info

    vocab = tokenizer.get_vocab()
    encoded_vocab, vocab_size = _build_encoded_vocab(
        vocab,
        int(model_vocab_size or 0),
    )
    stop_token_ids = [int(token_id) for token_id in stop_token_ids]
    if not stop_token_ids:
        raise ValueError("stop_token_ids cannot be empty")

    if _is_fast_tokenizer(tokenizer):
        metadata = {
            "vocab_size": vocab_size,
            "stop_token_ids": stop_token_ids,
            "hf_tokenizer_json": _get_hf_tokenizer_json(tokenizer),
        }
        return serialize_grammar_tokenizer_info(
            encoded_vocab, json.dumps(metadata, separators=(",", ":"))
        )

    if _is_tiktoken_tokenizer(tokenizer):
        vocab_type = (
            "BYTE_LEVEL" if _is_byte_level_tokenizer(tokenizer) else "RAW"
        )
        metadata = {
            "vocab_size": vocab_size,
            "stop_token_ids": stop_token_ids,
            "vocab_type": vocab_type,
            "add_prefix_space": False,
        }
        return serialize_grammar_tokenizer_info(
            encoded_vocab, json.dumps(metadata, separators=(",", ":"))
        )

    if _is_sentencepiece_tokenizer(tokenizer):
        vocab_type = "BYTE_FALLBACK" if "<0x0A>" in vocab else "RAW"
        metadata = {
            "vocab_size": vocab_size,
            "stop_token_ids": stop_token_ids,
            "vocab_type": vocab_type,
            "add_prefix_space": True,
        }
        return serialize_grammar_tokenizer_info(
            encoded_vocab, json.dumps(metadata, separators=(",", ":"))
        )

    raise ValueError(f"Unsupported tokenizer type: {type(tokenizer)}")
