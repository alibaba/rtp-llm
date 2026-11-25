import importlib.util
import os
from typing import Any, List, Sequence

from rtp_llm.config.stop_words import checkpoint_eos_ids, primary_eos_id


def _make_xgrammar_bindings_findable() -> None:
    # xgrammar loads libxgrammar_bindings.so through tvm_ffi, which locates the
    # file via the wheel RECORD and then LD_LIBRARY_PATH. Bazel pip runfiles
    # omit RECORD, so prepend the package directory, where the .so lives.
    spec = importlib.util.find_spec("xgrammar")
    if spec is None or not spec.submodule_search_locations:
        return
    lib_dir = list(spec.submodule_search_locations)[0]
    if not os.path.exists(os.path.join(lib_dir, "libxgrammar_bindings.so")):
        return
    paths = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in paths.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = (
            lib_dir + os.pathsep + paths if paths else lib_dir
        )


def build_grammar_tokenizer_info_json(
    tokenizer: Any,
    *,
    model_vocab_size: int,
    stop_token_ids: Sequence[int],
) -> str:
    """Serialize tokenizer metadata through xgrammar's public Python API."""
    _make_xgrammar_bindings_findable()
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


def grammar_stop_token_ids(
    tokenizer: Any,
    model_config: Any,
    stop_words: Sequence[Sequence[int]],
    override_stop_tokens: Sequence[int],
) -> List[int]:
    """Use the same terminal tokens for vocabulary compilation and matching."""
    if override_stop_tokens:
        return sorted(set(override_stop_tokens))

    hf = tokenizer.get_real_tokenizer()
    special = set(getattr(hf, "all_special_ids", ()))
    # Fast tokenizers can have special added tokens outside all_special_ids.
    added_tokens = getattr(hf, "added_tokens_decoder", {})
    special.update(i for i, token in added_tokens.items() if token.special)
    special.update(checkpoint_eos_ids(tokenizer))
    ids = {seq[0] for seq in stop_words if len(seq) == 1 and seq[0] in special}
    eos = primary_eos_id(tokenizer, model_config)
    if eos is not None and eos >= 0:
        ids.add(eos)
    return sorted(ids)


def build_model_grammar_tokenizer_info_json(
    tokenizer: Any,
    model_config: Any,
    stop_words: Sequence[Sequence[int]],
    override_stop_tokens: Sequence[int],
) -> str:
    """Build admission metadata from the tokenizer already loaded by Dash SC."""
    real_tokenizer = tokenizer.get_real_tokenizer()
    if real_tokenizer is None:
        return ""

    return build_grammar_tokenizer_info_json(
        real_tokenizer,
        model_vocab_size=int(model_config.vocab_size or 0),
        stop_token_ids=grammar_stop_token_ids(
            tokenizer, model_config, stop_words, override_stop_tokens
        ),
    )
