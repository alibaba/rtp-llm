import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast

from rtp_llm.utils.fuser import fetch_remote_file_to_local


def _load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerBase:
    local_path = fetch_remote_file_to_local(os.path.expanduser(tokenizer_path.strip()))
    try:
        return AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
    except (ValueError, KeyError, OSError):
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        tokenizer_file = os.path.join(local_path, "tokenizer.json")
        if os.path.exists(tokenizer_file):
            tokenizer = Tokenizer.from_file(tokenizer_file)
            return PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                eos_token="<|endoftext|>",
                pad_token="<|endoftext|>",
            )
        raise


def get_prompt(tokenizer: Any, prompt: str, seqlen: int):
    while len(tokenizer.encode(prompt)) < seqlen:
        prompt += prompt
    for dec_step in [1024, 256, 64, 16, 2, 1]:
        while len(tokenizer.encode(prompt[:-dec_step])) >= seqlen:
            prompt = prompt[:-dec_step]
    return prompt


def _create_query_worker(args: Tuple[str, int]) -> Tuple[int, str]:
    """Top-level worker for ProcessPoolExecutor (must be picklable)."""
    tokenizer_path, input_len = args
    tokenizer = _load_tokenizer(tokenizer_path)
    base_query = "hello " * (input_len + 20)
    left, right = 0, len(base_query)
    while left < right:
        mid = (left + right) // 2
        current_query = base_query[:mid]
        current_len = len(tokenizer.encode(current_query))
        if current_len == input_len:
            return (input_len, current_query)
        elif current_len < input_len:
            left = mid + 1
        else:
            right = mid
    return (input_len, base_query[:left])


def create_query(
    tokenizer_path: str = "",
    input_len_list: Optional[List[int]] = None,
    max_workers: int = 8,
) -> Dict[int, str]:
    if input_len_list is None:
        input_len_list = []
    tokenizer_path = tokenizer_path or os.environ.get(
        "TOKENIZER_PATH", os.environ.get("CHECKPOINT_PATH", "")
    )
    tokenizer_path = fetch_remote_file_to_local(
        os.path.expanduser(tokenizer_path.strip())
    )

    effective_workers = min(max_workers, len(input_len_list))
    logging.info(
        f"Creating queries for {len(input_len_list)} input lengths "
        f"with {effective_workers} workers"
    )

    if effective_workers <= 1:
        return dict(_create_query_worker((tokenizer_path, x)) for x in input_len_list)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    worker_args = [(tokenizer_path, x) for x in input_len_list]
    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        results = list(executor.map(_create_query_worker, worker_args))
    return dict(results)


def _build_query_from_word(tokenizer: Any, word: str, target_len: int) -> str:
    base = word * (target_len + 20)
    left, right = 0, len(base)
    while left < right:
        mid = (left + right) // 2
        if len(tokenizer.encode(base[:mid])) < target_len:
            left = mid + 1
        else:
            right = mid
    return base[:left]


def _build_reuse_pair(
    tokenizer: Any,
    prefix_word: str,
    suffix_word: str,
    input_len: int,
    reuse_len: int,
) -> Tuple[str, str]:
    warmup = _build_query_from_word(tokenizer, prefix_word, input_len)
    prefix = _build_query_from_word(tokenizer, prefix_word, reuse_len)
    new_len = input_len - len(tokenizer.encode(prefix))
    suffix_base = suffix_word * (new_len + 20)
    left, right = 0, len(suffix_base)
    while left < right:
        mid = (left + right) // 2
        if len(tokenizer.encode(prefix + suffix_base[:mid])) < input_len:
            left = mid + 1
        else:
            right = mid
    test = prefix + suffix_base[:left]
    return (warmup, test)


REUSE_PHASE_WORDS = [
    ("hello ", "world "),
    ("apple ", "grape "),
    ("brick ", "stone "),
]


def _create_reuse_query_worker(
    args: Tuple[str, int, float],
) -> Tuple[int, List[Tuple[str, str]]]:
    tokenizer_path, input_len, reuse_ratio = args
    tokenizer = _load_tokenizer(tokenizer_path)
    reuse_len = int(input_len * reuse_ratio)

    pairs = []
    for prefix_word, suffix_word in REUSE_PHASE_WORDS:
        warmup, test = _build_reuse_pair(
            tokenizer,
            prefix_word,
            suffix_word,
            input_len,
            reuse_len,
        )
        pairs.append((warmup, test))

    actual_prefix_len = len(
        tokenizer.encode(_build_query_from_word(tokenizer, "hello ", reuse_len))
    )
    logging.info(
        f"Reuse query: input_len={input_len}, "
        f"reuse_len={reuse_len}, "
        f"actual_prefix_len={actual_prefix_len}, "
        f"ratio={actual_prefix_len/input_len:.1%}, "
        f"phases={len(pairs)}"
    )

    return (input_len, pairs)


def create_reuse_query(
    reuse_ratio: float,
    tokenizer_path: str = "",
    input_len_list: Optional[List[int]] = None,
    max_workers: int = 8,
) -> Dict[int, List[Tuple[str, str]]]:
    """Create reuse query pairs for each phase (jit_warmup, measure, profile).

    Each phase uses different words so cache entries don't interfere.
    Returns dict mapping input_len -> [(warmup, test), (warmup, test), (warmup, test)].
    """
    if input_len_list is None:
        input_len_list = []
    tokenizer_path = tokenizer_path or os.environ.get(
        "TOKENIZER_PATH", os.environ.get("CHECKPOINT_PATH", "")
    )
    tokenizer_path = fetch_remote_file_to_local(
        os.path.expanduser(tokenizer_path.strip())
    )

    effective_workers = min(max_workers, len(input_len_list))
    logging.info(
        f"Creating reuse queries (ratio={reuse_ratio}) for "
        f"{len(input_len_list)} input lengths with {effective_workers} workers"
    )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    worker_args = [(tokenizer_path, x, reuse_ratio) for x in input_len_list]
    if effective_workers <= 1:
        return dict(_create_reuse_query_worker(a) for a in worker_args)

    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        results = list(executor.map(_create_reuse_query_worker, worker_args))
    return dict(results)
