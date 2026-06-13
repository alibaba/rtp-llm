import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
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


@dataclass
class ReuseCacheQuery:
    seed_query: str
    hit_queries: List[str]
    target_reuse_len: int
    target_hit_rate: float


def _stable_single_token_id(
    tokenizer: Any,
    candidates: List[str],
    forbidden: Optional[set] = None,
) -> int:
    forbidden = forbidden or set()
    for candidate in candidates:
        ids = tokenizer.encode(candidate)
        if len(ids) != 1 or ids[0] in forbidden:
            continue
        token_id = ids[0]
        repeated = [token_id] * 32
        text = tokenizer.decode(
            repeated,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if tokenizer.encode(text) == repeated:
            return token_id
    raise ValueError("No stable single-token candidate found for reuse-cache query")


def _decode_exact_token_ids(tokenizer: Any, token_ids: List[int]) -> str:
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    encoded = tokenizer.encode(text)
    if encoded != token_ids:
        raise ValueError(
            "Decoded reuse-cache query does not round-trip through tokenizer: "
            f"wanted {len(token_ids)} tokens, got {len(encoded)}"
        )
    return text


def target_reuse_len_for_hit_rate(
    input_len: int,
    hit_rate: float,
    seq_size_per_block: int,
) -> int:
    if input_len <= 1:
        raise ValueError(f"input_len must be > 1, got {input_len}")
    if not 0.0 < hit_rate < 1.0:
        raise ValueError(f"hit_rate must be in (0, 1), got {hit_rate}")

    block_tokens = max(1, seq_size_per_block)
    target_blocks = max(1, round((input_len * hit_rate) / block_tokens))
    target_reuse_len = target_blocks * block_tokens
    max_reuse_len = max(1, input_len - 1)
    if block_tokens > 1 and input_len > block_tokens:
        max_reuse_len = input_len - (input_len % block_tokens or block_tokens)
    return min(target_reuse_len, max_reuse_len)


def _create_reuse_cache_query_worker(
    args: Tuple[str, int, float, int, int]
) -> Tuple[int, ReuseCacheQuery]:
    tokenizer_path, input_len, hit_rate, seq_size_per_block, num_variants = args
    tokenizer = _load_tokenizer(tokenizer_path)

    target_reuse_len = target_reuse_len_for_hit_rate(
        input_len, hit_rate, seq_size_per_block
    )
    suffix_len = input_len - target_reuse_len
    if suffix_len <= 0:
        raise ValueError(
            f"reuse-cache target leaves no suffix: input_len={input_len}, "
            f"target_reuse_len={target_reuse_len}"
        )

    candidates = [
        " hello",
        " cache",
        " data",
        " query",
        " token",
        " alpha",
        " beta",
        " gamma",
        " delta",
        " value",
        " prefix",
        " suffix",
        " reuse",
        " tensor",
        " batch",
        " model",
    ]
    prefix_token = _stable_single_token_id(tokenizer, candidates)
    suffix_tokens: List[int] = []
    used = {prefix_token}
    for _ in range(max(1, num_variants)):
        token_id = _stable_single_token_id(tokenizer, candidates, used)
        suffix_tokens.append(token_id)
        used.add(token_id)

    prefix_ids = [prefix_token] * target_reuse_len
    seed_query = _decode_exact_token_ids(tokenizer, prefix_ids)

    hit_queries = [
        _decode_exact_token_ids(
            tokenizer,
            prefix_ids + [suffix_token] * suffix_len,
        )
        for suffix_token in suffix_tokens
    ]
    return (
        input_len,
        ReuseCacheQuery(
            seed_query=seed_query,
            hit_queries=hit_queries,
            target_reuse_len=target_reuse_len,
            target_hit_rate=target_reuse_len / input_len,
        ),
    )


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


def create_reuse_cache_queries(
    tokenizer_path: str = "",
    input_len_list: Optional[List[int]] = None,
    hit_rate: float = 0.0,
    seq_size_per_block: int = 1,
    num_variants: int = 3,
    max_workers: int = 8,
) -> Dict[int, ReuseCacheQuery]:
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
        f"Creating reuse-cache queries for {len(input_len_list)} input lengths "
        f"with hit_rate={hit_rate}, seq_size_per_block={seq_size_per_block}, "
        f"variants={num_variants}, workers={effective_workers}"
    )

    worker_args = [
        (tokenizer_path, x, hit_rate, seq_size_per_block, num_variants)
        for x in input_len_list
    ]
    if effective_workers <= 1:
        return dict(_create_reuse_cache_query_worker(x) for x in worker_args)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        results = list(executor.map(_create_reuse_cache_query_worker, worker_args))
    return dict(results)
