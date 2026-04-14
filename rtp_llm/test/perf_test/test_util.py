import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from rtp_llm.utils.fuser import fetch_remote_file_to_local


def _load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerBase:
    local_path = fetch_remote_file_to_local(os.path.expanduser(tokenizer_path.strip()))
    return AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)


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
