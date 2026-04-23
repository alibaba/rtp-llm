import os
from typing import Any, Dict, List

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from rtp_llm.utils.fuser import fetch_remote_file_to_local


def _load_tokenizer(model_type: str, tokenizer_path: str) -> PreTrainedTokenizerBase:
    """Load tokenizer, with GLM-5 compatible path (bypass invalid tokenizer_config.json)."""
    if model_type == "glm_5":
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        tokenizer_file = os.path.join(tokenizer_path, "tokenizer.json")
        if not os.path.exists(tokenizer_file):
            raise FileNotFoundError(
                f"GLM-5 tokenizer requires tokenizer.json at {tokenizer_file}"
            )
        tokenizer = Tokenizer.from_file(tokenizer_file)
        return PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            eos_token="<|endoftext|>",
            pad_token="<|endoftext|>",
            unk_token="<|endoftext|>",
        )
    if os.path.isdir(tokenizer_path):
        return AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True, local_files_only=True
        )
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def get_prompt(tokenizer: Any, prompt: str, seqlen: int):
    while len(tokenizer.encode(prompt)) < seqlen:
        prompt += prompt
    for dec_step in [1024, 256, 64, 16, 2, 1]:
        while len(tokenizer.encode(prompt[:-dec_step])) >= seqlen:
            prompt = prompt[:-dec_step]
    return prompt


def create_query(
    model_type: str, tokenizer_path: str, input_len_list: List[int]
) -> Dict[int, str]:
    tokenizer_path = fetch_remote_file_to_local(tokenizer_path)

    def _create_query_single(tokenizer: PreTrainedTokenizerBase, input_len: int) -> str:
        base_query = "hello " * (input_len + 20)

        def get_token_length(text: str) -> int:
            return len(tokenizer.encode(text))

        left, right = 0, len(base_query)
        while left < right:
            mid = (left + right) // 2
            current_query = base_query[:mid]
            current_len = get_token_length(current_query)
            if current_len == input_len:
                return current_query
            elif current_len < input_len:
                left = mid + 1
            else:
                right = mid
        return base_query[:left]

    tokenizer = _load_tokenizer(model_type, tokenizer_path)
    return {x: _create_query_single(tokenizer, x) for x in input_len_list}
