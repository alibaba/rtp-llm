import os
import torch
import asyncio
import torch

from concurrent.futures import ThreadPoolExecutor, Future
from typing import List

from maga_transformer.utils.word_util import remove_padding_eos, get_stop_word_slices, \
            truncate_response_with_stop_words, truncate_token_with_stop_word_id, match_stop_words

class TokenProcessor:
    def __init__(self, tokenizer, special_tokens):
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
    
    def decode(self, token_id: List[int]) -> str:
        return self.tokenizer.decode(token_id)
    
    def encode(self, prompt: str) -> torch.tensor:
        th = self.tokenizer.encode(prompt)
        return th

    def process_stop_str(self, 
                         text: str, 
                         stop_word_str_list: List[str]):
        trunc_text = truncate_response_with_stop_words(text, stop_word_str_list)
        token_buffer = text[len(trunc_text):]
        return trunc_text, token_buffer
        
    def get_stop_word_str_slices(self, stop_word_str_list: List[str]):
        return get_stop_word_slices(stop_word_str_list)
    
    def remove_padding_eos(self, token_id: List[int]):
        return remove_padding_eos(token_id, self)
