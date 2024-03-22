
import json
from typing import List, Dict, Union, Any
from transformers import PreTrainedTokenizerBase

def _format_tokens(content_tokens: List[int], role_special_tokens: Any) -> List[int]:
    content_tokens = role_special_tokens.token_ids + content_tokens + role_special_tokens.eos_token_ids
    return content_tokens

 # from modeling_baichuan.py
def encode_chatapi(messages: List[Dict[str, str]], special_tokens: Any, tokenizer: PreTrainedTokenizerBase) -> List[int]:
    max_input_tokens = 2 ** 32 # int max, maybe support max_history_len in generate_config
    total_input: List[int] = []
    round_input: List[int] = []
    system_input = ''
    for i, message in enumerate(messages[::-1]):
        content_tokens = tokenizer.encode(message['content'])
        if message['role'] == 'user':
            round_input = _format_tokens(content_tokens, special_tokens.user) + round_input
            if total_input and len(total_input) + len(round_input) > max_input_tokens:
                break
            else:
                total_input = round_input + total_input
                if len(total_input) >= max_input_tokens:
                    break
                else:
                    round_input = []
        elif message['role'] == 'assistant':
            round_input = _format_tokens(content_tokens, special_tokens.assistant) + round_input
        elif message['role'] == 'system':
            if i != len(messages) - 1:
                raise Exception('system role must be 1st message')
            system_input = message['content']
        else:
            raise ValueError(f"message role not supported yet: {message['role']}")
    if system_input:
        total_input = _format_tokens(tokenizer.encode(system_input), special_tokens.system) + total_input
    if special_tokens.bos_token_id != -1:
        total_input = [special_tokens.bos_token_id] + total_input
    total_input = total_input[-max_input_tokens:]  # truncate left
    total_input += special_tokens.assistant.token_ids
    return total_input
