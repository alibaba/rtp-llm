import torch
import numpy as np
from typing import Any, List, Union, Dict, Tuple
from maga_transformer.utils.time_util import current_time_ms
from maga_transformer.metrics import kmonitor, GaugeMetrics

from transformers import PreTrainedTokenizerBase
from maga_transformer.config.exceptions import FtRuntimeException, ExceptionType
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.async_decoder_engine.embedding.interface import EngineInputs
import threading

class CommonInputGenerator(object):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, config: GptInitModelParameters):
        self.tokenizer_ = tokenizer
        self.config_ = config
        self.lock = threading.Lock()

    @torch.inference_mode()
    def generate( # type: ignore
        self,
        prompt: Union[List[str], str, List[Tuple[str, str]]],
        truncate: bool = True,
        tokenizer_config: Dict[str, Any] = {}
    ) -> EngineInputs:        
        with self.lock:
            if isinstance(prompt, str):
                prompt = [prompt]
            begin_time = current_time_ms()
            # align images and prompts
            # do batch encode and split into embedding input per batch
            assert self.tokenizer_ is not None, "tokenizer should not be None"
            # truncate with tokenizer max_seq_len
            truncate_length = self.config_.max_seq_len
            if self.config_.position_ids_style == 1:
                truncate_length = self.config_.max_seq_len - (self.config_.special_tokens.pad_token_id + 1)
            encoded = self.tokenizer_(prompt, max_length=truncate_length, return_attention_mask=False, padding=False, return_length=True, truncation=truncate, return_tensors='np', **tokenizer_config)

            combo_tokens = torch.from_numpy(np.concatenate(encoded['input_ids'])).to(torch.int32)
            if 'token_type_ids' in encoded:
                combo_token_types = torch.from_numpy(np.concatenate(encoded['token_type_ids'])).to(torch.int32)
            else:
                combo_token_types = torch.zeros_like(combo_tokens, dtype=torch.int32)
            input_lengths = torch.from_numpy(encoded['length']).to(torch.int32)

            for length in encoded['length']:
                if length > self.config_.max_seq_len:
                    raise FtRuntimeException(ExceptionType.LONG_PROMPT_ERROR, f"one of prompt length: {length} > max_length: {self.config_.max_seq_len}")

            kmonitor.report(GaugeMetrics.PRE_PIPELINE_RT_METRIC, current_time_ms() - begin_time)
            kmonitor.report(GaugeMetrics.INPUT_TOKEN_SIZE_METRIC, len(combo_tokens))
            return EngineInputs(token_ids=combo_tokens, token_type_ids=combo_token_types, input_lengths=input_lengths)