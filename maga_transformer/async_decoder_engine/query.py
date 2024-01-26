import copy
import torch
import logging
import numpy as np
from typing import Any, List, Optional
from threading import Lock
from maga_transformer.utils.time_util import current_time_ms
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.config.generate_config import GenerateConfig
from transformers.generation.stopping_criteria import StoppingCriteria
from maga_transformer.utils.stop_utils import create_stop_criteria_list
from maga_transformer.async_decoder_engine.ptuning import PrefixType
from maga_transformer.utils.util import to_cuda, to_cpu
from maga_transformer.metrics import kmonitor, GaugeMetrics
from maga_transformer.utils.model_weight import LoraResource
from maga_transformer.tokenizer.tokenizer_base import TokenizerBase

'''
input_tokens:    prompt tokens
max_seq_len:     max total len of tokens
reuse_length:    kvcache reuse length(like prompt tuning)
prefix_length:   prefix length for ptuning
generate_config: config for sampling
block_indice:    kvcache block indice for paged attention
slice_length:    length to slice when return result to server
images:          multimodal param for image
'''

class QueryStats:
    def __init__(
            self,
            input_tokens: torch.Tensor,
            tokenizer: Optional[TokenizerBase],
            max_seq_len: int,
            reuse_length: int,
            generate_config: GenerateConfig,
            block_indice: List[int],
            slice_length : int,
            images: List[str] = [],
            adapter_name: str = "",
            lora_resource: Optional[LoraResource] = None
    ) -> None:
        self.max_seq_len = max_seq_len
        self.reuse_length = reuse_length
        self.images = images
        self.context_length = input_tokens.shape[-1]
        self.generate_config = generate_config
        self.max_new_tokens = self.generate_config.max_new_tokens
        self.min_new_tokens = self.generate_config.min_new_tokens
        self.adapter_name = adapter_name
        self.error_info_ = ''
        self.stop = False
        self.running_ = False
        self.finish = False
        self.slice_length = slice_length
        self.iter_count = 0

        self.lock = Lock()
        self.seq_length: int = self.context_length
        self.beam_width: int = generate_config.num_beams
        self.input_token_ids = input_tokens
        self.output_token_ids_: torch.Tensor = torch.zeros([self.beam_width, max_seq_len], dtype=torch.int32)
        self.output_token_ids_[0, : self.seq_length] = input_tokens
        self.hidden_states = None
        self.logits = None
        self.loss = None
        self.cum_log_probs = torch.zeros([self.beam_width], dtype=torch.float32)
        self.medusa_state = None
        self.begin_time = current_time_ms()
        self.has_reported_metric = False
        self.stop_criteria_list: List[StoppingCriteria] = \
            create_stop_criteria_list(generate_config.stop_words_list,
                                      generate_config.stop_words_str,
                                      tokenizer)


        self.block_indice: List[List[int]] = [block_indice]
        self.lora_resource = lora_resource

    def acquire(self):
        if self.lora_resource != None:
            self.lora_resource.read_acquire(self.adapter_name)

    def release(self):
        if self.lora_resource != None:
            self.lora_resource.read_release(self.adapter_name)

    @property
    def chat_id(self):
        return self.generate_config.chat_id

    def has_error(self):
        return True if self.error_info_ else False

    def has_timeout(self) -> bool:
        if self.generate_config.timeout_ms > 0 and self.generate_config.timeout_ms < current_time_ms() - self.begin_time:
            self.set_error(f"query has been running {current_time_ms() - self.begin_time} ms timeout")
            return True
        else:
            return False

    @property
    def error_info(self):
        return self.error_info_

    def set_error(self, err: str):
        self.finish = True
        self.error_info_ = err

    def set_stop(self):
        self.stop = True

    @property
    def running(self):
        return self.running_

    def set_running(self):
        self.running_ = True

    def report_wait_time(self):
        kmonitor.report(GaugeMetrics.ASYNC_WAIT_WAIT_TIME_METRIC, current_time_ms() - self.begin_time)

    def report_first_token_rt(self):
        if not self.has_reported_metric:
            kmonitor.report(GaugeMetrics.FT_FIRST_TOKEN_RT_METRIC, current_time_ms() - self.begin_time)
            self.has_reported_metric = True

    @property
    def output_token_ids(self):
        with self.lock:
            return self.output_token_ids_[:, :self.seq_length]

    def set_reuse_length(self, reuse_length: int):
        self.reuse_length = reuse_length

    @property
    def sliced_output_token_ids(self):
        with self.lock:
            return self.output_token_ids_[:, self.slice_length: self.seq_length]

    def increase_iter(self):
        self.iter_count += 1

    def update(self, hidden_states: torch.Tensor, logits: torch.Tensor,
               token: torch.Tensor, cum_log_probs: torch.Tensor
    ):
        with self.lock:
            self.hidden_states = hidden_states
            self.logits = logits
            self.output_token_ids_[:, self.seq_length] = token
            self.cum_log_probs = cum_log_probs
            self.seq_length += 1

    def add_block_index(self, block_index: List[List[int]]):
        with self.lock:
            assert len(block_index) == len(self.block_indice)
            for i in range(len(block_index)):
                self.block_indice[i].extend(block_index[i])

    def pop_block_indice(self):
        with self.lock:
            block_indice = self.block_indice
            self.block_indice = [[]]
            return block_indice

    def _invoke_stop_words_criterion(self):
        for stop_criteria in self.stop_criteria_list:
            tokens_to_check = self.output_token_ids[0]
            if stop_criteria(tokens_to_check.tolist(), self.context_length):
                return True
        return False

    def need_finish(self):
        return self.finish or self._out_of_max(self.max_seq_len) or self.has_timeout() or \
            self.has_error() or self.stop or (self.seq_length >= self.min_new_tokens + self.context_length and self._invoke_stop_words_criterion())

    def _out_of_max(self, max_seq_len: int) -> bool:
        return self.seq_length >= min(max_seq_len, self.max_new_tokens + self.context_length)

    def __repr__(self) -> str:
        return self.__str__()
