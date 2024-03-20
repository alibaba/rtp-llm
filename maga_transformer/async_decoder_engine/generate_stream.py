import copy
import torch
import logging
import numpy as np
from enum import Enum
from typing import Any, List, Optional
from threading import Lock
from pydantic import BaseModel, Field, PrivateAttr

from maga_transformer.utils.time_util import current_time_ms
from maga_transformer.utils.stop_utils import create_stop_criteria_list
from maga_transformer.metrics import kmonitor, GaugeMetrics
from maga_transformer.models.base_model import GenerateInput, GenerateOutput
from maga_transformer.async_decoder_engine.ptuning.ptuning import PrefixInfo
from maga_transformer.utils.util import AtomicCounter

class Status(Enum):
    WAITING = 0
    RUNNING = 1
    STOPPED = 2
    FINISHED = 3

class GenerateStatus(BaseModel):
    status: Status = Status.WAITING
    error_info: str = ''

stream_counter = AtomicCounter()

class GenerateStream(BaseModel):
    _input: GenerateInput
    _status: GenerateStatus = PrivateAttr(default_factory=GenerateStatus)
    _output: GenerateOutput = PrivateAttr(default_factory=GenerateOutput)
    _complete_token_ids: torch.Tensor
    _max_seq_len: int = PrivateAttr(default_factory=int)
    _block_indice : List[List[int]] = PrivateAttr(default_factory=list)
    _reuse_length : int = PrivateAttr(default_factory=int)
    _resource_dtors: List[Any] = []
    medusa_state: Any = None
    _stream_id: int = PrivateAttr(default_factory=int)
    _released: bool = PrivateAttr(default_factory=bool)

    def __init__(self, input: GenerateInput, max_seq_len: int=2048):
        super().__init__()
        self._input = input
        self._max_seq_len = max_seq_len
        self._seq_length = self.input_length
        self._begin_time = current_time_ms()
        self._resource_dtors = []

        self._output.output_ids = torch.zeros([self._input.generate_config.num_beams, 0], dtype=torch.int32)
        self._complete_token_ids = torch.zeros([self._input.generate_config.num_beams, max_seq_len], dtype=torch.int32)
        self._complete_token_ids[0, :self._seq_length] = self._input.token_ids
        self._cum_log_probs = torch.zeros([self._input.generate_config.num_beams], dtype=torch.float32)
        self._stop_criteria_list = create_stop_criteria_list(
            self._input.generate_config.stop_words_list,
            self._input.generate_config.stop_words_str,
            self._input.tokenizer)
        self._ptuning_info = PrefixInfo()
        self._lock = Lock()

        global stream_counter
        self._stream_id = stream_counter.increment()
        self._released = False
        # for mulit_task_prompt
        self._require_release = True

    def add_resource_dtor(self, dtor: Any):
        self._resource_dtors.append(dtor)
        
    def set_require_release(self, require_release: bool):
        self._require_release = require_release
        
    def clear_resource_dtor(self):
        self._resource_dtors.clear()

    def release_resource(self):        
        if self._released or not self._require_release:
            return
        for dtor in self._resource_dtors:
            dtor()
        self._resource_dtors.clear()
        self._released = True

    def update_prefix(self, ptuning_info: PrefixInfo):
        self._ptuning_info = ptuning_info
        if ptuning_info.prefix_tensors is not None:
            self._input.update_prefix(ptuning_info.prefix_tensors)
            self._seq_length = self.input_length
            self._complete_token_ids[0, :self._seq_length] = self._input.token_ids

    @property
    def ptuning_info(self):
        return self._ptuning_info

    @property
    def cum_log_probs(self):
        return self._cum_log_probs

    @property
    def complete_token_ids(self):
        return self._complete_token_ids[:,:self.seq_length]

    @property
    def input_token_ids(self):
        return self._complete_token_ids[:, :self.input_length]

    @property
    def reuse_length(self):
        return self._reuse_length

    @property
    def input_length(self):
        return self._input.input_length

    @property
    def prefix_length(self):
        return self._input.prefix_length

    @property
    def seq_length(self):
        return self._seq_length

    @property
    def generate_config(self):
        return self._input.generate_config

    @property
    def images(self):
        return self._input.images

    @property
    def lora_id(self):
        return self._input.lora_id

    @property
    def output(self):
        with self._lock:
            output = copy.copy(self._output)
            output.aux_info = copy.copy(output.aux_info)
            return output

    def set_stop(self, err: str):
        with self._lock:
            self._status.status = Status.STOPPED
            self._status.error_info = err

    def stop_and_release(self, err: str):
        self.set_stop(err)
        self.release_resource()

    def set_running(self):
        with self._lock:
            if self._stopped:
                return False
            self._report_wait_time()
            self._status.status = Status.RUNNING
            return True

    def _set_finished(self):
        self._status.status = Status.FINISHED

    @property
    def _stopped(self):
        return self._status.status == Status.STOPPED

    @property
    def stopped(self):
        with self._lock:
            return self._status.status == Status.STOPPED

    @property
    def stop_reason(self):
        with self._lock:
            return self._status.error_info

    @property
    def finished(self):
        with self._lock:
            return self._status.status == Status.FINISHED

    def check_timeout(self):
        running_time = current_time_ms() - self._begin_time
        if self.generate_config.timeout_ms > 0 and self.generate_config.timeout_ms < running_time:
            self.stop_and_release(f"query has been running {running_time} ms, it's timeout")

    def _report_wait_time(self):
        kmonitor.report(GaugeMetrics.ASYNC_WAIT_WAIT_TIME_METRIC, current_time_ms() - self._begin_time)

    def _report_first_token_rt(self):
        kmonitor.report(GaugeMetrics.FT_FIRST_TOKEN_RT_METRIC, current_time_ms() - self._begin_time)

    def set_kvcache(self, block_indice: List[List[int]], reuse_length: int):
        self._block_indice = block_indice
        self._reuse_length = reuse_length

    def update(self,
               new_tokens: torch.Tensor,
               num_new_tokens: int,
               finished: bool,
               hidden_states: Optional[torch.Tensor],
               logits: Optional[torch.Tensor],
               cum_log_probs: Optional[torch.Tensor]):
        with self._lock:
            if self._output.aux_info.iter_count == 0:
                self._report_first_token_rt()

            # NOTE: new tokens indicate num of newly genearted tokens
            # typically 1 but can be > 1 under speculative decoding
            # This differs from new_tokens.shape[-1] under beam search case,
            # which needs to update all the generated tokens each update.
            update_length = new_tokens.shape[-1]
            update_to_pos = self._seq_length + num_new_tokens
            update_from_pos = update_to_pos - update_length
            self._complete_token_ids[:, update_from_pos:update_to_pos] = new_tokens
            for i in range(num_new_tokens):
                self._seq_length += 1
                if self._need_finish():
                    finished = True
                    break
            if finished:
                self._set_finished()
            self._output.output_ids = self._complete_token_ids[:,self.input_length:self._seq_length]
            self._output.input_ids = self.input_token_ids
            self._output.hidden_states = hidden_states
            if len(self._input.generate_config.select_tokens_id) > 0:
                self._output.logits = logits[:,self._input.generate_config.select_tokens_id]
            else:
                self._output.logits = logits

            self._output.finished = finished
            self._output.aux_info.cost_time = current_time_ms() - self._begin_time
            self._output.aux_info.input_len = self._input.prompt_length
            self._output.aux_info.prefix_len = self._input.prefix_length
            self._output.aux_info.output_len = self._output.output_ids.shape[-1]
            self._output.aux_info.cum_log_probs = cum_log_probs.tolist() if cum_log_probs is not None else None
            self._output.aux_info.iter_count += 1
            self._output.aux_info.reuse_len = self._reuse_length

    def set_loss(self, loss):
        self._output.loss = loss

    def add_block_index(self, block_index: List[List[int]]):
        assert len(block_index) == len(self._block_indice)
        for i in range(len(block_index)):
            self._block_indice[i].extend(block_index[i])

    def pop_block_indice(self):
        block_indice = self._block_indice
        self._block_indice = [[]]
        return block_indice

    @property
    def block_indice(self):
        return self._block_indice

    def _need_finish(self):
        return self._seq_length >= min(self._max_seq_len, self.generate_config.max_new_tokens + self._input.input_length) \
            or (self._seq_length >= self.generate_config.min_new_tokens + self._input.input_length and self._invoke_stop_words_criterion())

    def _invoke_stop_words_criterion(self):
        if self._seq_length == self.input_length:
            return False
        tokens_to_check = self._complete_token_ids[0,self.input_length:self._seq_length].tolist()
        for stop_criteria in self._stop_criteria_list:
            if stop_criteria(tokens_to_check):
                return True
        return False

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)
