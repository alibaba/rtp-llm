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
from maga_transformer.models.base_model import BaseTokenizer

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
            tokenizer: Optional[BaseTokenizer],
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
        self.adapter_name = adapter_name
        self.error_info_ = ''
        self.stop = False
        self.running_ = False
        self.finish = False
        self.slice_length = slice_length

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
    def max_new_tokens(self) -> int:
        return self.generate_config.max_new_tokens

    @property
    def chat_id(self):
        return self.generate_config.chat_id

    def has_error(self):
        return True if self.error_info_ else False

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
        if self.begin_time:
            kmonitor.report(GaugeMetrics.FT_FIRST_TOKEN_RT_METRIC, current_time_ms() - self.begin_time)
            self.begin_time = 0

    @property
    def output_token_ids(self):
        with self.lock:
            return self.output_token_ids_[:, :self.seq_length]

    @property
    def sliced_output_token_ids(self):
        with self.lock:
            return self.output_token_ids_[:, self.slice_length: self.seq_length]


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
            self.block_indice = []
            return block_indice

    def _invoke_stop_words_criterion(self):
        for stop_criteria in self.stop_criteria_list:
            tokens_to_check = self.output_token_ids[0]
            if stop_criteria(tokens_to_check.tolist(), self.context_length):
                return True
        return False

    def need_finish(self):
        return self.finish or self._out_of_max(self.max_seq_len) or \
            self.has_error() or self.stop or self._invoke_stop_words_criterion()

    def _out_of_max(self, max_seq_len: int) -> bool:
        return self.seq_length >= min(max_seq_len, self.max_new_tokens + self.context_length)

    def __repr__(self) -> str:
        return self.__str__()

class BatchQuery:
    def __init__(self, count_prefix_length:bool, gen_num_per_circle: int, nccl_op: Any) -> None:
        self.gen_num_per_circle = gen_num_per_circle
        self.nccl_op_ = nccl_op
        if g_parallel_info.tp_size > 1:
            assert self.nccl_op_ is not None, "nccl op should not be None when tp_size > 1"
        # input
        self.count_prefix_length = count_prefix_length
        self.queries: List[QueryStats] = []
        self.generate_batch_size: int = 0
        self.context_batch_size: int = 0
        self.beam_width: int = 1
        self.cache_block_indice: torch.Tensor = torch.zeros((1,1), dtype=torch.int32)
        self.output_token_ids: torch.Tensor = torch.zeros((1,1), dtype=torch.int32)
        self.images: List[str] = []
        self.generate_configs: List[GenerateConfig] = []
        self.merge_generate_config: GenerateConfig = GenerateConfig()
        self.seq_lengths_list: List[int] = []
        self.reuse_lengths_list: List[int] = []
        self.context_lengths_list: List[int] = []
        self.record_index_prob: Optional[torch.Tensor] = None
        self.lora_names: List[int] = []
        self.lora_ids: List[int] = []

        # output
        self.update_length: List[int] = []
        self.finished: Optional[torch.Tensor] = None
        self.hidden_states: Optional[torch.Tensor] = None
        self.logits: Optional[torch.Tensor] = None
        self.cum_log_probs: Optional[torch.Tensor] = None
        self.updated_token_ids: Optional[torch.Tensor] = None
        self.output_log_probs: Optional[torch.Tensor] = None
        self.output_index_prob: Optional[torch.Tensor] = None
        self.medusa_state: Optional[List[Any]] = None

    def deepcopy(self) -> 'BatchQuery':
        new_batch_query = BatchQuery(self.count_prefix_length, self.gen_num_per_circle, self.nccl_op_)
        new_batch_query.queries = [q for q in self.queries]
        new_batch_query.generate_batch_size = self.generate_batch_size
        new_batch_query.context_batch_size = self.context_batch_size
        new_batch_query.cache_block_indice = copy.deepcopy(self.cache_block_indice)
        new_batch_query.output_token_ids = copy.deepcopy(self.output_token_ids)
        new_batch_query.images = copy.deepcopy(self.images)
        new_batch_query.generate_configs = self.generate_configs
        new_batch_query.merge_generate_config = self.merge_generate_config
        new_batch_query.seq_lengths_list = copy.deepcopy(self.seq_lengths_list)
        new_batch_query.reuse_lengths_list = copy.deepcopy(self.reuse_lengths_list)
        new_batch_query.context_lengths_list = copy.deepcopy(self.context_lengths_list)
        new_batch_query.lora_names = copy.deepcopy(self.lora_names)
        new_batch_query.lora_ids = copy.deepcopy(self.lora_ids)
        new_batch_query.update_length = self.update_length
        return new_batch_query

    def __str__(self):
        return f'generate_batch_size: {self.generate_batch_size}, \
                context_batch_size: {self.context_batch_size}, \
                output_token_ids: {self.output_token_ids}, \
                seq_length: {self.seq_lengths_list},\
                context_length: {self.context_lengths_list}, \
                lora_names: {self.lora_names}, \
                lora_ids: {self.lora_ids}'

    def tp_sync(self):
        if g_parallel_info.tp_size <= 1:
            return
        check_num: int = 998244353
        check_num2: int = 1000000007
        shape_hints = torch.IntTensor([check_num, self.generate_batch_size, self.context_batch_size, self.output_token_ids.shape[1], self.cache_block_indice.shape[1], check_num2])
        shape_hints = to_cuda(shape_hints)
        self.nccl_op_.broadcast_tp([shape_hints])
        torch.cuda.current_stream().synchronize()
        shape_hints = shape_hints.cpu().numpy()
        assert shape_hints[0] == check_num and shape_hints[5] == check_num2
        if g_parallel_info.tp_rank == 0:
            seq_lengths_tensor = to_cuda(torch.IntTensor(self.seq_lengths_list))
            reuse_lengths_tensor = to_cuda(torch.IntTensor(self.reuse_lengths_list))
            context_lengths_tensor = to_cuda(torch.IntTensor(self.context_lengths_list))
            output_token_ids = to_cuda(self.output_token_ids)
            cache_block_indice = to_cuda(self.cache_block_indice)
            lora_ids_tensor = to_cuda(torch.IntTensor(self.lora_ids))
        else:
            self.generate_batch_size = int(shape_hints[1])
            self.context_batch_size = int(shape_hints[2])
            output_token_ids = torch.zeros((self.total_batch_size, int(shape_hints[3])), dtype=torch.int32, device="cuda:0")
            cache_block_indice = torch.zeros(
                (self.total_batch_size, int(shape_hints[4])), dtype=torch.int32, device="cuda:0")
            seq_lengths_tensor = torch.zeros((self.generate_batch_size), dtype=torch.int32, device="cuda:0")
            reuse_lengths_tensor = torch.zeros((self.total_batch_size), dtype=torch.int32, device="cuda:0")
            context_lengths_tensor = torch.zeros((self.total_batch_size), dtype=torch.int32, device="cuda:0")
            lora_ids_tensor = torch.zeros((max(1, self.total_batch_size)), dtype=torch.int32, device="cuda:0")
        self.nccl_op_.broadcast_tp([cache_block_indice, output_token_ids, seq_lengths_tensor, reuse_lengths_tensor, context_lengths_tensor, lora_ids_tensor])
        if g_parallel_info.tp_rank > 0:
            self.cache_block_indice = to_cpu(cache_block_indice)
            self.output_token_ids = to_cpu(output_token_ids)
            self.seq_lengths_list = to_cpu(seq_lengths_tensor).numpy().tolist()
            self.reuse_lengths_list = to_cpu(reuse_lengths_tensor).numpy().tolist()
            self.context_lengths_list = to_cpu(context_lengths_tensor).numpy().tolist()
            self.lora_ids = to_cpu(lora_ids_tensor).numpy().tolist()

    @property
    def max_context_length(self):
        return max(self.context_lengths_list) if self.context_lengths_list else 0

    @property
    def max_seq_length(self):
        return max(self.seq_lengths_list) if self.seq_lengths_list else 0

    @property
    def decoder_batch_size(self):
        return self.beam_width * self.generate_batch_size + self.context_batch_size

    @property
    def total_batch_size(self):
        return self.generate_batch_size + self.context_batch_size

    def update_all_errors(self, err: str):
        for q in self.queries:
            q.set_error(err)

    def has_context_query(self):
        return self.context_batch_size > 0

    @property
    def context_query_reuse_lengths_list(self) -> List[int]:
        assert len(self.reuse_lengths_list) == self.generate_batch_size * self.beam_width + self.context_batch_size
        return self.reuse_lengths_list[self.generate_batch_size * self.beam_width:]

    @property
    def context_query_context_lengths_list(self) -> List[int]:
        return self.context_lengths_list[self.generate_batch_size * self.beam_width:]

    def context_query_output_tokens(self, index: int) -> torch.Tensor:
        index = index + self.generate_batch_size * self.beam_width
        start_index = 0
        end_index = self.context_lengths_list[index]
        # 除去ptuningv2以外，前缀token不参与计算
        if self.count_prefix_length:
            start_index += self.reuse_lengths_list[index]
            end_index += self.reuse_lengths_list[index]

        return self.output_token_ids[index, start_index: end_index]

    def generate_query_last_token(self, index: int) -> torch.Tensor:
        assert index < self.generate_batch_size
        start_idx = index * self.beam_width
        end_idx = (index + 1) * self.beam_width
        return self.output_token_ids[start_idx: end_idx, self.seq_lengths_list[start_idx] - 1]

    def check(self):
        assert len(self.context_lengths_list) == self.decoder_batch_size
        assert len(self.reuse_lengths_list) == self.decoder_batch_size
        assert len(self.seq_lengths_list) == self.generate_batch_size * self.beam_width
        assert len(self.generate_configs) == self.total_batch_size
        for length in self.seq_lengths_list + self.context_lengths_list:
            if length <= 0:
                raise Exception(
                    f"got length not valid, length_list: {self.seq_lengths_list} {self.context_lengths_list}"
                )

    def add_new_query(self, new_queries: List[QueryStats]):
        if g_parallel_info.tp_rank > 0:
            return
        self.clear()
        [q.set_running() for q in new_queries]
        [q.acquire() for q in new_queries]

        self.generate_batch_size = len(self.queries)
        self.context_batch_size = len(new_queries)
        self.queries += new_queries

    def generate(self):
        if g_parallel_info.tp_rank > 0:
            return
        total_batch_size = len(self.queries)
        if total_batch_size > 0:
            self.beam_width = self.queries[0].beam_width

        cache_block_indice = np.zeros(
            [self.decoder_batch_size, max([len(q.block_indice[0]) for q in self.queries])],
            dtype=np.int32
        )
        output_token_ids = np.zeros(
            [self.decoder_batch_size, max([q.seq_length for q in self.queries]) + self.gen_num_per_circle],
            dtype=np.int32
        )
        images = []
        lora_names = []
        lora_ids = []
        for i, query in enumerate(self.queries):
            images.extend(query.images)
            query.images = []
            if i < self.generate_batch_size:
                start_batch_idx = i * self.beam_width
                end_batch_idx = start_batch_idx + self.beam_width
                cache_block_indice[start_batch_idx:end_batch_idx, :len(query.block_indice[0])] = query.block_indice
                output_token_ids[start_batch_idx:end_batch_idx, :query.seq_length] = query.output_token_ids
                self.seq_lengths_list.extend([query.seq_length] * self.beam_width)
                self.reuse_lengths_list.extend([QueryHelper.decoder_prefix_length(self.count_prefix_length, query)] * self.beam_width)
                self.context_lengths_list.extend([query.context_length] * self.beam_width)

            else:
                batch_idx = self.generate_batch_size * self.beam_width + i - self.generate_batch_size
                cache_block_indice[batch_idx, :len(query.block_indice[0])] = query.block_indice[0]
                output_token_ids[batch_idx, :query.seq_length] = query.output_token_ids[0]
                self.seq_lengths_list.extend([query.seq_length])
                self.reuse_lengths_list.append(QueryHelper.context_prefix_length(query))
                self.context_lengths_list.append(QueryHelper.context_input_length(self.count_prefix_length, query))
            if query.adapter_name is not None:
                lora_names += [query.adapter_name]
                lora_ids += [query.lora_resource.get_id(query.adapter_name)]

            self.generate_configs.append(query.generate_config)
        self.seq_lengths_list = self.seq_lengths_list[:self.generate_batch_size * self.beam_width]
        self.output_token_ids = torch.IntTensor(output_token_ids)
        self.images = images
        self.cache_block_indice = torch.IntTensor(cache_block_indice)
        self.merge_generate_config = self.union_generate_config(self.generate_configs)
        self.lora_names = lora_names
        self.lora_ids = lora_ids
        if len(self.lora_ids) == 0:
            self.lora_ids = [-1]
        self.check()

    def clear(self):
        self.generate_batch_size: int = 0
        self.merge_generate_config = GenerateConfig()
        self.context_batch_size: int = 0
        self.beam_width: int = 1
        self.generate_configs = []
        self.seq_lengths_list = []
        self.reuse_lengths_list = []
        self.context_lengths_list = []
        self.cache_block_indice = None
        self.output_token_ids = None
        self.record_index_prob = None

    def record_update_tensors(self, finished: torch.Tensor, update_length: List[int], hidden_states: torch.Tensor, logits: torch.Tensor,
                              cum_log_probs: torch.Tensor, updated_token_ids: torch.Tensor,
                              output_log_probs: Optional[torch.Tensor] = None,
                              output_index_prob: Optional[torch.Tensor] = None,
                              medusa_states: Optional[List[Any]] = None) -> None:
        self.finished = to_cpu(finished)
        self.hidden_states = to_cpu(hidden_states)
        self.logits = to_cpu(logits)
        self.cum_log_probs = to_cpu(cum_log_probs)
        self.updated_token_ids = to_cpu(updated_token_ids)
        self.update_length = update_length

        # not to cpu
        self.medusa_state = medusa_states
        self.output_log_probs = output_log_probs
        self.output_index_prob = output_index_prob

    @property
    def max_token_len(self):
        return max(self.max_seq_length, self.max_context_length)

    def slice_output_token(self, start: int, end: int, gen_len: int) -> torch.Tensor:
        assert self.updated_token_ids is not None
        max_token_len = max(self.max_seq_length, self.max_context_length)
        return self.updated_token_ids[start: end, max_token_len: max_token_len + gen_len].contiguous()

    # generate config for sample
    # TODO: do not gen generate config, gen sample config
    @staticmethod
    def union_generate_config(configs: List['GenerateConfig']):
        top_k: List[int] = []
        top_p: List[float] = []
        min_new_tokens: List[int] = []
        repetition_penalty: List[float] = []
        for config in configs:
            top_k.append(config.top_k)
            top_p.append(config.top_p)
            min_new_tokens.append(config.min_new_tokens)
            repetition_penalty.append(config.repetition_penalty)

        res = GenerateConfig(
            top_k=top_k,
            top_p=top_p,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            eos_token_id=configs[0].eos_token_id,
            num_beams=configs[0].num_beams,
        )
        res.gen_hash_value()
        return res

class QueryHelper(object):
    @staticmethod
    def context_prefix_length(query: QueryStats) -> int:
        return query.reuse_length

    @staticmethod
    def decoder_prefix_length(count_prefix_length: bool, query: QueryStats) -> int:
        if count_prefix_length:
            return 0
        else:
            return query.reuse_length

    @staticmethod
    def context_input_length(count_prefix_length: bool, query: QueryStats) -> int:
        if not count_prefix_length:
            context_input_token_num = query.context_length
        else:
            context_input_token_num =  query.context_length - query.reuse_length
        return context_input_token_num
