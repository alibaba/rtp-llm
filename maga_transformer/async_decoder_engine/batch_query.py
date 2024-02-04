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
from maga_transformer.async_decoder_engine.generate_stream import GenerateStream
from maga_transformer.utils.util import to_cuda, to_cpu
from maga_transformer.metrics import kmonitor, GaugeMetrics
from maga_transformer.tokenizer.tokenizer_base import TokenizerBase


class BatchQuery:
    def __init__(self, count_prefix_length:bool, gen_num_per_circle: int, nccl_op: Any) -> None:
        self.gen_num_per_circle = gen_num_per_circle
        self.nccl_op_ = nccl_op
        if g_parallel_info.tp_size > 1:
            assert self.nccl_op_ is not None, "nccl op should not be None when tp_size > 1"
        # input
        self.count_prefix_length = count_prefix_length
        self.context_streams: List[GenerateStream] = []
        self.decode_streams: List[GenerateStream] = []
        self.generate_batch_size: int = 0
        self.context_batch_size: int = 0
        self.num_beams: int = 1
        self.cache_block_indice: torch.Tensor = torch.zeros((1,1), dtype=torch.int32)
        self.output_token_ids: torch.Tensor = torch.zeros((1,1), dtype=torch.int32)
        self.images: List[str] = []
        self.generate_configs: List[GenerateConfig] = []
        self.merge_generate_config: GenerateConfig = GenerateConfig()
        self.seq_lengths_list: List[int] = []
        self.reuse_lengths_list: List[int] = []
        self.context_lengths_list: List[int] = []
        self.record_index_prob: Optional[torch.Tensor] = None
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

    def __str__(self):
        return f'generate_batch_size: {self.generate_batch_size}, \
                context_batch_size: {self.context_batch_size}, \
                output_token_ids: {self.output_token_ids}, \
                seq_length: {self.seq_lengths_list},\
                context_length: {self.context_lengths_list}, \
                lora_ids: {self.lora_ids}'

    def deepcopy(self) -> 'BatchQuery':
        new_batch_query = BatchQuery(self.count_prefix_length, self.gen_num_per_circle, self.nccl_op_)
        new_batch_query.context_streams = copy.copy(self.context_streams)
        new_batch_query.decode_streams = copy.copy(self.decode_streams)
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
        new_batch_query.lora_ids = copy.deepcopy(self.lora_ids)
        new_batch_query.update_length = self.update_length
        return new_batch_query

    def tp_sync(self):
        if g_parallel_info.tp_size <= 1:
            return
        check_num: int = 998244353
        check_num2: int = 1000000007
        shape_hints = torch.IntTensor([
            check_num,
            self.generate_batch_size, self.context_batch_size, self.num_beams,
            self.output_token_ids.shape[1], self.cache_block_indice.shape[1],
            check_num2
        ])
        shape_hints = to_cuda(shape_hints)
        self.nccl_op_.broadcast_tp([shape_hints])
        torch.cuda.current_stream().synchronize()
        shape_hints = shape_hints.cpu().numpy()
        assert shape_hints[0] == check_num and shape_hints[-1] == check_num2

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
            self.num_beams = int(shape_hints[3])
            output_token_ids = torch.zeros((self.decoder_batch_size, int(shape_hints[4])), dtype=torch.int32, device="cuda:0")
            cache_block_indice = torch.zeros(
                (self.decoder_batch_size, int(shape_hints[5])), dtype=torch.int32, device="cuda:0")
            seq_lengths_tensor = torch.zeros((self.generate_batch_size * self.num_beams), dtype=torch.int32, device="cuda:0")
            reuse_lengths_tensor = torch.zeros((self.decoder_batch_size), dtype=torch.int32, device="cuda:0")
            context_lengths_tensor = torch.zeros((self.decoder_batch_size), dtype=torch.int32, device="cuda:0")
            lora_ids_tensor = torch.zeros((max(1, self.total_batch_size)), dtype=torch.int32, device="cuda:0")
        self.nccl_op_.broadcast_tp([
            cache_block_indice, output_token_ids, seq_lengths_tensor,
            reuse_lengths_tensor, context_lengths_tensor, lora_ids_tensor
        ])
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
        return self.num_beams * self.generate_batch_size + self.context_batch_size

    @property
    def total_batch_size(self):
        return self.generate_batch_size + self.context_batch_size

    def has_context_query(self):
        return self.context_batch_size > 0

    @property
    def context_query_reuse_lengths_list(self) -> List[int]:
        assert len(self.reuse_lengths_list) == self.generate_batch_size * self.num_beams + self.context_batch_size
        return self.reuse_lengths_list[self.generate_batch_size * self.num_beams:]

    @property
    def context_query_context_lengths_list(self) -> List[int]:
        return self.context_lengths_list[self.generate_batch_size * self.num_beams:]

    def context_query_output_tokens(self, index: int) -> torch.Tensor:
        index = index + self.generate_batch_size * self.num_beams
        start_index = 0
        end_index = self.context_lengths_list[index]
        # 除去ptuningv2以外，前缀token不参与计算
        if self.count_prefix_length:
            start_index += self.reuse_lengths_list[index]
            end_index += self.reuse_lengths_list[index]

        return self.output_token_ids[index, start_index: end_index]

    def generate_query_last_token(self, index: int) -> torch.Tensor:
        assert index < self.generate_batch_size
        start_idx = index * self.num_beams
        end_idx = (index + 1) * self.num_beams
        return self.output_token_ids[start_idx: end_idx, self.seq_lengths_list[start_idx] - 1]

    def check(self):
        assert len(self.context_lengths_list) == self.decoder_batch_size
        assert len(self.reuse_lengths_list) == self.decoder_batch_size
        assert len(self.seq_lengths_list) == self.generate_batch_size * self.num_beams
        assert len(self.generate_configs) == self.total_batch_size
        for length in self.seq_lengths_list + self.context_lengths_list:
            if length <= 0:
                raise Exception(
                    f"got length not valid, length_list: {self.seq_lengths_list} {self.context_lengths_list}"
                )

    def add_new_stream(self, new_streams: List[GenerateStream]):
        if g_parallel_info.tp_rank > 0:
            return
        self.decode_streams.extend(self.context_streams)
        self.context_streams = new_streams
        self.clear()
        [q.set_running() for q in self.context_streams]

        self.generate_batch_size = len(self.decode_streams)
        self.context_batch_size = len(self.context_streams)

    def remove_stream(self, stream):
        if stream in self.decode_streams:
            self.decode_streams.remove(stream)
        if stream in self.context_streams:
            self.context_streams.remove(stream)

    @property
    def streams(self):
        return self.decode_streams + self.context_streams

    def generate_model_input(self):
        if g_parallel_info.tp_rank > 0:
            return
        total_batch_size = len(self.streams)
        if total_batch_size > 0:
            self.num_beams = self.streams[0].generate_config.num_beams

        cache_block_indice = np.zeros(
            [self.decoder_batch_size, max([len(q.block_indice[0]) for q in self.streams])],
            dtype=np.int32
        )
        output_token_ids = np.zeros(
            [self.decoder_batch_size, max([q.seq_length for q in self.streams]) + self.gen_num_per_circle],
            dtype=np.int32
        )

        for idx, stream in enumerate(self.decode_streams):
            start_batch_idx = idx * self.num_beams
            end_batch_idx = start_batch_idx + self.num_beams
            cache_block_indice[start_batch_idx:end_batch_idx, :len(stream.block_indice[0])] = stream.block_indice
            output_token_ids[start_batch_idx:end_batch_idx, :stream.seq_length] = stream.complete_token_ids
            self.seq_lengths_list.extend([stream.seq_length] * self.num_beams)
            self.reuse_lengths_list.extend([QueryHelper.decoder_prefix_length(self.count_prefix_length, stream)] * self.num_beams)
            self.context_lengths_list.extend([stream.input_length] * self.num_beams)

        images = []
        for idx, stream in enumerate(self.context_streams):
            images.extend(stream.images)
            batch_idx = self.generate_batch_size * self.num_beams + idx
            cache_block_indice[batch_idx, :len(stream.block_indice[0])] = stream.block_indice[0]
            output_token_ids[batch_idx, :stream.seq_length] = stream.complete_token_ids[0]
            self.seq_lengths_list.extend([stream.seq_length])
            self.reuse_lengths_list.append(QueryHelper.context_prefix_length(stream))
            self.context_lengths_list.append(stream.seq_length - stream.reuse_length * int(self.count_prefix_length))

        lora_ids = []
        for stream in self.streams:
            lora_ids.append(stream.lora_id)
            self.generate_configs.append(stream.generate_config)

        self.seq_lengths_list = self.seq_lengths_list[:self.generate_batch_size * self.num_beams]
        self.output_token_ids = torch.IntTensor(output_token_ids)
        self.images = images
        self.cache_block_indice = torch.IntTensor(cache_block_indice)
        self.merge_generate_config = GenerateConfig.merge_generate_config(self.generate_configs)
        self.lora_ids = lora_ids
        self.check()

    def update_all_errors(self, err: str):
        for s in self.streams:
            s.set_stop(err)
        self.context_streams.clear()
        self.decode_streams.clear()

    def clear(self):
        self.generate_batch_size: int = 0
        self.merge_generate_config = GenerateConfig()
        self.context_batch_size: int = 0
        self.num_beams: int = 1
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
        self.hidden_states = hidden_states
        self.logits = logits
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
    def context_prefix_length(stream) -> int:
        return stream.reuse_length

    @staticmethod
    def decoder_prefix_length(count_prefix_length: bool, stream) -> int:
        if count_prefix_length:
            return 0
        else:
            return stream.reuse_length
