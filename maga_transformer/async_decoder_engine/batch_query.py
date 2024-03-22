import copy
import torch
import logging
import numpy as np
from typing import Any, List, Optional, Tuple
from threading import Lock
from pydantic import BaseModel
from PIL import Image

from maga_transformer.utils.time_util import current_time_ms
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.utils.stop_utils import create_stop_criteria_list
from maga_transformer.async_decoder_engine.ptuning.ptuning import PrefixInfo
from maga_transformer.async_decoder_engine.generate_stream import GenerateStream
from maga_transformer.utils.util import to_cuda, to_cpu
from maga_transformer.metrics import kmonitor, GaugeMetrics

class ModelOutput(BaseModel):
    finished: Optional[torch.Tensor] = None
    update_length: List[int] = []
    update_token_ids: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    cum_log_probs: Optional[torch.Tensor] = None
    output_log_probs: Optional[torch.Tensor] = None
    output_index_prob: Optional[torch.Tensor] = None
    medusa_states: Optional[List[Any]] = None

    class Config:
        arbitrary_types_allowed = True

class BatchQuery:
    def __init__(self, gen_num_per_circle: int, nccl_op: Any) -> None:
        self.gen_num_per_circle = gen_num_per_circle
        self.nccl_op_ = nccl_op
        if g_parallel_info.tp_size > 1:
            assert self.nccl_op_ is not None, "nccl op should not be None when tp_size > 1"
        # input
        self.context_streams: List[GenerateStream] = []
        self.decode_streams: List[GenerateStream] = []
        self.generate_batch_size: int = 0
        self.context_batch_size: int = 0
        self.num_beams: int = 1
        self.cache_block_indice: torch.Tensor = torch.zeros((1,1), dtype=torch.int32)
        self.output_token_ids: torch.Tensor = torch.zeros((1,1), dtype=torch.int32)
        self.images: List[Any] = []
        self.generate_configs: List[GenerateConfig] = []
        self.merge_generate_config: GenerateConfig = GenerateConfig()
        self.seq_lengths_list: List[int] = []
        self.reuse_lengths_list: List[int] = []
        self.context_lengths_list: List[int] = []
        self.record_index_prob: Optional[torch.Tensor] = None
        self.lora_ids: List[int] = []
        self.calculate_loss: List[int] = []
        self._ptuning_info = PrefixInfo()

        self.model_output = ModelOutput()

    def __str__(self):
        return f'generate_batch_size: {self.generate_batch_size}, \
                context_batch_size: {self.context_batch_size}, \
                output_token_ids: {self.output_token_ids}, \
                seq_length: {self.seq_lengths_list},\
                context_length: {self.context_lengths_list}, \
                lora_ids: {self.lora_ids}'

    def deepcopy(self) -> 'BatchQuery':
        new_batch_query = BatchQuery(self.gen_num_per_circle, self.nccl_op_)
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
        new_batch_query.calculate_loss = copy.deepcopy(self.calculate_loss)
        new_batch_query._ptuning_info = self._ptuning_info
        new_batch_query.model_output.update_length = self.model_output.update_length
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
            self._ptuning_info.ptuning,
            self._ptuning_info.count_length,
            self._ptuning_info.count_prefix_length,
            check_num2
        ])
        shape_hints = to_cuda(shape_hints)
        self.nccl_op_.broadcast_tp([shape_hints])
        torch.cuda.current_stream().synchronize()
        shape_hints = shape_hints.cpu().numpy()
        assert shape_hints[0] == check_num and shape_hints[-1] == check_num2, 'check sum error'

        if g_parallel_info.tp_rank == 0:
            seq_lengths_tensor = to_cuda(torch.IntTensor(self.seq_lengths_list))
            reuse_lengths_tensor = to_cuda(torch.IntTensor(self.reuse_lengths_list))
            context_lengths_tensor = to_cuda(torch.IntTensor(self.context_lengths_list))
            output_token_ids = to_cuda(self.output_token_ids)
            cache_block_indice = to_cuda(self.cache_block_indice)
            lora_ids_tensor = to_cuda(torch.IntTensor(self.lora_ids))
            calculate_loss_tensor = to_cuda(torch.IntTensor(self.calculate_loss))
        else:
            self.generate_batch_size = int(shape_hints[1])
            self.context_batch_size = int(shape_hints[2])
            self.num_beams = int(shape_hints[3])
            output_token_ids = torch.zeros((self.decoder_batch_size, int(shape_hints[4])), dtype=torch.int32, device="cuda:0")
            cache_block_indice = torch.zeros(
                (self.decoder_batch_size, int(shape_hints[5])), dtype=torch.int32, device="cuda:0")
            self._ptuning_info.ptuning = bool(shape_hints[6])
            self._ptuning_info.count_length = bool(shape_hints[7])
            self._ptuning_info.count_prefix_length = bool(shape_hints[8])
            seq_lengths_tensor = torch.zeros((self.generate_batch_size * self.num_beams), dtype=torch.int32, device="cuda:0")
            reuse_lengths_tensor = torch.zeros((self.decoder_batch_size), dtype=torch.int32, device="cuda:0")
            context_lengths_tensor = torch.zeros((self.decoder_batch_size), dtype=torch.int32, device="cuda:0")
            lora_ids_tensor = torch.zeros((max(1, self.total_batch_size)), dtype=torch.int32, device="cuda:0")
            calculate_loss_tensor = torch.zeros((self.context_batch_size), dtype=torch.int32, device="cuda:0")
        self.nccl_op_.broadcast_tp([
            cache_block_indice, output_token_ids, seq_lengths_tensor,
            reuse_lengths_tensor, context_lengths_tensor, lora_ids_tensor,
            calculate_loss_tensor
        ])
        torch.cuda.current_stream().synchronize()
        if g_parallel_info.tp_rank > 0:
            self.cache_block_indice = to_cpu(cache_block_indice)
            self.output_token_ids = to_cpu(output_token_ids)
            self.seq_lengths_list = to_cpu(seq_lengths_tensor).numpy().tolist()
            self.reuse_lengths_list = to_cpu(reuse_lengths_tensor).numpy().tolist()
            self.context_lengths_list = to_cpu(context_lengths_tensor).numpy().tolist()
            self.lora_ids = to_cpu(lora_ids_tensor).numpy().tolist()
            self.calculate_loss = to_cpu(calculate_loss_tensor).numpy().tolist()

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
        if self._ptuning_info.count_prefix_length:
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
        self.context_streams = []
        for stream in new_streams:
            if not stream.set_running():
                # because the query thread will call set_stop. We check the status with a lock here, and the stopped stream is discarded directly.
                stream.release_resource()
            else:
                self.context_streams.append(stream)
        self.clear()
        
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

    @property
    def streams_num(self):
        return len(self.streams)

    def empty(self):
        return self.streams_num == 0

    def generate_model_input(self):
        if g_parallel_info.tp_rank > 0:
            return
        total_batch_size = len(self.streams)
        if total_batch_size > 0:
            self.num_beams = self.streams[0].generate_config.num_beams
        self._ptuning_info = self.streams[0].ptuning_info
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
            self.reuse_lengths_list.extend([stream.reuse_length * (1 - int(self._ptuning_info.count_prefix_length))] * self.num_beams)
            self.context_lengths_list.extend([stream.input_length] * self.num_beams)

        images = []
        for idx, stream in enumerate(self.context_streams):
            images.extend(stream.images)
            batch_idx = self.generate_batch_size * self.num_beams + idx
            cache_block_indice[batch_idx, :len(stream.block_indice[0])] = stream.block_indice[0]
            output_token_ids[batch_idx, :stream.seq_length] = stream.complete_token_ids[0]
            self.seq_lengths_list.extend([stream.seq_length])
            self.reuse_lengths_list.append(stream.reuse_length)
            self.context_lengths_list.append(stream.seq_length - stream.reuse_length * int(self._ptuning_info.count_prefix_length))

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
        self.calculate_loss = [c.calculate_loss for c in self.generate_configs[self.generate_batch_size:]]
        self.check()

    def update_all_errors(self, err: str):
        for stream in self.streams:
            stream.stop_and_release(err)
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

    def update_output(self, model_output):
        self.model_output = model_output

    @property
    def max_token_len(self):
        return max(self.max_seq_length, self.max_context_length)

    def slice_output_token(self, start: int, end: int, slice_len: int, update_from_pos: int = -1) -> torch.Tensor:
        assert self.model_output.update_token_ids is not None
        if update_from_pos < 0:
            update_from_pos = max(self.max_seq_length, self.max_context_length)
        return self.model_output.update_token_ids[start: end, update_from_pos: update_from_pos + slice_len].contiguous()

    def update_streams(self):
        def try_get(list, idx1, idx2):
            return list[idx1:idx2] if list is not None else None

        finished = self.model_output.finished.tolist()
        for i, stream in enumerate(self.streams):
            start_idx = i * self.num_beams
            end_idx = (i + 1) * self.num_beams
            num_new_tokens = self.model_output.update_length[i] # for sepculative decoding
            stream.medusa_state = self.model_output.medusa_states[i] if self.model_output.medusa_states else None

            new_tokens = self.slice_output_token(
                start_idx, end_idx, num_new_tokens).reshape(self.num_beams, -1)
            if (self.num_beams > 1) and (start_idx < len(self.seq_lengths_list)):
                # previous generated tokens
                generate_start_pos = self.context_lengths_list[start_idx]
                generated_length = self.seq_lengths_list[start_idx] - generate_start_pos
                previous_tokens = self.slice_output_token(
                    start_idx, end_idx, generated_length, generate_start_pos).reshape(self.num_beams, -1)
                new_tokens = torch.concatenate([previous_tokens, new_tokens], dim=1)
            stream.update(new_tokens,
                          num_new_tokens,
                          finished[start_idx],
                          try_get(self.model_output.hidden_states, start_idx, end_idx),
                          try_get(self.model_output.logits, start_idx, end_idx),
                          try_get(self.model_output.cum_log_probs, start_idx, end_idx))
            stream.check_timeout()
            if stream.finished or stream.stopped:
                self.remove_stream(stream)
                stream.release_resource()

    def get_prefix_args(self) -> Tuple[torch.IntTensor, torch.BoolTensor, torch.IntTensor]:
        count_length = torch.BoolTensor([self._ptuning_info.count_length])
        if self._ptuning_info.ptuning:
            max_length = 0 if self.generate_batch_size == 0 else \
                max(self.reuse_lengths_list[:self.generate_batch_size * self.num_beams])
        else:
            max_length = 0
        max_prefix_length = torch.IntTensor([max_length])
        prefix_lengths = torch.IntTensor(self.reuse_lengths_list)
        return prefix_lengths, count_length, max_prefix_length
