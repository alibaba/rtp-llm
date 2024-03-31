import os
import torch
import logging
import random
from enum import Enum
from typing import List, Optional, Tuple, Any
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.async_decoder_engine.batch_query import BatchQuery, ModelOutput
from maga_transformer.async_decoder_engine.cache_manager import CacheManager
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.utils.sample_utils import BaseSampler, SamplerSetupParams, SamplingParams
from maga_transformer.utils.dump_config_utils import dump_engine_to_table
from maga_transformer.models.base_model import BaseModel
from maga_transformer.ops.gpt_ops.gpt_op import GptOp
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.utils.util import to_cuda, to_cpu

DEFAULT_NEW_SAMPLER_BATCH_SIZE=128

class ModelType(Enum):
    Normal = "normal"
    Speculative = "speculative"
    Medusa = 'medusa'

class ModelOps(object):
    def __init__(self, type: ModelType, model: BaseModel, config: GptInitModelParameters, gpt_op: GptOp,
                 generate_config: GenerateConfig, sampler: BaseSampler):
        self.type = type
        self.model = model
        self.config = config
        self.gpt_op = gpt_op
        self.generate_config = generate_config
        self.sampler = sampler

class ExecutorBase(object):
    def process(self, batch_query: BatchQuery) -> None:
        raise NotImplementedError()

class NormalModelExecutor(ExecutorBase):
    def __init__(self, model_ops: ModelOps, cache_manager: CacheManager):
        self.model_ops = model_ops
        self.cache_manager_ = cache_manager
        dump_engine_to_table(self.create_config_json())

    @property
    def base_model_ops(self):
        return self.model_ops

    @staticmethod
    def _to_cuda_tensor(t: Optional[List[Any]], dtype: torch.dtype=torch.int32):
        return to_cuda(torch.tensor(t, dtype=dtype)) if t is not None else None

    def process(self, batch_query: BatchQuery) -> None:
        all_hidden_states = self._process(batch_query)
        hidden_states = self._select_last_hidden_states(batch_query, all_hidden_states)
        logits = self._post_transformer_nn(hidden_states)
        self._calculate_loss(batch_query, all_hidden_states)
        if g_parallel_info.tp_size > 1 and g_parallel_info.tp_rank > 0:
            return
        with torch.cuda.nvtx.range('post_process'):
            self._post_process(batch_query, logits, hidden_states)

    def create_config_json(self):
        config_json = {
            "executor_type": type(self).__name__,
        }
        return config_json

    def _select_last_hidden_states(self, batch_query: BatchQuery, hidden_states: torch.Tensor):
        index_list = list(range(0, batch_query.generate_batch_size * batch_query.num_beams))
        offset = batch_query.generate_batch_size * batch_query.num_beams - 1
        for i in range(0, batch_query.context_batch_size):
            offset = offset + batch_query.context_query_context_lengths_list[i]
            index_list.append(offset)
        return hidden_states.index_select(0, torch.tensor(index_list, device="cuda:0"))

    def _select_context_hidden_states(self, batch_query: BatchQuery, hidden_states: torch.Tensor, idx):
        offset = batch_query.generate_batch_size * batch_query.num_beams
        for i in range(idx):
            offset += batch_query.context_query_context_lengths_list[i]
        return hidden_states[offset:offset + batch_query.context_query_context_lengths_list[idx],...]

    def _process(self, batch_query: BatchQuery) -> torch.Tensor:
        with torch.cuda.nvtx.range('pre_process'):
            input_embeds, attention_mask, position_ids = self._pre_process(batch_query)
            k_cache, v_cache = self.cache_manager_.get_kv_cache_base()
            k_cache_scale, v_cache_scale = self.cache_manager_.get_kv_cache_scale_base()
            prefix_lengths, count_length, max_prefix_length = batch_query.get_prefix_args()

        with torch.cuda.nvtx.range('run_model'):
            hidden_states = self.model_ops.gpt_op.forward(
                decoder_input=input_embeds,
                key_cache=k_cache,
                value_cache=v_cache,
                key_cache_scale=k_cache_scale,
                value_cache_scale=v_cache_scale,
                input_lengths=torch.tensor(batch_query.context_lengths_list, dtype=torch.int32),
                sequence_lengths=torch.tensor([i - 1 for i in batch_query.seq_lengths_list], dtype=torch.int32),
                block_index_map=batch_query.cache_block_indice,
                position_ids=position_ids,
                attention_mask=attention_mask,
                linear_bias_slopes=self.model_ops.model.linear_bias_slopes,
                prefix_lengths=prefix_lengths,
                count_length=count_length,
                max_prefix_length=max_prefix_length,
                lora_ids=torch.IntTensor(batch_query.lora_ids))

        return hidden_states

    def _create_position_ids_for_rotary(self, batch_query: BatchQuery) -> Optional[torch.Tensor]:
        if self.model_ops.model.position_encoding is None:
            return None
        # generate query
        position_ids = [i - 1 for i in batch_query.seq_lengths_list]
        # context query
        for i in range(batch_query.generate_batch_size, batch_query.total_batch_size):
            position_ids.extend(range(batch_query.reuse_lengths_list[i], batch_query.reuse_lengths_list[i] + batch_query.context_lengths_list[i]))
        return to_cuda(torch.IntTensor(position_ids))

    def _packed_tokens(self, batch_query: BatchQuery) -> Tuple[torch.Tensor, List[Any]]:
        combo_tokens: List[int] = []
        combo_imgs: List[Any] = []
        for i in range(batch_query.generate_batch_size):
            combo_tokens.extend(batch_query.generate_query_last_token(i).numpy().tolist())
        for i in range(batch_query.context_batch_size):
            combo_tokens.extend(batch_query.context_query_output_tokens(i).numpy().tolist())
            combo_imgs = batch_query.images
        if (not self.model_ops.config.is_multimodal):
            if any([t < 0 or t >= self.model_ops.config.vocab_size for t in combo_tokens]):
                raise Exception(f'tokens: {combo_tokens} not in vocab_size: {self.model_ops.config.vocab_size}')
        else:
            special_set = set([v for v in self.model_ops.config.vit_related_params.vit_special_token_ids.values()])
            if any([((t < 0 or t >= self.model_ops.config.vocab_size) and (t not in special_set)) for t in combo_tokens]):
                raise Exception(f'tokens: {combo_tokens} not in vocab_size: {self.model_ops.config.vocab_size}')
        return to_cuda(torch.IntTensor(combo_tokens)), combo_imgs

    # static for ut
    @staticmethod
    def append_reuse_mask(attention_mask: torch.Tensor, context_length_list: List[int],
                           reuse_length_list: List[int]) -> torch.Tensor:

        max_reuse_length = max(reuse_length_list)
        if max_reuse_length == 0:
            return attention_mask
        max_context_length = max(context_length_list)
        batch_size = attention_mask.size(0)
        final_attention_mask = torch.cat((attention_mask, torch.zeros(batch_size, max_context_length, max_reuse_length).to(attention_mask)), dim=-1)
        for i in range(0, batch_size):
            final_attention_mask[i] = final_attention_mask[i].roll(reuse_length_list[i], dims=-1)
            final_attention_mask[i, :context_length_list[i], :reuse_length_list[i]] = 1
        return final_attention_mask

    def _pre_process(
        self, batch_query: BatchQuery,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        model = self.model_ops.model
        combo_tokens, images = self._packed_tokens(batch_query)
        position_ids = self._create_position_ids_for_rotary(batch_query)

        assert model.word_embedding is not None
        input_embeds = model.async_input_word_embedding(combo_tokens, [images])

        if model.position_encoding is not None:
            input_embeds += model.position_encoding(position_ids)

        if self.model_ops.model.pre_decoder_layernorm is not None:
            input_embeds = model.pre_decoder_layernorm(input_embeds)

        if self.model_ops.gpt_op.use_fmha:
            attention_mask = None
        else:
            attention_mask = self._create_context_attention_mask(batch_query)

        return input_embeds, attention_mask, position_ids

    def _create_context_attention_mask(self, batch_query: BatchQuery):
        if batch_query.has_context_query():
            attention_mask = self.model_ops.model.create_context_decoder_mask(
                batch_query.context_query_context_lengths_list)
            attention_mask = self.append_reuse_mask(
                attention_mask, batch_query.context_query_context_lengths_list, batch_query.context_query_reuse_lengths_list)
            return attention_mask
        return None

    def _extend_tensor_for_new_beams(self, tensor: torch.Tensor, batch_query: BatchQuery) -> torch.Tensor:
        original_shape = tensor.shape
        assert (original_shape[0] == batch_query.decoder_batch_size)
        start_idx = batch_query.generate_batch_size * batch_query.num_beams
        end_idx = start_idx + batch_query.context_batch_size
        extended_tensor = self.model_ops.model.dup_dim0_for_beam_search(
            tensor[start_idx: end_idx], batch_query.num_beams
        )
        tensor = torch.concat([tensor[:start_idx], extended_tensor], dim=0)
        return tensor

    def _prepare_kv_cache_for_beams(self, batch_query: BatchQuery,
                                    key_cache: torch.Tensor, value_cache: torch.Tensor
    ) -> None:
        generate_block_rows = batch_query.generate_batch_size * batch_query.num_beams
        new_blocks: List[torch.Tensor] = [batch_query.cache_block_indice[:generate_block_rows]]
        for idx in range(batch_query.context_batch_size):
            query_cache_blocks = batch_query.cache_block_indice[generate_block_rows + idx]
            block_num = int(query_cache_blocks.count_nonzero().item())

            allocation_row_num = batch_query.num_beams - 1
            new_block_num = self.cache_manager_.malloc(block_num * allocation_row_num)
            query_new_blocks = torch.Tensor(new_block_num, device=query_cache_blocks.device) \
                .reshape(allocation_row_num, block_num) \
                .type_as(query_cache_blocks)
            query_new_blocks = torch.concat([
                query_new_blocks,
                torch.zeros([allocation_row_num, len(query_cache_blocks) - block_num],
                            dtype=torch.int32, device=query_cache_blocks.device)
            ], dim=1)

            src_blocks = query_cache_blocks[:block_num]
            for beam_idx in range(allocation_row_num):
                target_blocks = query_new_blocks[beam_idx, :block_num]
                key_cache[:, target_blocks] = key_cache[:, src_blocks]
                value_cache[:, target_blocks] = value_cache[:, src_blocks]

            query_new_blocks = torch.concat([query_cache_blocks.unsqueeze(0), query_new_blocks], dim=0)
            batch_query.context_streams[idx].set_kvcache(query_new_blocks[:,:block_num].tolist(),
                                                         batch_query.context_streams[idx].reuse_length)
            new_blocks.extend([query_new_blocks])

        batch_query.cache_block_indice = torch.concat(new_blocks, dim=0)

    def _prepare_beam_search(self, batch_query: BatchQuery, logits: torch.Tensor,
                             key_cache: torch.Tensor, value_cache: torch.Tensor
    ) -> torch.Tensor:
        logits = self._extend_tensor_for_new_beams(logits, batch_query)
        batch_query.output_token_ids = self._extend_tensor_for_new_beams(batch_query.output_token_ids, batch_query)
        self._prepare_kv_cache_for_beams(batch_query, key_cache, value_cache)
        return logits

    def _post_transformer_nn(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.model_ops.model.post_decoder_layernorm is not None:
            hidden_states = self.model_ops.model.post_decoder_layernorm(hidden_states)
        assert self.model_ops.model.lm_head is not None
        logits = self.model_ops.model.lm_head(hidden_states).float()

        if 'CHECK_LOGITS_NAN' in os.environ:
            logits_cpu = to_cpu(logits.view(-1))
            if any(torch.isnan(logits_cpu).numpy().tolist()):
                raise Exception(f'logits has nan: {logits_cpu}')

        return logits

    def _reconstruct_sampler(self, batch_query: BatchQuery) -> None:
        if self.model_ops.generate_config.is_same(batch_query.merge_generate_config):
            return

        new_config = batch_query.merge_generate_config
        new_config.random_seed = [random.randint(0, 1000000000) for _ in range(batch_query.total_batch_size)]
        self.model_ops.generate_config = new_config
        if new_config.num_beams != self.model_ops.sampler.config.num_beams:
            self.model_ops.sampler = self.model_ops.model.create_sampler(new_config)
        else:
            self.model_ops.sampler.config = new_config
        self.model_ops.sampler.setup(SamplerSetupParams(batch_query.total_batch_size, self.model_ops.config.special_tokens.eos_token_id, 0, None))

    def _calculate_loss(self, batch_query: BatchQuery, all_hidden_states: torch.Tensor):
        for context_idx, calculate_loss in enumerate(batch_query.calculate_loss):
            if not calculate_loss:
                continue
            hidden_states = self._select_context_hidden_states(
                batch_query, all_hidden_states, context_idx)
            logits = self._post_transformer_nn(hidden_states)
            if g_parallel_info.tp_size > 1 and g_parallel_info.tp_rank > 0:
                continue
            stream = batch_query.context_streams[context_idx]
            shift_labels = stream.complete_token_ids[0, 1:stream.input_length].type(torch.int64)
            shift_logits = logits[:stream.input_length - 1, ]
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.to("cuda:0"), shift_labels.to("cuda:0"))

            if stream.generate_config.calculate_loss == 1:
                loss_mean = loss.sum(dim=0) / (stream.input_length - 1)
                loss_mean = loss_mean.exp()
                stream.set_loss(loss_mean)
            elif stream.generate_config.calculate_loss == 2:
                stream.set_loss(loss)
            else:
                raise Exception("calculate_loss in generate_config can only be 0, 1 or 2")

    def _post_process(
            self,
            batch_query: BatchQuery,
            logits: torch.Tensor,
            hidden_states: torch.Tensor) -> None:
        key_cache, value_cache = self.cache_manager_.get_kv_cache_base()
        if batch_query.num_beams > 1:
            logits = self._prepare_beam_search(batch_query, logits, key_cache, value_cache)

        fake_input_lengths = batch_query.seq_lengths_list + batch_query.context_query_context_lengths_list
        # NOTE: beam width of all queries are assured to be the same. See Scheduler::check_query_to_append.
        gen_lengths = [
            fake_input_lengths[i] - batch_query.context_lengths_list[i] + batch_query.max_seq_length - 1 \
                for i in range(len(fake_input_lengths))
        ]
        input_lengths = self._to_cuda_tensor(fake_input_lengths)
        sequence_lengths = self._to_cuda_tensor(gen_lengths)
        # TODO: These tensors are allocated on each iteration. Try allocate them once for each query.

        finished = torch.zeros((batch_query.total_batch_size * batch_query.num_beams), device="cuda:0").bool()
        self._reconstruct_sampler(batch_query)

        token_ids = to_cuda(batch_query.output_token_ids.permute(1, 0).contiguous())

        cum_log_probs = torch.concat(
            [stream.cum_log_probs for stream in batch_query.streams], dim=0
        ).to("cuda:0")
        output_log_probs = torch.zeros((batch_query.total_batch_size), dtype=torch.float, device='cuda:0')
        index_log_prob = None
        if batch_query.record_index_prob is not None:
            index_log_prob = torch.zeros((batch_query.total_batch_size), dtype=torch.float, device='cuda:0')

        self.model_ops.sampler.do_sampling(SamplingParams(
            batch_query.max_token_len,
            batch_query.total_batch_size,
            batch_query.num_beams,
            batch_query.max_token_len,
            token_ids,
            logits.view(batch_query.total_batch_size, batch_query.num_beams, -1),
            finished,
            input_lengths,
            sequence_lengths,
            key_cache,
            value_cache,
            cum_log_probs,
            batch_query.cache_block_indice,
            output_log_probs,
            index_log_prob,
            batch_query.record_index_prob,
        ))

        output_token_ids = token_ids.permute(1, 0)
        batch_query.update_output(ModelOutput(
            finished=finished.cpu(),
            update_length=[1] * batch_query.total_batch_size,
            update_token_ids=output_token_ids.cpu(),
            hidden_states=hidden_states,
            logits=logits,
            cum_log_probs=cum_log_probs.cpu(),
            output_log_probs=output_log_probs,
            output_index_prob=index_log_prob))
