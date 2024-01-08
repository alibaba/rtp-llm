import os
import torch
import logging
from typing import List, Optional, Tuple, Union, Any
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.utils.model_weight import LoRAMap
from maga_transformer.async_decoder_engine.query_manager import QueryManager, BatchQuery
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.utils.sample_utils import BaseSampler, SamplerSetupParams, SamplingParams
from maga_transformer.utils.dump_config_utils import dump_engine_to_table
from maga_transformer.models.base_model import BaseModel
from maga_transformer.ops.gpt_ops.gpt_op import GptOp
from maga_transformer.distribute.worker_info import g_parallel_info
import random
from enum import Enum
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

class BaseModelExecutor(ExecutorBase):
    def __init__(self, model_ops: ModelOps, query_manager: QueryManager):
        self.model_ops = model_ops
        self.query_manager_ = query_manager
        dump_engine_to_table(self.create_config_json())

    @staticmethod
    def _to_cuda_tensor(t: Optional[List[Any]], dtype: torch.dtype=torch.int32):
        return to_cuda(torch.tensor(t, dtype=dtype)) if t is not None else None

    def process(self, batch_query: BatchQuery) -> None:
        all_hidden_states = self._process(batch_query)
        hidden_states = self._unpack_hidden_states(batch_query, all_hidden_states)
        if g_parallel_info.tp_size > 1 and g_parallel_info.tp_rank > 0:
            return
        with torch.cuda.nvtx.range('post_process'):
            self._post_process(batch_query, hidden_states, all_hidden_states)

    def create_config_json(self):
        config_json = {
            "engine_type": type(self).__name__,
        }
        config_json.update(self.query_manager_.create_config_json())
        return config_json

    def _unpack_hidden_states(self, batch_query: BatchQuery, hidden_states: torch.Tensor):
        index_lst = list(range(0, batch_query.generate_batch_size * batch_query.beam_width))
        offset = batch_query.generate_batch_size * batch_query.beam_width - 1
        for i in range(0, batch_query.context_batch_size):
            offset = offset + batch_query.context_query_context_lengths_list[i]
            index_lst.append(offset)
        return hidden_states.index_select(0, torch.tensor(index_lst, device="cuda:0"))

    def _process(self, batch_query: BatchQuery) -> torch.Tensor:
        # be1 = time.perf_counter()
        with torch.cuda.nvtx.range('pre_process'):
            input_embeds, attention_mask, position_ids = self._pre_process(batch_query)
            k_cache, v_cache = self.query_manager_.get_kv_cache_base()
            k_cache_scale, v_cache_scale = self.query_manager_.get_kv_cache_scale_base()
            prefix_lengths, count_length, max_prefix_length = self.query_manager_.get_prefix_args(batch_query)

            lora_ids = [self.model_ops.gpt_op.weight.lora_map.get_id(lora_name) for lora_name in batch_query.lora_names]
            # TODO(ldj) when tp > 1 broadcast lora ids.
            if len(lora_ids) == 0:
                lora_ids = [-1]

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
                lora_ids=torch.IntTensor(lora_ids))

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

        attention_mask = self._create_context_attention_mask(batch_query)

        return input_embeds, attention_mask, position_ids

    def _create_context_attention_mask(self, batch_query: BatchQuery):
        if batch_query.has_context_query():
            attention_mask = self.model_ops.model.create_context_decoder_mask(
                batch_query.context_query_context_lengths_list,
                max(batch_query.context_query_context_lengths_list))
            attention_mask = self.append_reuse_mask(
                attention_mask, batch_query.context_query_context_lengths_list, batch_query.context_query_reuse_lengths_list)
            return attention_mask
        return None

    def _extend_tensor_for_new_beams(self, tensor: torch.Tensor, batch_query: BatchQuery) -> torch.Tensor:
        original_shape = tensor.shape
        assert (original_shape[0] == batch_query.decoder_batch_size)
        start_idx = batch_query.generate_batch_size * batch_query.beam_width
        end_idx = start_idx + batch_query.context_batch_size
        extended_tensor = self.model_ops.model.dup_dim0_for_beam_search(
            tensor[start_idx: end_idx], batch_query.beam_width
        )
        tensor = torch.concat([tensor[:start_idx], extended_tensor], dim=0)
        return tensor

    def _prepare_kv_cache_for_beams(self, batch_query: BatchQuery,
                                    key_cache: torch.Tensor, value_cache: torch.Tensor
    ) -> None:
        generate_block_rows = batch_query.generate_batch_size * batch_query.beam_width
        new_blocks: List[torch.Tensor] = [batch_query.cache_block_indice[:generate_block_rows]]
        for idx in range(batch_query.context_batch_size):
            query_cache_blocks = batch_query.cache_block_indice[generate_block_rows + idx]
            block_num = int(query_cache_blocks.count_nonzero().item())

            allocation_row_num = batch_query.beam_width - 1
            new_block_num = self.query_manager_.cache_manager_.malloc(block_num * allocation_row_num)
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
            batch_query.queries[batch_query.generate_batch_size + idx].block_indice = \
                query_new_blocks[:,:block_num].tolist() # type: ignore

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
        # We use logits of fp32 type to avoid overflow issue.
        assert self.model_ops.model.lm_head is not None
        if self.model_ops.model.use_fp32_to_compute_logit:
            # The FT GPT op internally uses FP32 compute type for matrix multiplication.
            # This will produce the same result with the end-to-end FT's GPT op.
            logits = torch.nn.functional.linear(hidden_states.float(),
                                                self.model_ops.model.lm_head.weight)
        else:
            logits = self.model_ops.model.lm_head(hidden_states).float()
        # print('hidden_states2:', logits)
        return logits

    def _reset_sampler(self, batch_query: BatchQuery) -> None:
        new_config = batch_query.merge_generate_config
        new_config.random_seed = [random.randint(0, 1000000000) for _ in range(batch_query.total_batch_size)]
        self.model_ops.generate_config = new_config
        if new_config.num_beams != self.model_ops.sampler.config.num_beams:
            self.model_ops.sampler = self.model_ops.model.create_sampler(new_config)
        else:
            self.model_ops.sampler.config = new_config
        self.model_ops.sampler.setup(SamplerSetupParams(batch_query.total_batch_size, self.model_ops.config.special_tokens.eos_token_id, 0, None))

    def _calculate_loss(self,
            batch_query: BatchQuery, all_hidden_states: torch.Tensor):
        for query in batch_query.queries:
            if query.generate_config.calculate_loss and query.loss == None:
                all_logits = self._post_transformer_nn(all_hidden_states)
                break
        start_idx = 0
        for i, query in enumerate(batch_query.queries[:]):
            if not query.generate_config.calculate_loss:
                continue
            if query.loss != None:
                continue
            shift_labels = query.output_token_ids_[0, 1:query.context_length].type(torch.int64).contiguous()
            shift_logits = all_logits[start_idx : start_idx + query.context_length - 1, ].contiguous()
            start_idx += query.context_length
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.to("cuda:0"), shift_labels.to("cuda:0"))
            if query.generate_config.calculate_loss == 1:
                loss_mean = loss.sum(dim=0) / (query.context_length - 1)
                loss_mean = loss_mean.exp()
                query.loss = loss_mean
            elif query.generate_config.calculate_loss == 2:
                query.loss = loss
            else:
                raise Exception("calculate_loss in generate_config can only be 0, 1 or 2")

    def _post_process(
            self,
            batch_query: BatchQuery,
            hidden_states: torch.Tensor,
            all_hidden_states: torch.Tensor
    ) -> None:
        self._calculate_loss(batch_query, all_hidden_states)
        logits = self._post_transformer_nn(hidden_states)

        beam_width = batch_query.beam_width
        key_cache, value_cache = self.query_manager_.get_kv_cache_base()
        if beam_width > 1:
            logits = self._prepare_beam_search(batch_query, logits, key_cache, value_cache)

        fake_input_lengths = batch_query.seq_lengths_list + batch_query.context_query_context_lengths_list
        total_batch_size = batch_query.total_batch_size
        # NOTE: beam width of all queries are assured to be the same. See QueryManager::check_query_to_append.
        gen_lengths = [
            fake_input_lengths[i] - batch_query.context_lengths_list[i] + batch_query.max_seq_length - 1 \
                for i in range(len(fake_input_lengths))
        ]
        input_lengths = self._to_cuda_tensor(fake_input_lengths)
        sequence_lengths = self._to_cuda_tensor(gen_lengths)
        assert sequence_lengths is not None
        assert input_lengths is not None

        # TODO: These tensors are allocated on each iteration. Try allocate them once for each query.
        finished = torch.zeros((total_batch_size * beam_width), device="cuda:0").bool()
        if not self.model_ops.generate_config.is_same(batch_query.merge_generate_config):
            self._reset_sampler(batch_query)

        # shape(max_length + gen_num, batch_size)
        token_ids = to_cuda(batch_query.output_token_ids.permute(1, 0).contiguous())
        cum_log_probs = torch.concat(
            [query.cum_log_probs for query in batch_query.queries], dim=0
        ).to("cuda:0")

        output_log_probs = torch.zeros((batch_query.total_batch_size), dtype=torch.float, device='cuda:0')

        index_log_prob = None
        if batch_query.record_index_prob is not None:
            index_log_prob = torch.zeros((batch_query.total_batch_size), dtype=torch.float, device='cuda:0')

        if 'CHECK_LOGITS_NAN' in os.environ:
            logits_cpu = to_cpu(logits.view(-1))
            if any(torch.isnan(logits_cpu).numpy().tolist()):
                raise Exception(f'logits has nan: {logits_cpu}')
        self.model_ops.sampler.do_sampling(SamplingParams(
            batch_query.max_token_len,
            total_batch_size,
            beam_width,
            batch_query.max_token_len,
            token_ids,
            logits.view(total_batch_size, beam_width, -1),
            finished,
            input_lengths,
            sequence_lengths,
            key_cache,
            value_cache,
            cum_log_probs, # cum_log_probs,
            batch_query.cache_block_indice,
            output_log_probs,
            index_log_prob,
            batch_query.record_index_prob,
        ))
            # print(f"cum_log_probs: {cum_log_probs}")
        output_token_ids = token_ids.permute(1, 0)
        # TODO(wangyin): reshape
        batch_query.record_update_tensors(finished,
                                          [1] * batch_query.total_batch_size,
                                          hidden_states,
                                          logits,
                                          cum_log_probs,
                                          output_token_ids,
                                          output_log_probs,
                                          index_log_prob)

