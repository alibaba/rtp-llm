import copy
import torch
import numpy as np
from typing import Any, List, Tuple
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.async_decoder_engine.medusa.medusa_config import MedusaState, MedusaBuffer
from maga_transformer.async_decoder_engine.batch_query import BatchQuery, ModelOutput
from maga_transformer.async_decoder_engine.normal_model_executor import NormalModelExecutor, ModelOps
from maga_transformer.async_decoder_engine.medusa.utils import generate_candidates, evaluate_posterior
from maga_transformer.utils.util import to_cuda

class MedusaModelExecutor(NormalModelExecutor):
    def __init__(self, model_ops: ModelOps, cache_manager, medusa_buffer: MedusaBuffer):
        super().__init__(model_ops, cache_manager)
        assert self.model_ops.model.lm_head is not None, "model lm_head should not be None"
        assert self.model_ops.model.medusa_head is not None, "model medusa_head should not be None"
        assert model_ops.config.medusa_config is not None, "medusa_config shoule not be None"
        self.medusa_config = model_ops.config.medusa_config
        self.medusa_buffer = medusa_buffer

    def _create_batch_query(self, batch_query: BatchQuery) -> BatchQuery:
        medusa_query = BatchQuery(batch_query.gen_num_per_circle, batch_query.nccl_op_)
        medusa_query.context_batch_size = batch_query.total_batch_size
        medusa_query.cache_block_indice = batch_query.cache_block_indice
        medusa_query.context_streams = batch_query.context_streams
        medusa_query.decode_streams = batch_query.decode_streams
        medusa_query.lora_ids = copy.deepcopy(batch_query.lora_ids)

        validate_token_length = self.medusa_buffer.medusa_attn_mask.size(0)
        max_medusa_length = max([q.seq_length for q in batch_query.streams]) + validate_token_length

        output_token_ids = torch.zeros(
            [batch_query.total_batch_size, max_medusa_length],
            dtype=torch.int32
        )

        for i, stream in enumerate(batch_query.streams):
            # means not first into model
            if i < batch_query.generate_batch_size:
                medusa_state: MedusaState = stream.medusa_state
                medusa_query.context_lengths_list.append(validate_token_length)
                medusa_query.reuse_lengths_list.append(stream.seq_length)
                # TODO 这部分应该是能被batch优化掉的 后面改
                output_token_ids[i][stream.seq_length: validate_token_length + stream.seq_length] = medusa_state.tree_candidates[0].cpu()
            else:
                medusa_query.reuse_lengths_list.append(0)
                medusa_query.context_lengths_list.append(stream.input_length)
                output_token_ids[i, :stream.seq_length] = stream.complete_token_ids[0]
            medusa_query.generate_configs.append(None)
        medusa_query.output_token_ids = output_token_ids
        medusa_query.check()
        return medusa_query

    def _create_medusa_state(self, logits: torch.Tensor, medusa_logits: List[torch.Tensor]) -> MedusaState:
        candidates, tree_candidates = generate_candidates(medusa_logits, logits, self.medusa_buffer.tree_indices, self.medusa_buffer.retrieve_indices, self.medusa_config.top_k)
        return MedusaState(candidates, tree_candidates)

    def _tree_validate(self, block_indice: List[int], prev_length: int, generate_config: GenerateConfig,  tree_logits: torch.Tensor, tree_medusa_logits: List[torch.Tensor], medusa_state: MedusaState) -> Tuple[MedusaState, torch.Tensor]:
        logits = tree_logits[0, self.medusa_buffer.retrieve_indices]
        medusa_logits = tree_medusa_logits[:, 0, self.medusa_buffer.retrieve_indices]


        best_candidate, accept_length = evaluate_posterior(
                logits, medusa_state.candidates, generate_config.temperature, self.medusa_config.posterior_threshold, self.medusa_config.posterior_alpha
        )

        logits = logits[None, best_candidate, accept_length : accept_length + 1]
        medusa_logits = medusa_logits[
            :, None, best_candidate, accept_length : accept_length + 1
        ]

        new_tokens = medusa_state.candidates[None, best_candidate, : accept_length + 1]

        candidates, tree_candidates = generate_candidates(
                medusa_logits,
                logits,
                self.medusa_buffer.tree_indices,
                self.medusa_buffer.retrieve_indices,
                self.medusa_config.top_k
        )

        tgt_seq_idxs = list(range(prev_length, prev_length + accept_length + 1))
        src_seq_idxs = self.medusa_buffer.retrieve_indices[best_candidate, : accept_length + 1] + prev_length
        self.cache_manager_.copy_kvcache_from_seq_idxs(block_indice, src_seq_idxs.tolist(), tgt_seq_idxs)

        return MedusaState(candidates, tree_candidates), new_tokens.squeeze_(0)

    def _tree_sample(self, batch_query: BatchQuery, hidden_states: torch.Tensor) -> Tuple[List[Any], List[Any], List[Any]]:
        assert self.model_ops.model.lm_head is not None
        assert self.model_ops.model.medusa_head is not None
        if self.model_ops.model.post_decoder_layernorm is not None:
            hidden_states = self.model_ops.model.post_decoder_layernorm(hidden_states)
        medusa_states_list: List[MedusaState] = []
        accept_tokens_list: List[torch.Tensor] = []
        finished_list: List[bool] = []
        logits = self.model_ops.model.lm_head(hidden_states)
        medusa_logits = self.model_ops.model.medusa_head(hidden_states)
        bias = 0
        for i in range(batch_query.total_batch_size):
            if batch_query.reuse_lengths_list[i] == 0:
                medusa_states_list.append(
                    self._create_medusa_state(logits[bias + batch_query.context_lengths_list[i] - 1: bias + batch_query.context_lengths_list[i]].unsqueeze(1),
                                              medusa_logits[:, bias + batch_query.context_lengths_list[i] - 1: bias + batch_query.context_lengths_list[i], :].unsqueeze(1))
                )
                accept_tokens_list.append(torch.empty([0]))
                finished_list.append(False)
            else:
                medusa_state, accept_tokens = self._tree_validate(batch_query.streams[i].block_indice[0],
                                                                  batch_query.reuse_lengths_list[i],
                                                                  batch_query.streams[i].generate_config,
                                                                  logits[bias: bias + batch_query.context_lengths_list[i]].unsqueeze(0),
                                                                  medusa_logits[:, bias: bias + batch_query.context_lengths_list[i], :].unsqueeze(1),
                                                                  batch_query.streams[i].medusa_state)
                medusa_states_list.append(medusa_state)
                accept_tokens_list.append(accept_tokens)
                finished_list.append(self.model_ops.config.special_tokens.eos_token_id in accept_tokens)
            bias += batch_query.context_lengths_list[i]

        return finished_list, accept_tokens_list, medusa_states_list

    def _create_context_attention_mask(self, batch_query: BatchQuery):
        max_context_length = max(batch_query.context_lengths_list)
        total_attention_mask = torch.zeros([batch_query.total_batch_size, max_context_length, max_context_length], dtype=torch.bool)
        for i in range(batch_query.total_batch_size):
            if batch_query.reuse_lengths_list[i] == 0:
                attention_mask = torch.ones(
                    (batch_query.context_lengths_list[i], batch_query.context_lengths_list[i]), dtype=torch.bool).tril()
            else:
                attention_mask = self.medusa_buffer.medusa_attn_mask
            total_attention_mask[i][:attention_mask.shape[0], :attention_mask.shape[1]] = attention_mask

        total_attention_mask = self.append_reuse_mask(
            total_attention_mask, batch_query.context_query_context_lengths_list, batch_query.context_query_reuse_lengths_list)
        return total_attention_mask.to(self.model_ops.model.dtype).cuda()

    def _create_position_ids_for_rotary(self, batch_query: BatchQuery):
        position_ids: List[int] = []
        for i in range(batch_query.total_batch_size):
            if batch_query.reuse_lengths_list[i] == 0:
                position_ids.extend(list(range(batch_query.context_lengths_list[i])))
            else:
                position_ids.extend([x + batch_query.reuse_lengths_list[i] for x in self.medusa_buffer.medusa_position_ids])
        return to_cuda(torch.IntTensor(position_ids))

    def process(self, batch_query: BatchQuery) -> None:
        medusa_query = self._create_batch_query(batch_query)
        all_hidden_states = self._process(medusa_query)
        finished_list, accept_tokens_list, medusa_states_list = self._tree_sample(medusa_query, all_hidden_states)
        update_lens = [len(x) for x in accept_tokens_list]
        for i, output_token_id in enumerate(batch_query.output_token_ids):
            output_token_id[batch_query.max_token_len : batch_query.max_token_len+update_lens[i]] = accept_tokens_list[i]
        batch_query.update_output(ModelOutput(
            finished=torch.tensor(finished_list, dtype=torch.bool),
            update_length=update_lens,
            update_token_ids=batch_query.output_token_ids,
            medusa_states=medusa_states_list))
