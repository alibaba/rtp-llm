import torch
import random
from typing import List, Tuple, Any
from maga_transformer.utils.util import to_cpu
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.async_decoder_engine.batch_query import BatchQuery, ModelOutput
from maga_transformer.async_decoder_engine.normal_model_executor import NormalModelExecutor, ModelOps, ExecutorBase

class SpModelExecutor(ExecutorBase):
    def __init__(self, validate_executor: NormalModelExecutor, sp_executor: NormalModelExecutor, gen_num: int):
        self.validate_executor = validate_executor
        self.sp_executor = sp_executor
        self.gen_num = gen_num

    @property
    def base_model_ops(self):
        return self.validate_executor.base_model_ops

    def process(self, batch_query: BatchQuery) -> None:
        with torch.cuda.nvtx.range("speculative gen"):
            cum_probs, output_tokens = self._speculative_gen(batch_query)
        with torch.cuda.nvtx.range("speculative validate"):
            finished, hidden_states, logits, dynamic_decoder_tokens, update_length = self._speculative_validate(batch_query, cum_probs, output_tokens)
            if g_parallel_info.tp_rank > 0:
                return
            batch_query.output_token_ids[:,batch_query.max_token_len : batch_query.max_token_len+self.gen_num] = dynamic_decoder_tokens
        batch_query.update_output(ModelOutput(
            finished=finished,
            update_length=update_length,
            hidden_states=hidden_states,
            logits=logits,
            cum_log_probs=torch.zeros([batch_query.total_batch_size]),
            update_token_ids= batch_query.output_token_ids))

    def _speculative_update(self, batch_query: BatchQuery, new_tokens: torch.Tensor):
        if batch_query.context_batch_size > 0:
            batch_query.seq_lengths_list += batch_query.context_lengths_list[batch_query.generate_batch_size: ]
            batch_query.generate_batch_size += batch_query.context_batch_size
            batch_query.context_batch_size = 0
            batch_query.reuse_lengths_list = [0] * batch_query.total_batch_size
            batch_query.calculate_loss = []
        for i, token in enumerate(new_tokens):
            batch_query.output_token_ids[i][batch_query.seq_lengths_list[i]] = token
        batch_query.seq_lengths_list = [x + 1 for x in batch_query.seq_lengths_list]

    # 这里本来应该用_post_process，但是由于性能和逻辑现在都不对，所以先简单实现一个
    def _speculative_sampler(self, model_ops: ModelOps, batch_query: BatchQuery, hidden_states: torch.Tensor):
        if model_ops.model.post_decoder_layernorm is not None:
            hidden_states = model_ops.model.post_decoder_layernorm(hidden_states)
        assert model_ops.model.lm_head is not None
        logits = model_ops.model.lm_head(hidden_states).float()

        finished = torch.zeros((batch_query.total_batch_size)).bool()
        soft = torch.nn.functional.softmax(logits, dim=-1)
        tokens = torch.multinomial(soft, num_samples=1).squeeze(1)
        # tokens = torch.argmax(soft, dim=-1)
        return to_cpu(finished), to_cpu(tokens), to_cpu(soft)

    def _speculative_gen(self, batch_query: BatchQuery) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # 跑context_decoder，初始化kvcache
        log_prob_list: List[torch.Tensor] = []
        output_token_list: List[torch.Tensor] = []
        gen_batch_query = batch_query.deepcopy()
        # init context decoder
        for _ in range(self.gen_num):
            gen_batch_query.tp_sync()
            self.sp_executor.process(gen_batch_query)
            if g_parallel_info.tp_rank == 0:
                assert gen_batch_query.model_output.output_log_probs is not None
                assert gen_batch_query.model_output.update_token_ids is not None
                log_prob_list.append(gen_batch_query.model_output.output_log_probs)
                output_tokens = gen_batch_query.model_output.update_token_ids
                new_tokens = output_tokens[:, gen_batch_query.max_token_len]
                output_token_list.append(new_tokens)
                self._speculative_update(gen_batch_query, new_tokens)
            else:
                log_prob_list.append(torch.empty(1,0))
                output_token_list.append(torch.empty(1,0))
        return log_prob_list, output_token_list

    # 如果output_token_list len = N， 那么只有 N-1 个token会被用于验证
    def _create_validate_query(self, batch_query: BatchQuery, output_token_list: List[torch.Tensor]):
        query = batch_query.deepcopy()
        if g_parallel_info.tp_rank == 0:            
            validate_token_num = len(output_token_list)
            for i in range(query.total_batch_size):
                if i < query.generate_batch_size:
                    query.reuse_lengths_list[i] = query.seq_lengths_list[i] - 1
                    query.context_lengths_list[i] = validate_token_num
                else:
                    query.context_lengths_list[i] += validate_token_num - 1

                index_base = query.reuse_lengths_list[i] + query.context_lengths_list[i] - validate_token_num + 1
                for j in range(0, validate_token_num - 1):
                    query.output_token_ids[i][index_base + j] = output_token_list[j][i]
            query.context_batch_size += query.generate_batch_size
            query.generate_batch_size = 0
            query.seq_lengths_list = []
            query.calculate_loss = [0] * query.context_batch_size            
        query.tp_sync()
        return query

    def _unpack_validate_list(self, validate_query: BatchQuery, validate_len: int, hidden_states: torch.Tensor) -> torch.Tensor:
        index_lst: List[int] = []
        for i in range(0, validate_len):
            offset = 0
            for idx in range(0, validate_query.total_batch_size):
                index_lst.append(offset + validate_query.context_lengths_list[idx] - validate_len + i)
                offset = offset + validate_query.context_lengths_list[idx]
        return hidden_states.index_select(0, torch.tensor(index_lst, device="cuda:0"))

    def _creata_fake_sample_query(self, batch_query: BatchQuery, validate_len: int, output_token_list: List[torch.Tensor]):
        fake_sample_query = batch_query.deepcopy()
        fake_sample_query.context_streams *= validate_len
        fake_sample_query.decode_streams *= validate_len
        fake_sample_query.context_batch_size *= validate_len
        fake_sample_query.generate_batch_size *= validate_len
        fake_sample_query.seq_lengths_list *= validate_len
        fake_sample_query.context_lengths_list *= validate_len
        fake_sample_query.output_token_ids = fake_sample_query.output_token_ids.repeat(validate_len, 1)
        fake_sample_query.merge_generate_config = GenerateConfig.merge_generate_config(fake_sample_query.generate_configs * validate_len)
        fake_sample_query.record_index_prob = torch.cat(output_token_list).cuda()

        return fake_sample_query

    def _speculative_accept(self, batch_query: BatchQuery, token_probs: List[torch.Tensor], output_token_list: List[torch.Tensor], validate_hiddens: torch.Tensor):
        def accept(sp_prob: float, model_prob: float):
            rand = random.random()
            if sp_prob <= model_prob:
                return True
            else:
                return rand < model_prob / (sp_prob + 0.00001)

        # shape: [batch, gen_num]        
        validate_logits = self.validate_executor._post_transformer_nn(validate_hiddens)
        if g_parallel_info.tp_rank > 0:
            return None, None, None, None, None
        fake_batch_query = self._creata_fake_sample_query(batch_query, self.gen_num, output_token_list)        
        self.validate_executor._post_process(fake_batch_query, validate_logits, validate_hiddens)
        next_tokens = fake_batch_query.slice_output_token(0, fake_batch_query.total_batch_size + 1, 1)
        index_probs = fake_batch_query.model_output.output_index_prob

        res: List[List[int]] = [[] for i in range(batch_query.total_batch_size)]
        end = [False] * batch_query.total_batch_size
        finished = torch.zeros((batch_query.total_batch_size), device="cuda:0").bool()

        tokens = next_tokens.cpu().view(self.gen_num, batch_query.total_batch_size)
        logits = index_probs.view(self.gen_num, batch_query.total_batch_size, -1)
        for i in range(self.gen_num):
            for j in range(0, batch_query.total_batch_size):
                if not end[j]:
                    if accept(float(torch.exp(token_probs[i][j])), float(torch.exp(logits[i][j]))):
                        res[j].append(int(output_token_list[i][j]))
                    else:
                        res[j].append(int(tokens[i][j]))
                        end[j] = True
                    if res[j][-1] == self.validate_executor.model_ops.config.special_tokens.eos_token_id:
                        end[j] = True
                        finished[j] = True
            if all(end):
                break
        # from List[List] -> List[tensor], where tensor size = [1, x]
        gen_result: torch.Tensor = torch.ones([batch_query.total_batch_size, self.gen_num], dtype=torch.int, device='cpu')
        gen_length: List[int] = []
        for i, single_res in enumerate(res):
            gen_result[i][:len(single_res)] = torch.tensor(single_res, dtype=torch.int, device='cpu')
            gen_length.append(len(single_res))        
        return finished, torch.empty([batch_query.total_batch_size, 0]), torch.empty([batch_query.total_batch_size, 0]), gen_result, gen_length

    def _speculative_validate(self, batch_query: BatchQuery, cum_probs: List[torch.Tensor], output_token_list: List[torch.Tensor]):
        validate_batch_query = self._create_validate_query(batch_query, output_token_list)
        # shape = [batch_size, validate_len]
        hidden_states = self.validate_executor._process(validate_batch_query)
        validate_hiddens = self._unpack_validate_list(validate_batch_query, len(output_token_list), hidden_states)
        return self._speculative_accept(batch_query, cum_probs, output_token_list, validate_hiddens)            