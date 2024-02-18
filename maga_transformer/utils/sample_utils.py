import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, NamedTuple

from transformers.generation.utils import GenerationMixin
from transformers.generation.configuration_utils import GenerationConfig as HfGenerateConfig
from transformers.generation.stopping_criteria import StoppingCriteriaList, StoppingCriteria
from transformers.generation.logits_process import LogitsProcessorList

from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.ops.comm.dynamic_decode_op import DynamicDecodeOp, SampleConfig

FT_DEFAULT_MAX_NEW_TOKENS = 2048
FT_DEFAULT_MAX_LENGTH = 2048

class SamplerSetupParams(NamedTuple):
    max_batch_size: int # sampler allocates buffers according to this size.
    eos_token_id: int
    max_input_length: int
    input_tensor: Optional[torch.Tensor]

class SamplingParams(NamedTuple):
    step: int
    batch_size: int
    beam_width: int
    max_input_length: int
    output_token_ids: torch.Tensor
    logits: torch.Tensor
    finished: torch.Tensor
    input_lengths: torch.Tensor
    sequence_lengths: torch.Tensor
    key_cache: Optional[torch.Tensor]
    value_cache: Optional[torch.Tensor]
    cum_log_probs: Optional[torch.Tensor]
    block_index_map: Optional[torch.Tensor]
    output_log_probs: Optional[torch.Tensor] = None
    index_log_probs: Optional[torch.Tensor] = None
    output_logit_index: Optional[torch.Tensor] = None

class BaseSampler(object):
    config: GenerateConfig
    def setup(self, params: SamplerSetupParams) -> None:
        raise NotImplementedError()

    def do_sampling(self, params: SamplingParams) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

class HuggingfaceSampler(BaseSampler, GenerationMixin):
    def __init__(self, generate_config: GenerateConfig,
                 logits_processor: Optional[LogitsProcessorList] = None,
                 stopping_criteria: Optional[StoppingCriteriaList] = None
    ):
        self.config = HfGenerateConfig()
        self.config.update(**generate_config.model_dump())
        if self.config.max_length is None:
            self.config.max_length = FT_DEFAULT_MAX_LENGTH
        self.logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        self.stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        self.logits_warper = self._get_logits_warper(self.config) if self.config.do_sample else None

    def setup(self, params: SamplerSetupParams) -> None:
        self.eos_token_id = params.eos_token_id
        self.logits_processor = self._get_logits_processor(
            self.config, params.max_input_length, params.input_tensor, None, self.logits_processor) # type: ignore

    def do_sampling(self, params: SamplingParams) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = params.batch_size
        input_ids = params.output_token_ids.view(-1, batch_size)[:params.step , ...].permute(1, 0)
        logits = params.logits.view(batch_size, -1)
        finished = params.finished

        next_tokens_scores = self.logits_processor(input_ids.to(torch.int64), logits)
        if self.config.do_sample == False:
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        else:
            next_tokens_scores = self.logits_warper(input_ids, logits)
            probs = torch.nn.functional.softmax(next_tokens_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        should_stop = self.stopping_criteria(input_ids, next_tokens_scores)
        finished = ((next_tokens == self.eos_token_id) | should_stop)
        for i in range(params.output_token_ids.shape[1]):
            params.output_token_ids[params.step][i] = next_tokens[i]
            params.finished[i] |= finished[i]
            params.sequence_lengths[i] += 1

class FtSampler(BaseSampler):
    def __init__(self, config: GenerateConfig, dynamic_decoder: DynamicDecodeOp):
        self.config = config
        self.dynamic_decoder = dynamic_decoder

    def setup(self, params: SamplerSetupParams) -> None:
        sample_config = SampleConfig(params.max_batch_size, self.config)
        self.dynamic_decoder.set_up(params.max_batch_size, 1, sample_config)
        self.sample_config = sample_config
        self.eos_token_id = params.eos_token_id
        self.eos_token_ids = torch.tensor(
            [params.eos_token_id] * params.max_batch_size, dtype=torch.int32, device='cuda'
        )

    def do_sampling(self, params: SamplingParams) -> None:
        batch_size = params.batch_size
        beam_width = params.beam_width
        self.dynamic_decoder.forward(
            params.logits.view(batch_size, beam_width, -1),
            params.step,
            params.max_input_length,
            0, # ite
            batch_size,
            self.eos_token_ids,
            params.finished,
            params.sequence_lengths,
            params.output_token_ids.view(-1, batch_size, beam_width),
            None, # embdding_bias
            params.input_lengths,
            None, # sequence_limit_len
            params.cum_log_probs,
            params.output_log_probs,
            params.index_log_probs,
            params.output_logit_index,
            self.sample_config
        ) # type: ignore

class BeamSearchSampler(FtSampler):
    def __init__(self, config: GenerateConfig, dynamic_decoder: DynamicDecodeOp) -> None:
        super().__init__(config, dynamic_decoder)

    def setup(self, params: SamplerSetupParams) -> None:
        super().setup(params)
        self.beam_width = self.config.num_beams
        self.batch_size = params.max_batch_size
        self.scores = torch.zeros((self.batch_size, self.beam_width),
                                  dtype=torch.float32, device='cuda')
        self.first_token_generated = False

    def do_beam_search(self,
                       step: int,
                       batch_size: int,
                       logits: torch.Tensor,
                       eos_token_id: int,
                       finished: torch.Tensor,
                       sequence_lengths: torch.Tensor,
                       output_token_ids: torch.Tensor,
                       key_cache: torch.Tensor,
                       value_cache: torch.Tensor,
                       cum_log_probs: torch.Tensor,
                       block_index_map: Optional[torch.Tensor] = None
    ) -> None:
        # TODO(wangyin): sequence_lengths is not required here anymore. try move it somewhere else.
        if block_index_map == None:
            sequence_lengths += 1

        vocab_size = logits.shape[-1]
        log_logits = F.log_softmax(logits, dim=2)
        if (block_index_map != None):
            # NOTE: in async mode, default scores is shape for max batch size.
            # the actual scores are stored externally via cum_log_probs
            self.scores = cum_log_probs.view([batch_size, self.beam_width])

        new_scores = log_logits + self.scores.unsqueeze(-1)

        sorted_scores_list = []
        sorted_indices_list = []
        for i in range(batch_size):
            if (cum_log_probs[i * self.beam_width].item() == 0):
                query_scores, query_indices = torch.sort(new_scores[i, 0, :], descending=True)
                sorted_scores_list.append(query_scores)
                sorted_indices_list.append(query_indices)
            else:
                query_scores, query_indices = torch.sort(new_scores[i, : , :].view(-1), descending=True)
                sorted_scores_list.append(query_scores)
                sorted_indices_list.append(query_indices)

        min_token_length = min([scores_tensor.shape[0] for scores_tensor in sorted_scores_list])
        sorted_scores_list = [scores_tensor[:min_token_length] for scores_tensor in sorted_scores_list]
        sorted_indices_list = [indices_tensor[:min_token_length] for indices_tensor in sorted_indices_list]
        sorted_scores = torch.stack(sorted_scores_list, dim=0)
        indices = torch.stack(sorted_indices_list, dim=0)

        beam_search_space = 2 * self.beam_width * self.beam_width
        best_beam_ids = (indices[:, : beam_search_space] / vocab_size).trunc().long()
        best_words = indices[:, : beam_search_space] % vocab_size
        best_scores = sorted_scores[:, : beam_search_space]
        output_tokens = output_token_ids.permute(1, 2, 0)

        for batch_idx in range(batch_size):
            if finished.view([batch_size, self.beam_width])[batch_idx, 0]:
                continue
            batch_beams = list(zip(best_words[batch_idx], best_scores[batch_idx], best_beam_ids[batch_idx]))
            batch_beams = batch_beams[:self.beam_width]
            best_beams = output_tokens[batch_idx].new([item[2] for item in batch_beams])
            tokens = output_tokens[batch_idx, best_beams, :]
            output_token_ids[:step, batch_idx] = tokens[:, :step].permute(1, 0)
            output_token_ids[step, batch_idx] = tokens.new([item[0] for item in batch_beams])

            self.scores[batch_idx] = self.scores.new([item[1] for item in batch_beams])
            cum_log_probs.view(batch_size, self.beam_width)[batch_idx] = self.scores[batch_idx]

            if (block_index_map != None):
                # indicates async mode.
                # key_cache: [layers, reserved_blocks, local_head_num, size_per_head // x, block_num, x]
                # value_cache: [layers, reserved_blocks, local_head_num, block_num, size_per_head]
                src_blocks = block_index_map.view([batch_size, self.beam_width, -1])[batch_idx, best_beams.cpu()]
                tgt_blocks = block_index_map.view([batch_size, self.beam_width, -1])[batch_idx]
                key_cache[:, tgt_blocks] = key_cache[:, src_blocks]
                value_cache[:, tgt_blocks] = value_cache[:, src_blocks]
            else:
                # key_cache: [layers, batch, local_head_num, size_per_head // x, memory_max_len, x]
                # value_cache: [layers, batch, local_head_num, memory_max_len, size_per_head]
                key_cache.view([key_cache.shape[0], batch_size, self.beam_width, -1])[:, batch_idx] = \
                    key_cache.view([key_cache.shape[0], batch_size, self.beam_width, -1])[:, batch_idx, best_beams]
                value_cache.view([value_cache.shape[0], batch_size, self.beam_width, -1])[:, batch_idx] = \
                    value_cache.view([value_cache.shape[0], batch_size, self.beam_width, -1])[:, batch_idx, best_beams]

            stop_words_list = self.config.stop_words_list + [[eos_token_id]]
            for stop_word in stop_words_list:
                if output_token_ids[step + 1 - len(stop_word): step + 1, batch_idx, 0].tolist() == stop_word:
                    finished.view([batch_size, self.beam_width])[batch_idx] = True

    def do_sampling(self, params: SamplingParams) -> None:
        assert (params.cum_log_probs != None)
        assert (params.key_cache != None)
        assert (params.value_cache != None)
        batch_size = params.batch_size
        beam_width = params.beam_width
        self.do_beam_search(
            params.step,
            params.batch_size,
            params.logits.view(batch_size, beam_width, -1),
            self.eos_token_id,
            params.finished,
            params.sequence_lengths,
            params.output_token_ids.view(-1, batch_size, beam_width),
            params.key_cache,
            params.value_cache,
            params.cum_log_probs,
            params.block_index_map
        )

class StopWordIdsCriteria(StoppingCriteria):
    def __init__(self, stop_word_ids: List[int]):
        self.stop_word_ids = stop_word_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return torch.all(input_ids[:,-2:].cpu() == torch.Tensor(self.stop_word_ids)).item() # type: ignore
    

    
