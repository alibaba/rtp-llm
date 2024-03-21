import sys
from typing import Optional, List, Union, Any
import torch
import random
from maga_transformer.ops.ft_op_base import FTOPBase
from maga_transformer.config.exceptions import ExceptionType, FtRuntimeException
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.utils.word_util import to_word_list_format, get_list_dim
from maga_transformer.distribute.worker_info import g_parallel_info, g_master_info

FloatLike=Optional[Union[List[float], float]]
IntLike = Optional[Union[List[int], int]]
Long = Optional[Union[List[int], int]]


'''
top_k: IntTensor, (batch_size,) top-k sampling. The number of most probable
    tokens to keep for sampling per sentence in a batcch.
top_p: FloatTensor, (batch_size,), top-p sampling. The cumulative probability
    of to filter the set of most probable tokens.
top_p_decay: FloatTensor, (batch_size,)
    The decay of top-p value for top_p sampling.
top_p_min: FloatTensor, (batch_size,)
    The minimum top p values in top-p decaying.
top_p_reset_ids: IntTensor, (batch_size,)
    reset ids for resetting top_p values for top p sampling
temperature: FloatTensor, (batch_size,),
    The temperature value for smoothing the logit distribution.
repetition_penalty: FloatTensor, (batch_size,),
    The repetition penalty.
presence_penalty: FloatTensor, (batch_size,),
    The presence penalty, which is exclusive with repetition_penalty.
    Only one of repetition and presence penalties is allowed.
min_length: IntTensor, (batch_size,),
    Minimum length for each sentences. EOS is masked if length is below min.
random_seed: LongTensor [1] or [batch_size]
    Random seed to initialize the random table in sampling.
len_penalty: FloatTensor, (batch_size,)
    The exponent of the length penalty of beam scores.
beam_search_diversity_rate: FloatTensor, (batch_size,),
    The diversity rate of beam search.
stop_words_list: IntTensor, (batch_size, 2, stop_words_length)
    When FT generates words in this list, it will stop the generation. An extension of stop id.
bad_words_list IntTensor, (batch_size, 2, bad_words_length)
    The words in the list will never be sampled.
sequence_limit_lengths: IntTensor, (batch_size,), The maximum length of a generated sequence.
'''
class SampleConfig(object):
    def __init__(self, batch_size: int, generate_config: GenerateConfig):
        self.batch_size = batch_size
        self.top_k = self.convert_to_tensor(generate_config.top_k, torch.IntTensor)
        self.top_p = self.convert_to_tensor(generate_config.top_p, torch.FloatTensor)
        self.temperature = self.convert_to_tensor(generate_config.temperature, torch.FloatTensor)
        self.repetition_penalty = self.convert_to_tensor(generate_config.repetition_penalty, torch.FloatTensor)
        # self.presence_penalty = self.convert_to_tensor(generate_config.presence_penalty, torch.FloatTensor)
        self.presence_penalty = None
        self.min_length = self.convert_to_tensor(generate_config.min_new_tokens, torch.IntTensor)
        # self.len_penalty = self.convert_to_tensor(generate_config.len_penalty, torch.FloatTensor)
        self.len_penalty = None
        # self.beam_search_diversity_rate = self.convert_to_tensor(generate_config.beam_search_diversity_rate, torch.FloatTensor)
        self.beam_search_diversity_rate = None
        # set default random seed, do not set if world_size > 1 since it will inference sample result
        if generate_config.random_seed == None and g_parallel_info.world_size == 1:
            self.random_seed = self.convert_to_tensor([random.randint(0, sys.maxsize) for i in range(batch_size)], torch.LongTensor)
        else:
            self.random_seed = self.convert_to_tensor(generate_config.random_seed, torch.LongTensor)
        self.top_p_decay = self.convert_to_tensor(generate_config.top_p_decay, torch.FloatTensor, 'cuda')
        self.top_p_min = self.convert_to_tensor(generate_config.top_p_min, torch.FloatTensor, 'cuda')
        self.top_p_reset_ids = self.convert_to_tensor(generate_config.top_p_reset_ids, torch.IntTensor, 'cuda')
        # self.stop_words_list = self.convert_stop_list(batch_size, generate_config.stop_words_list)
        # self.bad_words_list = self.convert_stop_list(batch_size, generate_config.bad_words_list)

    def convert_to_tensor(self, origin: Any, dest_type: Any, device: str = 'cpu') -> Optional[torch.Tensor]:
        if origin is None:
            return None
        if isinstance(origin, torch.Tensor):
            return origin.to(device)
        elif isinstance(origin, list):
            if (all(i != None for i in origin)):
                return dest_type(origin).to(device)
            return None
        else:
            return dest_type([origin]).to(device)

    '''
    possiable input_shape:
    1. [word_list_len, word_list] -> expand to [batch_size, word_list_len, word_list]
    2. [batch_size, word_list_len, word_list]
    '''
    def convert_stop_list(self, batch_size: int, word_list: Optional[List[int]]) -> Optional[torch.Tensor]:
        if not word_list:
            return None
        dim = get_list_dim(word_list)
        if dim < 2:
            raise FtRuntimeException(ExceptionType.ERROR_STOP_LIST_FORMAT, "stop_word_list should at least 2-dim list")
        if dim == 2:
            word_list = [word_list] * batch_size
        elif dim == 3:
            if len(word_list) != batch_size:
                raise FtRuntimeException(ExceptionType.ERROR_STOP_LIST_FORMAT, "stop_list first dim shoule equal to batch_size")
        else:
            raise FtRuntimeException(ExceptionType.ERROR_STOP_LIST_FORMAT, "stop_word_list should at most 3-dim list")

        stop_list_tensor = torch.IntTensor(to_word_list_format(word_list))
        if stop_list_tensor.size(0) != batch_size and stop_list_tensor.size(1) == 2:
            raise FtRuntimeException(ExceptionType.ERROR_STOP_LIST_FORMAT, "stop list tensor shape should be [batch_size, 2, seq_len]")
        return stop_list_tensor

class DynamicDecodeOp(FTOPBase):
    def __init__(self,
                 vocab_size: int,
                 vocab_size_padded: int=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.vocab_size_padded = vocab_size_padded
        self.ft_op = torch.classes.FasterTransformer.DynamicDecodeOp( # type: ignore
            vocab_size,
            vocab_size_padded,
            g_parallel_info.tp_size,
            g_parallel_info.pp_size,
            torch.float,
            g_master_info.ip,
            g_master_info.dynamic_decoder_nccl_port)

    def _initialize_op(self, force_init: bool=False):
        pass

    def set_up(self,
               batch_size: int,
               beam_width: int,
               config: SampleConfig):
        self.ft_op.setup(batch_size, # type: ignore
                         beam_width,
                         config.top_k,
                         config.top_p,
                         config.temperature,
                         config.repetition_penalty,
                         config.presence_penalty,
                         config.min_length,
                         config.len_penalty,
                         config.beam_search_diversity_rate,
                         config.random_seed,
                         config.top_p_decay,
                         config.top_p_min,
                         config.top_p_reset_ids)

    def broadcast_from_last_pipeline(self, tensors: List[torch.Tensor]):
        self.ft_op.broadcast_from_last_pipeline(tensors) # type: ignore

    def forward(self, # type: ignore
                logits: torch.Tensor,
                step: int,
                max_input_length: int,
                ite: int,
                local_batch_size: int,
                eos_token_ids: torch.Tensor,
                finished: torch.Tensor,
                sequence_lengths: Optional[torch.Tensor],
                output_token_ids: torch.Tensor,
                embedding_bias:Optional[torch.Tensor] = None,
                input_lengths: Optional[torch.Tensor] = None,
                sequence_limit_lengths: Optional[torch.Tensor] = None,
                cum_log_probs: Optional[torch.Tensor] = None,
                output_log_probs: Optional[torch.Tensor] = None,
                index_log_probs: Optional[torch.Tensor] = None,
                output_logit_index: Optional[torch.Tensor] = None,
                config: SampleConfig = SampleConfig(0, GenerateConfig())):
        # outputs: output hidden states
        assert self.ft_op is not None
        assert output_token_ids is not None
        assert finished is not None

        should_stop = self.ft_op.forward(
            logits,
            step,
            max_input_length,
            ite,
            local_batch_size,
            eos_token_ids,
            config.top_k,
            config.top_p,
            config.temperature,
            config.repetition_penalty,
            config.presence_penalty,
            config.min_length,
            config.len_penalty,
            config.beam_search_diversity_rate,
            config.top_p_decay,
            config.top_p_min,
            config.top_p_reset_ids,
            embedding_bias,
            input_lengths,
            sequence_limit_lengths,
            None, # config.stop_words_list,
            None, # config.bad_words_list,
            None, # src_cache_indirection,
            output_token_ids,
            finished,
            sequence_lengths,
            cum_log_probs,
            output_log_probs, # output_log_probs,
            index_log_probs,
            output_logit_index,
            None, # parent_id,
            None, # tgt_cache_indirectio,
            None, # normed_score,
            None, # min_normed_score,
            None, # num_beam,
        )
        return should_stop
