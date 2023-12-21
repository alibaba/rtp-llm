from typing import Optional
import torch
import logging
from threading import Lock
from maga_transformer.utils.model_weight import LoRAMap
from maga_transformer.ops.ft_op_base import FTOPBase
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.distribute.worker_info import g_parallel_info, g_master_info


class GptOp(FTOPBase):
    def __init__(self, config: GptInitModelParameters):
        super().__init__()
        self.config = config
        self.lock = Lock()

    def _initialize_op(self, force_init: bool=False):
        assert self.weight
        if not force_init and self.ft_op is not None:
            return
        if self.ft_op is not None:
            del self.ft_op
        self.ft_op = torch.classes.FasterTransformer.ParallelGptOp( # type: ignore
            self.config,
            g_parallel_info.tp_size,
            g_parallel_info.pp_size,
            g_master_info.ip,
            g_master_info.gpt_nccl_port,
            self.weight.weights,
            self.weight.int8_weights,
            self.weight.int8_scales)
        
        for id, lora_weight in self.weight.lora_resource.lora_map.weights_map.items():
            self.ft_op.add_lora(id, lora_weight.lora_a_weights, lora_weight.lora_b_weights)

    def forward(self, # type: ignore
                decoder_input: torch.Tensor,
                key_cache: torch.Tensor,
                value_cache: torch.Tensor,
                key_cache_scale: torch.IntTensor,
                value_cache_scale: torch.IntTensor,
                input_lengths: torch.IntTensor,
                sequence_lengths: torch.IntTensor,
                block_index_map: torch.IntTensor,
                attention_mask: Optional[torch.Tensor] = None,
                linear_bias_slopes: Optional[torch.Tensor] = None,
                prefix_lengths: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                count_length: Optional[torch.Tensor] = None,
                max_prefix_length: Optional[torch.Tensor] = None,
                lora_ids: Optional[torch.Tensor] = None):
        """

        # Args.
            input_embeds: Tensor, (decoder_batch + context_token_num, hidden_dim),
                input hidden state to decoder.
            key_cache: Tensor, key cache buffer.
            value_cache: Tensor, value cache buffer.
            input_lengths IntTensor, (decoder_batch + context_decoder_batch,),
                the number of padded tokens.
            sequence_lengths: IntTensor, (decoder_batch),
                the current sequence lengths.
            block_index_map: torch.IntTensor
            attention_mask: torch.Tensor
                attention mask if needed
            linear_bias_slopes Tensor, (num_heads,)
                slopes head of linear position bias (ALiBi) (optional).
            prefix_lengths: IntTensor, (local_batch)
                prefix length of every input
        # Returns
            IntTensor, (decoder_batch + context_decoder_batch, hidden_dim) hidden_states
        """

        self._initialize_op()
        assert self.ft_op is not None
        with self.lock:            
            outputs = self.ft_op.forward(decoder_input,
                                        key_cache,
                                        value_cache,
                                        input_lengths,
                                        sequence_lengths,
                                        block_index_map,
                                        lora_ids,
                                        attention_mask,
                                        position_ids,
                                        linear_bias_slopes,
                                        prefix_lengths,
                                        count_length,
                                        max_prefix_length,
                                        key_cache_scale,
                                        value_cache_scale)
            return outputs