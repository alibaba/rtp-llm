from typing import Any, Optional
from threading import Lock
import torch
from maga_transformer.ops.ft_op_base import FTOPBase
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.distribute.worker_info import g_parallel_info, g_master_info

class GptDecoder(FTOPBase):
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
        self.ft_op = torch.classes.FasterTransformer.ParallelGptDecoderOp( # type: ignore
            self.config.gpt_init_params,
            g_parallel_info.tp_size,
            g_parallel_info.pp_size,
            g_master_info.ip,
            g_master_info.decoder_nccl_port,
            self.weight.weights,
            self.weight.int8_weights,
            self.weight.int8_scales)
        
        for id, lora_weight in self.weight.lora_resource.lora_map.weights_map.items():
            self.ft_op.add_lora(id, lora_weight.lora_a_weights, lora_weight.lora_b_weights)
            
    def forward(self, # type: ignore
                max_input_length: int,
                step: int,
                ite: int,
                input_embeds: torch.Tensor,
                sequence_lengths: torch.IntTensor,
                key_cache: torch.Tensor,
                value_cache: torch.Tensor,
                finished: torch.BoolTensor,
                input_lengths: torch.IntTensor,
                masked_tokens: Optional[torch.BoolTensor] = None,
                cache_indirection: Optional[torch.IntTensor] = None,
                linear_bias_slopes: Optional[torch.Tensor] = None,
                prefix_prompt: Optional[torch.Tensor] = None ,
                prefix_lengths: Optional[torch.Tensor] = None,
                max_prefix_lengths: Optional[torch.Tensor] = None,
                block_index_map: Optional[torch.Tensor] = None,
                lora_ids: Optional[torch.Tensor] = None,
                **kwargs: Any):
        """

        # Args.
            max_input_length: int, maximum input context length.
            step: int, the current step index.
            ite: int, local batch iteration.
            input_embeds: Tensor, (local_batch * beam, hidden_dim),
                input hidden state to decoder.
            sequence_lengths: IntTensor, (local_batch * beam,),
                the current sequence lengths.
            key_cache: Tensor, key cache buffer.
            value_cache: Tensor, value cache buffer.
            finished: BoolTensor, (local_batch * beam,),
                whether to finish sentence generation.
            input_lengths IntTensor, (local_batch * beam,),
                the number of padded tokens.
            masked_tokens: BoolTensor, (local_batch * beam, memory_length),
                a mask tensor that indicates padded tokens.
            cache_indirection: IntTensor, (local_batch * beam,),
                cache of beam positions if needed if beam > 1.
            linear_bias_slopes Tensor, (num_heads,)
                slopes head of linear position bias (ALiBi) (optional).
        # Returns
            IntTensor, (batch * beam,) output token ids.
        """

        self._initialize_op()
        assert self.ft_op is not None
        with self.lock:
            outputs = self.ft_op.forward(max_input_length,
                                        step,
                                        ite,
                                        input_embeds,
                                        sequence_lengths,
                                        finished,
                                        input_lengths,
                                        key_cache,
                                        value_cache,
                                        lora_ids,
                                        masked_tokens,
                                        cache_indirection,
                                        linear_bias_slopes,
                                        prefix_lengths,
                                        max_prefix_lengths,
                                        block_index_map)
            return outputs[0]