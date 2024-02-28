from typing import Any, Optional
from threading import Lock
import torch
from maga_transformer.ops.ft_op_base import FTOPBase
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.distribute.worker_info import g_parallel_info, g_master_info

class GptContextDecoder(FTOPBase):
    def __init__(self, config: GptInitModelParameters):
        super().__init__()
        self.config = config
        self.remove_padding = True
        self.ft_op = None
        self.weight = None
        self.lock = Lock()

    def _initialize_op(self, force_init: bool=False):
        assert self.weight
        if not force_init and self.ft_op is not None:
            return
        if self.ft_op is not None:
            del self.ft_op

        self.ft_op = torch.classes.FasterTransformer.ParallelGptContextDecoderOp( # type: ignore
            self.config.gpt_init_params,
            g_parallel_info.tp_size,
            g_parallel_info.pp_size,
            g_master_info.ip,
            g_master_info.context_decoder_nccl_port,
            self.weight.weights,
            self.remove_padding)
        
        for id, lora_weight in self.weight.lora_resource.lora_map.weights_map.items():
            self.ft_op.add_lora(id, lora_weight.lora_a_weights, lora_weight.lora_b_weights)

         
    def forward(self, # type: ignore
                input_embeds: torch.Tensor,
                attention_mask: torch.Tensor,
                input_lengths: torch.IntTensor,
                memory_length: Optional[int] = None,
                compact_index: Optional[torch.IntTensor] = None,
                batch_to_compact_index: Optional[torch.IntTensor] = None,
                linear_bias_slopes: Optional[torch.Tensor] = None,
                prefix_prompt: Optional[torch.Tensor] = None,
                prefix_lengths: Optional[torch.Tensor] = None,
                key_cache: Optional[torch.Tensor] = None,
                value_cache: Optional[torch.Tensor] = None,
                block_index_map: Optional[torch.Tensor] = None,
                lora_ids: Optional[torch.Tensor] = None,
                **kwargs: Any):
        """

        # Args.
            input_embeds: Tensor, (batch * beam, max_input_length, hidden_dim),
                input hidden states.
            attention_mask: Tensor, (batch * beam, max_input_length, max_input_length),
                input attention mask.
            input_lengths: (batch * beam,), input sequence lengths.
            memory_length: int, the length of memory to keep key/cache values.
            compact_index: IntTensor, (compact_batch_size,)
                The index of input sequences of a compact batch. If None, the FT op
                doesn't apply the shared context feature and as result the inference
                time may increase.
            batch_to_compact_index: IntTensor, (batch * beam,)
                The index map from the original input batch to the compact batch.
                This must be provided if compact_index is not None.
            linear_bias_slopes: (num_heads,)
                The slope per head of linear attention bias - ALiBi. If None, a base
                self attention will be performed.
        # Returns
            hidden_states: Tensor, (batch * beam, max_input_length, hidden_dim),
                decoder outputs.
            key_cache: Tensor, (num_layers, batch * beam, local_num_heads, size_per_head / x, memory_length, x),
                key cache of attention of inputs.
                x = 16 / sizeof(T), memory_length = max_input_length or max_input_length + gen_length
            value_cache: Tensor, (num_layers, batch * beam, local_num_heads, memory_length, hidden_dim)
                value cache of attention
            last_token_hidden_states: Tensor, (batch * beam, hidden_dim)
                hidden states of the last input token.
        """
        self._initialize_op()
        # outputs: output hidden states
        assert self.ft_op is not None
        with self.lock:
            decoder_ouptut, key_cache, value_cache, last_token_hidden_states = self.ft_op.forward(
                input_embeds,
                attention_mask,
                input_lengths,
                lora_ids,
                memory_length,
                compact_index,
                batch_to_compact_index,
                linear_bias_slopes,
                prefix_prompt,
                prefix_lengths,
                key_cache,
                value_cache,
                block_index_map)
            return decoder_ouptut, key_cache, value_cache, last_token_hidden_states