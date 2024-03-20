import gc
import torch
from typing import List, Any, Dict, Optional, Union
from maga_transformer.models.base_model import BaseModel, GenerateInput
from transformers import PreTrainedTokenizerBase
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.async_decoder_engine.ptuning.ptuning import PrefixParams, PrefixType
from maga_transformer.async_decoder_engine.decoder_engine import DecoderEngine
from maga_transformer.async_decoder_engine.cache_manager import CacheManager

class PtuningConstructor(object):
    def __init__(self, model: BaseModel, tokenizer: PreTrainedTokenizerBase, cache_manager: CacheManager, decoder_engine: DecoderEngine):
        self.model = model
        self.tokenizer = tokenizer
        self.config = model.config
        self.decoder_engine = decoder_engine
        self.cache_manager = cache_manager

    def construct(self) -> Optional[Union[PrefixParams, Dict[int, PrefixParams]]]:
        if self.model.prefix_encoder is not None:
            assert self.model.prefix_tokens is not None
            prefix_prompt = self.model.prefix_encoder(self.model.prefix_tokens.cuda().unsqueeze(0)).squeeze(0)
            return self.create_ptuning_v2_params(self.config, self.cache_manager, prefix_prompt)
        elif self.model.config.multi_task_prompt is not None:
            return self._create_multi_task_prompt_params(self.model.config.multi_task_prompt)
        else:
            return None
    
    # static for ut
    @staticmethod
    def create_ptuning_v2_params(config:GptInitModelParameters, cache_manager: CacheManager, prefix_prompt: torch.Tensor):
        assert isinstance(prefix_prompt,  torch.Tensor), "prefix prompt is not torch.Tensor"
        prefix_seq_length = prefix_prompt.size(-2)
        # prefix_prompt shape [layer_num * 2, head_num_kv, pre_seq_len, size_per_head]
        prefix_prompt = prefix_prompt.reshape(config.layer_num, 2, prefix_prompt.size(1), prefix_prompt.size(2), prefix_prompt.size(3)).permute(1, 0, 3, 2, 4).contiguous()
        prefix_blocks = (prefix_seq_length - 1) // config.seq_size_per_block + 1
        prefix_block_indice = cache_manager.malloc(prefix_blocks)

        PtuningConstructor._set_kv_prefix_block(config, cache_manager, prefix_prompt, prefix_block_indice)
        return PrefixParams(prefix_type=PrefixType.PTuningV2, prefix_length=prefix_seq_length, block_cache=prefix_block_indice, prefix_tensor=None)

    def _create_multi_task_prompt_params(self, multi_task_prompt: List[Dict[str, Any]]):
        multi_task_prompt_args: Dict[int, PrefixParams] = {}
        for info in multi_task_prompt:
            id: int = info['task_id']
            prompt: str = info['prompt']
            input_tokens = torch.IntTensor(self.tokenizer.encode(prompt))
            input = GenerateInput(token_ids=input_tokens, generate_config=GenerateConfig(max_new_tokens=1))
            stream = self.decoder_engine.create_stream(input)
            # clear _resource_dtors to avoid relese block
            stream.set_require_release(False)
            self.decoder_engine.step()
            assert stream.output.hidden_states is not None, "stream should be run once"
            assert len(stream.block_indice[0]) > 0, "stream should have block indice"
            multi_task_prompt_args[id] = PrefixParams(prefix_type=PrefixType.PromptTuning, prefix_length=len(input_tokens), block_cache=stream.block_indice[0], prefix_tensor=input_tokens)
        return multi_task_prompt_args

    # input shape [layer_num, pre_seq_len, head_num, size_per_head]
    # dest k shape [layer_num, block_nums, head_num, seq_num_block, size_per_head]
    # dest v shape [layer_num, block_nums, head_num, seq_num_block, size_per_head]
    @staticmethod
    def _set_kv_prefix_block(config: GptInitModelParameters, cache_manager: CacheManager, kv_prefix_prompt: torch.Tensor, prefix_block_indice: List[int]):
        k_prefix_prompt = kv_prefix_prompt[0]
        v_prefix_prompt = kv_prefix_prompt[1]
        layer_num = k_prefix_prompt.size(0)
        pre_seq_len = k_prefix_prompt.size(1)
        head_num = k_prefix_prompt.size(2)
        size_per_head = k_prefix_prompt.size(3)
        block_indice_length = len(prefix_block_indice)
        append_length = len(prefix_block_indice) * config.seq_size_per_block - pre_seq_len
        blank_tensor = torch.zeros(layer_num, append_length, head_num, size_per_head).to(k_prefix_prompt)
        # [layer_num, block_num * seq_num_per_block, head_num, size_per_head]
        tiled_k_prefix_prompt = torch.concat([k_prefix_prompt, blank_tensor], dim=1)
        tiled_v_prefix_prompt = torch.concat([v_prefix_prompt, blank_tensor], dim=1)
        tiled_k_prefix_prompt = tiled_k_prefix_prompt.reshape(layer_num, block_indice_length, config.seq_size_per_block, head_num, size_per_head).permute(0, 1, 3, 2, 4).contiguous()
        tiled_v_prefix_prompt = tiled_v_prefix_prompt.reshape(layer_num, block_indice_length, config.seq_size_per_block, head_num, size_per_head).permute(0, 1, 3, 2, 4).contiguous()
        for i in range(block_indice_length):
            cache_manager.set_kv_block_value(prefix_block_indice[i], tiled_k_prefix_prompt[ :, i, ...], tiled_v_prefix_prompt[ :, i, ...])