import gc
import torch
from typing import List, Any, Dict, Tuple
from maga_transformer.models.base_model import BaseModel, BaseTokenizer
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.async_decoder_engine.ptuning.ptuning import PrefixParams, PrefixType

def get_ptuning_params(model: BaseModel, tokenizer: BaseTokenizer):
    if model.prefix_encoder is not None:
        assert model.prefix_tokens is not None
        prefix_prompt = model.prefix_encoder(model.prefix_tokens.cuda().unsqueeze(0)).squeeze(0)
        return PrefixParams(prefix_prompt, PrefixType.PTuningV2, None)
    elif model.config.multi_task_prompt is not None:
        prefix_prompt, prefix_tensors = prepare_prompt(model, tokenizer, model.config.multi_task_prompt)
        return PrefixParams(prefix_prompt, PrefixType.PromptTuning, prefix_tensors)
    else:
        return None

def prepare_prompt(model: BaseModel, tokenizer: BaseTokenizer, multi_task_prompt: List[Dict[str, Any]]) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    multi_task_prompt_tensor: Dict[int, torch.Tensor] = {}
    multi_task_tensor_id: Dict[int, torch.Tensor] = {}
    def run_context_decoder(input_token_ids: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        generate_context = model.prepare_context(input_token_ids, input_lengths, [], GenerateConfig(max_new_tokens=0))
        ctx_output, k_cache, v_cache, _ = model.context_decoder.forward(
            input_embeds=generate_context.input_embeds,
            attention_mask=generate_context.attention_mask,
            input_lengths=generate_context.input_lengths,
            memory_length=generate_context.memory_length,
            **generate_context.extra_args)
        # print(k_cache, v_cache)
        kvcache = torch.concat([k_cache.unsqueeze_(0), v_cache.unsqueeze_(0)], dim=0)
        kvcache = kvcache.squeeze_(2).permute(1, 0, 2, 3, 4).contiguous()
        s = kvcache.shape            
        # offload到cpu，避免额外的浪费
        kvcache = kvcache.reshape(s[0] * s[1], s[2], s[3], s[4]).cpu()
        return kvcache

    for info in multi_task_prompt:
        id: int = info['task_id']
        prompt: str = info['prompt']
        input_tokens = torch.IntTensor(tokenizer.encode(prompt))
        input_length = torch.IntTensor([input_tokens.nelement()])
        kvcache = run_context_decoder(input_tokens.unsqueeze(0), input_length)
        multi_task_prompt_tensor[id] = kvcache
        multi_task_tensor_id[id] = input_tokens
    
    # 释放占用的显存，避免干扰kvcache allocate
    gc.collect()
    torch.cuda.empty_cache()

    return multi_task_prompt_tensor, multi_task_tensor_id