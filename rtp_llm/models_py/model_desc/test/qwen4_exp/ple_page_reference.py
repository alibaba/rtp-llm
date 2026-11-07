"""Page-by-page oracle for valid eager PLE prefill and page-aligned reuse.

This checks page orchestration against bulk prefill. PLE layer math has separate
upstream parity tests. Read and write physical tables directly so production
page-resolution helpers cannot hide a shared addressing bug.
"""

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import PLE_NGRAM_CTX_TAG, PLE_STATE_TAG


def _apply_ple(self, layer_idx, hyper_states, input_ids, attention_inputs):
    ple = self.ple_layers[str(layer_idx)]
    state_inputs = attention_inputs[PLE_STATE_TAG]
    ctx_inputs = attention_inputs[PLE_NGRAM_CTX_TAG]
    assert state_inputs.is_prefill and not state_inputs.is_target_verify
    assert not state_inputs.is_cuda_graph
    state_cache = self.kv_cache.get_layer_cache(layer_idx, PLE_STATE_TAG)
    ctx_cache = self.kv_cache.get_layer_cache(layer_idx, PLE_NGRAM_CTX_TAG)
    page_size = state_cache.seq_size_per_block
    context_len = ple.ple_embedding.context_len
    state_pool = state_cache.kv_cache_base.view(torch.bfloat16).reshape(
        -1, ple.short_conv_state_len, ple.hc_hidden_size
    )
    ctx_pool = ctx_cache.kv_cache_base.view(torch.int64).reshape(-1, context_len)
    state_table = state_inputs.kv_cache_block_id_device.tolist()
    ctx_table = ctx_inputs.kv_cache_block_id_device.tolist()
    ids = input_ids.flatten()
    lengths = state_inputs.input_lengths.tolist()
    prefixes = state_inputs.prefix_lengths.tolist() or [0] * len(lengths)
    result = torch.empty_like(hyper_states)
    offset = 0
    with ple.ple_embedding.lookup_context(False):
        for request, (length, prefix) in enumerate(zip(lengths, prefixes)):
            assert length > 0 and prefix % page_size == 0
            if prefix:
                previous_page = prefix // page_size - 1
                state = (
                    state_pool[state_table[request][previous_page]].unsqueeze(0).clone()
                )
                context = (
                    ctx_pool[ctx_table[request][previous_page]].unsqueeze(0).clone()
                )
            else:
                state = None
                context = ids.new_full((1, context_len), ple.ple_embedding.eos_token_id)
            for start in range(0, length, page_size):
                end = min(start + page_size, length)
                history = torch.cat(
                    (context, ids[None, offset + start : offset + end]), 1
                )
                hyper = hyper_states[offset + start : offset + end].unsqueeze(0)
                if state is None:
                    output, state = ple.prefill(hyper, history)
                else:
                    output, state = ple.prefill_with_state(hyper, history, state)
                result[offset + start : offset + end] = output[0]
                context = history[:, -context_len:]
                page = (prefix + end - 1) // page_size
                state_block = state_table[request][page]
                ctx_block = ctx_table[request][page]
                if state_block > 0 and ctx_block > 0:
                    state_pool[state_block].copy_(state[0])
                    ctx_pool[ctx_block].copy_(context[0])
            offset += length
    return hyper_states + result
