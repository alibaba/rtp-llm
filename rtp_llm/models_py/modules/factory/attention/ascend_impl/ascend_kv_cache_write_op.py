import torch
import torch_npu

from rtp_llm.ops.compute_ops import LayerKVCache


class AscendKVCacheWriteOp:
    """MHA KV Cache write using torch_npu._npu_reshape_and_cache."""

    def __init__(self, num_kv_heads, head_size, token_per_block):
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.token_per_block = token_per_block
        self.params = None

    def set_params(self, params):
        self.params = params

    def forward(self, key, value, kv_cache):
        if kv_cache is None:
            return

        k_cache = kv_cache.k_cache_base
        v_cache = kv_cache.v_cache_base

        slot_mapping = self.params.slot_mapping
        if slot_mapping.dtype != torch.int32:
            slot_mapping = slot_mapping.to(torch.int32)

        torch_npu._npu_reshape_and_cache(
            key=key,
            value=value,
            key_cache=k_cache,
            value_cache=v_cache,
            slot_indices=slot_mapping,
        )

    def _prepare_warmup_cache_indices(self, num_tokens, device):
        import torch
        batch_indices = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        positions = torch.arange(num_tokens, dtype=torch.int32, device=device)
        max_num_pages = (num_tokens + self.token_per_block - 1) // self.token_per_block
        kv_page_indices = positions // self.token_per_block
        kv_page_indptr = torch.tensor([0, max_num_pages], dtype=torch.int32, device=device)
        last_page_len = num_tokens % self.token_per_block
        if last_page_len == 0:
            last_page_len = self.token_per_block
        kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=device)
        return batch_indices, positions, kv_page_indices, kv_page_indptr, kv_last_page_len, max_num_pages
