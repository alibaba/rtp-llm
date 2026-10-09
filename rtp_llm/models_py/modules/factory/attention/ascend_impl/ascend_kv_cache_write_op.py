import torch
import torch_npu

from rtp_llm.ops.compute_ops import LayerKVCache


class AscendKVCacheWriteOp:
    """MHA KV Cache write using torch_npu.npu_scatter_pa_kv_cache.

    The combined KV buffer is [blocks, 2, seq, heads, dim] (BSND) after the
    C++ getLayerCache reshape.  kv_cache_base[:, 0/1] yields [blocks, seq,
    heads, dim] directly — no Python permute needed.  npu_scatter_pa_kv_cache
    supports non-contiguous inputs, so we scatter directly into the strided
    views without cloning.
    """

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

        kv_base = kv_cache.kv_cache_base
        # Already BSND [blocks, seq, heads, dim] from C++ reshape — no permute
        k_view = kv_base[:, 0]
        v_view = kv_base[:, 1]

        slot_mapping = self.params.slot_mapping
        if slot_mapping.dtype not in (torch.int32, torch.int64):
            slot_mapping = slot_mapping.to(torch.int32)

        torch_npu.npu_scatter_pa_kv_cache(
            key, value, k_view, v_view, slot_mapping, cache_mode = "Norm"
        )
