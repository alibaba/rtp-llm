from types import SimpleNamespace
from unittest import TestCase, main, skipUnless

import torch

from rtp_llm.models_py.modules.base.common.kvcache_store import (
    _cache_store_host_i32,
    create_write_cache_store_impl,
)


class CacheStoreHostMetadataTest(TestCase):
    def test_rejects_wrong_dtype(self):
        with self.assertRaisesRegex(RuntimeError, "must be int32"):
            _cache_store_host_i32(torch.ones(1, dtype=torch.int64), "lengths")

    @skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_moves_cuda_lengths_to_contiguous_cpu(self):
        lengths = torch.tensor([7, 11], dtype=torch.int32, device="cuda")

        host = _cache_store_host_i32(lengths, "lengths")

        self.assertEqual(host.device.type, "cpu")
        self.assertEqual(host.dtype, torch.int32)
        self.assertTrue(host.is_contiguous())
        self.assertEqual(host.tolist(), [7, 11])

    def test_multi_region_writer_uses_physical_block_tables(self):
        physical = [torch.tensor([[1]], dtype=torch.int32)]
        kernel = [torch.tensor([[8, 9, 10, 11, 12, 13, 14, 15]], dtype=torch.int32)]
        cache_store_inputs = SimpleNamespace(
            input_lengths_host=torch.tensor([18], dtype=torch.int32),
            prefix_lengths_host=torch.tensor([0], dtype=torch.int32),
        )
        attn_inputs = SimpleNamespace(
            is_prefill=True,
            cache_store_inputs=cache_store_inputs,
            input_lengths=cache_store_inputs.input_lengths_host,
            prefix_lengths=cache_store_inputs.prefix_lengths_host,
            context_parallel_info=None,
            kv_cache_block_id_host=torch.tensor([[1]], dtype=torch.int32),
            kv_cache_block_id_host_by_group=physical,
            kv_cache_kernel_block_id_host_by_group=kernel,
        )
        kv_cache = SimpleNamespace(layer_region_to_group_id=[[0]])

        writer = create_write_cache_store_impl(attn_inputs, kv_cache)

        self.assertIsNotNone(writer)
        self.assertIs(writer._block_ids_by_group[0], physical[0])
        self.assertEqual(writer._block_ids_by_group[0].tolist(), [[1]])


if __name__ == "__main__":
    main()
