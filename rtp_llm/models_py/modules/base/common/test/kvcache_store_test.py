from unittest import TestCase, main, skipUnless

import torch

from rtp_llm.models_py.modules.base.common.kvcache_store import (
    _cache_store_host_i32,
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


if __name__ == "__main__":
    main()
