"""Isolated PageRR packing: reserved startup pages and ordinary request pages."""
import unittest

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_page_rr_cache_gather import MlaPageRRCacheGather


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class WarmupCachePackTest(unittest.TestCase):
    def test_reserved_zero_and_regular_pages_preserve_payload(self):
        torch.cuda.set_device(0)
        adapter = MlaPageRRCacheGather(1024, 8, 0, 128)
        for fp8 in (False, True):
            features = 132 if fp8 else 576
            if fp8:
                cache = torch.randint(0, 256, (16, 128, features), device='cuda', dtype=torch.uint8)
                flat = cache.view(16, -1)
                canonical = torch.cat((flat[:, :128 * 128].view(16, 128, 128),
                                       flat[:, 128 * 128:].view(16, 128, 4)), -1)
            else:
                cache = torch.randn(16, 128, features, device='cuda', dtype=torch.bfloat16)
                canonical = cache
            descriptor = adapter.build_prefix_chunk_descriptor((0,), (0,), (1024,),
                                                                feature_width=features)
            for warmup, first in ((False, 8), (True, 0)):
                table = torch.arange(first, first + 8, device='cuda', dtype=torch.int32)[None]
                output = adapter._pack_prefix_chunk(cache, table, descriptor,
                    indexer_fp8=fp8, allow_reserved_block_zero=warmup)
                expected = canonical[first:first + 8].reshape_as(output)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
