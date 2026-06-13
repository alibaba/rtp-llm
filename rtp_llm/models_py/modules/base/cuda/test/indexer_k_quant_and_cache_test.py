from unittest import SkipTest, TestCase, main

import torch

from rtp_llm.ops.compute_ops import rtp_llm_ops


class IndexerKQuantAndCacheTest(TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

    def test_rejects_short_slot_mapping_before_kernel_launch(self):
        k = torch.randn(1, 128, dtype=torch.bfloat16, device=self.device)
        kv_cache = torch.empty(1, 64, 132, dtype=torch.uint8, device=self.device)
        slot_mapping = torch.empty(0, dtype=torch.int64, device=self.device)

        with self.assertRaisesRegex(RuntimeError, "slot_mapping size"):
            rtp_llm_ops.indexer_k_quant_and_cache(
                k, kv_cache, slot_mapping, 128, "ue8m0"
            )

    def test_valid_single_token_write(self):
        k = torch.randn(1, 128, dtype=torch.bfloat16, device=self.device)
        kv_cache = torch.zeros(1, 64, 132, dtype=torch.uint8, device=self.device)
        slot_mapping = torch.tensor([0], dtype=torch.int64, device=self.device)

        rtp_llm_ops.indexer_k_quant_and_cache(k, kv_cache, slot_mapping, 128, "ue8m0")
        torch.cuda.synchronize()
        self.assertGreater(int(kv_cache.sum().item()), 0)


if __name__ == "__main__":
    main()
