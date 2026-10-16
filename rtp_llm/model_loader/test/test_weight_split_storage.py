"""Weight sharding keeps dense independent storage without a second copy."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from rtp_llm.model_loader.weight_module import AtomicWeight


class WeightSplitStorageTest(unittest.TestCase):
    def config(self):
        return SimpleNamespace(
            tp_size=2,
            tp_rank=1,
            ep_size=1,
            ep_rank=0,
            dp_size=1,
            dp_rank=0,
            ffn_tp_rank=1,
            ffn_tp_size=2,
            hidden_size=16,
            head_num=2,
            head_num_kv=2,
            size_per_head=8,
            moe_pure_tp_mode=True,
            bit=8,
        )

    def split(self, tensor, columns):
        weight = AtomicWeight("storage_probe", [])

        def shard(t, tp, tp_rank, **kwargs):
            return t[:, tp_rank::tp] if columns else t[tp_rank::tp]

        with mock.patch.object(weight, "_get_split_func", return_value=shard):
            return weight._split(tensor, self.config())["storage_probe"]

    def test_values_layout_and_independent_ownership(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for columns in (False, True):
                source = torch.arange(99, dtype=dtype).reshape(9, 11)
                original = source.clone()
                expected = source[:, 1::2] if columns else source[1::2]
                output = self.split(source, columns)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                self.assertTrue(output.is_contiguous())
                output.zero_()
                torch.testing.assert_close(source, original, rtol=0, atol=0)

    def test_noncontiguous_cuda_shard_has_one_copy_peak(self):
        self.assertTrue(torch.cuda.is_available(), "GPU validation must not skip")
        source = torch.randn(2051, 1029, device="cuda", dtype=torch.bfloat16)
        expected = source[:, 1::2].cpu()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        output = self.split(source, True)
        torch.cuda.synchronize()
        peak_extra = torch.cuda.max_memory_allocated() - before
        self.assertLessEqual(peak_extra, output.numel() * output.element_size() + 65536)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        self.assertTrue(output.is_contiguous())


if __name__ == "__main__":
    unittest.main()
