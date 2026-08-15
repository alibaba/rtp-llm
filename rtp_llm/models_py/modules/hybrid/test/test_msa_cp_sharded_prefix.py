"""Unit coverage for MSA page-RR prefix scratch reconstruction."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.hybrid.msa_attention import MSAAttention

_MSA_MODULE = "rtp_llm.models_py.modules.hybrid.msa_attention"


class TestMSACpShardedPrefixRestore(unittest.TestCase):

    @patch(f"{_MSA_MODULE}.gather_cp_sharded_prefix_pool")
    def test_restores_multi_request_main_kv_and_idx_in_logical_order(self, gather):
        # Three logical pages, two tokens per page. Request 0 owns pages 0/1;
        # request 1 owns page 2. Values encode the global logical token order.
        main_pages = torch.zeros(3, 2, 1, 2, 1, dtype=torch.bfloat16)
        main_pages[:, 0, 0, :, 0] = torch.tensor([[1, 2], [3, 4], [5, 6]])
        main_pages[:, 1, 0, :, 0] = torch.tensor([[11, 12], [13, 14], [15, 16]])
        idx_pages = torch.tensor(
            [[[21], [22]], [[23], [24]], [[25], [26]]], dtype=torch.bfloat16
        )
        gather.side_effect = [main_pages, idx_pages]

        attn = MSAAttention.__new__(MSAAttention)
        torch.nn.Module.__init__(attn)
        attn._kv_sharded = True
        attn._cp_size = 2
        attn._cp_rank = 0
        attn.page_size = 2
        attn.kv_head_num = 1
        attn.head_dim = 1
        attn.idx_head_dim = 1
        attn._scratch_k = torch.zeros(12, 1, 1, dtype=torch.bfloat16)
        attn._scratch_v = torch.zeros_like(attn._scratch_k)
        attn._scratch_idx_k = torch.zeros_like(attn._scratch_k)
        attn._physical_block_table = lambda _inputs: torch.tensor([[3], [7]])
        attn._paged_kv_base_view = lambda _cache: torch.empty(8, 2, 1, 2, 1)
        attn._idx_k_paged_view = lambda _cache: torch.empty(8, 2, 1)
        req_to_token = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])

        attn._restore_cp_sharded_prefix_scratch(
            SimpleNamespace(), [4, 2], req_to_token, SimpleNamespace()
        )

        self.assertEqual(gather.call_count, 2)
        self.assertTrue(
            torch.equal(attn._scratch_k[[0, 1, 2, 3, 6, 7], 0, 0], torch.arange(1, 7))
        )
        self.assertTrue(
            torch.equal(attn._scratch_v[[0, 1, 2, 3, 6, 7], 0, 0], torch.arange(11, 17))
        )
        self.assertTrue(
            torch.equal(
                attn._scratch_idx_k[[0, 1, 2, 3, 6, 7], 0, 0],
                torch.arange(21, 27),
            )
        )

    @patch(f"{_MSA_MODULE}.gather_cp_sharded_prefix_pool")
    def test_zero_prefix_does_not_collect(self, gather):
        attn = MSAAttention.__new__(MSAAttention)
        torch.nn.Module.__init__(attn)
        attn._kv_sharded = True
        attn._restore_cp_sharded_prefix_scratch(
            SimpleNamespace(), [0], torch.zeros(1, 1), SimpleNamespace()
        )
        gather.assert_not_called()


if __name__ == "__main__":
    unittest.main()
