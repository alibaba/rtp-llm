import unittest

import torch

from rtp_llm.models.kimi_k3.decode_ktp import (
    DecodeOwnerLayout,
    build_owner_attention_inputs,
)
from rtp_llm.ops.compute_ops import PyAttentionInputs


class DecodeOwnerLayoutTest(unittest.TestCase):
    def test_fixed_partition_covers_each_request_once(self):
        owners = [DecodeOwnerLayout.fixed(32, 8, rank) for rank in range(8)]
        self.assertEqual([owner.local_batch for owner in owners], [4] * 8)
        covered = [idx for owner in owners for idx in range(owner.start, owner.stop)]
        self.assertEqual(covered, list(range(32)))

    def test_rejects_non_divisible_or_too_small_batch(self):
        for batch in (1, 7, 9, 15):
            with self.assertRaisesRegex(ValueError, "BS divisible by TP"):
                DecodeOwnerLayout.fixed(batch, 8, 0)

    def test_owner_attention_metadata_keeps_order_and_group_rows(self):
        batch = 16
        attention = PyAttentionInputs()
        attention.is_prefill = False
        attention.is_target_verify = False
        attention.is_cuda_graph = False
        attention.cache_store_inputs = None
        attention.input_lengths = torch.ones(batch, dtype=torch.int32)
        attention.input_lengths_host = torch.arange(
            100, 100 + batch, dtype=torch.int32
        )
        attention.prefix_lengths = torch.arange(batch, dtype=torch.int32)
        attention.prefix_lengths_host = attention.prefix_lengths.clone()
        attention.sequence_lengths = torch.arange(100, 100 + batch, dtype=torch.int32)
        attention.sequence_lengths_host = attention.sequence_lengths.clone()
        attention.sequence_lengths_plus_1_d = attention.sequence_lengths + 1
        attention.kv_cache_block_id_host = torch.arange(
            2 * batch * 3, dtype=torch.int32
        ).reshape(2, batch, 3)
        group = torch.arange(batch * 4, dtype=torch.int32).reshape(batch, 4)
        attention.kv_cache_block_id_host_by_group = [group]
        attention.kv_cache_kernel_block_id_host_by_group = [group]
        attention.kv_cache_kernel_block_id_device_by_group = [group]

        layout = DecodeOwnerLayout.fixed(batch, 8, 3)
        local = build_owner_attention_inputs(
            attention,
            layout,
            device=torch.device("cpu"),
            global_query_tokens=batch,
        )

        self.assertEqual((layout.start, layout.stop), (6, 8))
        self.assertEqual(local.prefix_lengths_host.tolist(), [6, 7])
        self.assertEqual(local.sequence_lengths_host.tolist(), [106, 107])
        self.assertEqual(local.cu_seqlens.tolist(), [0, 1, 2])
        self.assertEqual(local.kv_cache_block_id_host.shape, (2, 2, 3))
        self.assertTrue(
            torch.equal(local.kv_cache_block_id_host_by_group[0], group[6:8])
        )

    def test_rejects_q_len_greater_than_one(self):
        attention = PyAttentionInputs()
        attention.is_prefill = False
        attention.is_target_verify = False
        attention.is_cuda_graph = False
        attention.cache_store_inputs = None
        attention.input_lengths_host = torch.arange(100, 108, dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "q_len=1"):
            build_owner_attention_inputs(
                attention,
                DecodeOwnerLayout.fixed(8, 8, 7),
                device=torch.device("cpu"),
                global_query_tokens=9,
            )


if __name__ == "__main__":
    unittest.main()
