import unittest

import torch

from rtp_llm.models.kimi_k3.decode_ktp_cache_model import (
    DecodeKtpCacheSpec,
    allocate_decode_ktp_cache,
    build_owner_local_mla_block_table,
)


def _spec(**overrides) -> DecodeKtpCacheSpec:
    values = dict(
        global_batch=8,
        tp_size=8,
        tp_rank=0,
        kv_length=16,
        tokens_per_block=8,
        mla_layer_num=1,
        kda_layer_num=3,
        linear_num_heads=16,
        linear_head_dim=8,
        linear_conv_kernel_dim=4,
    )
    values.update(overrides)
    return DecodeKtpCacheSpec(**values)


class DecodeKtpCacheModelTest(unittest.TestCase):
    def test_128k_mla_bytes_match_full_576_bf16_formula(self) -> None:
        spec = _spec(kv_length=128 * 1024, tokens_per_block=64)
        self.assertEqual(spec.mla_bytes(owner_local=True), 144 * 1024**2)
        self.assertEqual(spec.mla_bytes(owner_local=False), 1152 * 1024**2)

    def test_owner_local_allocation_is_one_eighth_of_baseline_mla(self) -> None:
        spec = _spec(global_batch=32, tp_rank=3, kv_length=32)
        baseline = allocate_decode_ktp_cache(
            spec, owner_local_mla=False, device=torch.device("cpu")
        )
        owner = allocate_decode_ktp_cache(
            spec, owner_local_mla=True, device=torch.device("cpu")
        )

        self.assertEqual(
            baseline.mla_allocated_bytes, owner.mla_allocated_bytes * spec.tp_size
        )
        self.assertEqual(baseline.kda_allocated_bytes, owner.kda_allocated_bytes)
        baseline.assert_matches_spec()
        owner.assert_matches_spec()

    def test_owner_block_table_is_dense_and_non_overlapping(self) -> None:
        spec = _spec(global_batch=16, tp_rank=5, kv_length=17)
        table = build_owner_local_mla_block_table(spec, first_block_id=7)
        self.assertEqual(tuple(table.shape), (2, 3))
        self.assertEqual(table.flatten().tolist(), [7, 8, 9, 10, 11, 12])

    def test_rejects_kda_heads_not_divisible_by_tp(self) -> None:
        with self.assertRaisesRegex(ValueError, "KDA heads must be divisible"):
            _spec(linear_num_heads=12)


if __name__ == "__main__":
    unittest.main()
