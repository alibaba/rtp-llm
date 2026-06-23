from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.attn_type import HCA_KV, SWA_KV, TAG_BY_ATTN_TYPE
from rtp_llm.models_py.modules.dsv4.decode.forward import build_paged_pool_specs
from rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils import (
    require_pool_tokens_per_block,
)
from rtp_llm.models_py.modules.dsv4.fp8._cp_slot_mapping import cp_kv_slot_mapping
from rtp_llm.models_py.modules.dsv4.fp8.decode.paged_topk_translator import (
    translate_local_to_global_slots,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.pool_slot_mapping import (
    compute_kv_pool_slot_mapping,
)


class PoolSlotMappingSplitTest(unittest.TestCase):
    def test_require_pool_tokens_per_block_inferrs_from_scalar_region(self) -> None:
        class FakeKVCache:
            group_tags = [TAG_BY_ATTN_TYPE[HCA_KV], TAG_BY_ATTN_TYPE[SWA_KV]]
            seq_size_per_block = 16384
            kernel_seq_size_per_block = 128

        self.assertEqual(require_pool_tokens_per_block(FakeKVCache(), group=0), 128)
        self.assertEqual(require_pool_tokens_per_block(FakeKVCache(), group=1), 16384)
        self.assertEqual(require_pool_tokens_per_block(FakeKVCache(), region=1), 128)
        self.assertEqual(require_pool_tokens_per_block(FakeKVCache(), region=7), 16384)

    def test_require_pool_tokens_per_block_prefers_group_override(self) -> None:
        class FakeKVCache:
            group_tags = [TAG_BY_ATTN_TYPE[HCA_KV], TAG_BY_ATTN_TYPE[SWA_KV]]
            group_seq_size_per_block = [256, 1024]
            seq_size_per_block = 256
            kernel_seq_size_per_block = 256

        self.assertEqual(require_pool_tokens_per_block(FakeKVCache(), group=0), 256)
        self.assertEqual(require_pool_tokens_per_block(FakeKVCache(), group=1), 1024)
        self.assertEqual(require_pool_tokens_per_block(FakeKVCache(), region=int(SWA_KV)), 1024)

    def test_cp_kv_owner_uses_full_physical_block_not_compact_state_block(self) -> None:
        positions = torch.tensor([3, 259, 515, 771], dtype=torch.int64)
        b_idx = torch.zeros_like(positions)

        rank0 = cp_kv_slot_mapping(
            positions,
            torch.tensor([[10]], dtype=torch.int64),
            b_idx,
            tokens_per_block=256,
            kv_eb=64,
            ratio=4,
            cp_size=4,
            cp_rank=0,
            owner_tokens_per_block=256,
        )
        rank3 = cp_kv_slot_mapping(
            positions,
            torch.tensor([[13]], dtype=torch.int64),
            b_idx,
            tokens_per_block=256,
            kv_eb=64,
            ratio=4,
            cp_size=4,
            cp_rank=3,
            owner_tokens_per_block=256,
        )

        torch.testing.assert_close(rank0, torch.tensor([640, -1, -1, -1], dtype=torch.int64))
        torch.testing.assert_close(rank3, torch.tensor([-1, -1, -1, 832], dtype=torch.int64))

    def test_build_paged_pool_specs_uses_dsv4_pool_tokens(self) -> None:
        class FakeKVCache:
            group_tags = [TAG_BY_ATTN_TYPE[HCA_KV], TAG_BY_ATTN_TYPE[SWA_KV]]
            seq_size_per_block = 128
            kernel_seq_size_per_block = 128

        class FakeAttn:
            _kv_cache = None

            def _pool_entries_per_block(self, attn_type: int) -> int:
                if int(attn_type) == int(HCA_KV):
                    return 1
                if int(attn_type) == int(SWA_KV):
                    return 32
                return 0

        class FakeLayer:
            attn = FakeAttn()

        class FakeV4:
            layers = [FakeLayer()]

        specs = build_paged_pool_specs(FakeKVCache(), FakeV4(), max_seq_len=256)

        self.assertEqual(specs[int(HCA_KV)][1], 128)
        self.assertEqual(specs[int(SWA_KV)][1], 128)

    def test_require_pool_tokens_per_block_rejects_unknown_region(self) -> None:
        class FakeKVCache:
            group_tags = ["unknown"]
            seq_size_per_block = 16384
            kernel_seq_size_per_block = 128

        with self.assertRaisesRegex(RuntimeError, "cannot be inferred"):
            require_pool_tokens_per_block(FakeKVCache(), region=99)

    def test_compute_full_params_equal_entries(self) -> None:
        block_table = torch.tensor([[3, 4]], dtype=torch.int32)
        abs_pos = torch.tensor([0, 127, 128, 255, -1], dtype=torch.int32)

        got = compute_kv_pool_slot_mapping(
            block_table,
            abs_pos,
            pool_entries_per_block=128,
            pool_tokens_per_block=128,
            ring_entries=128,
        )

        expected = torch.tensor([384, 511, 512, 639, -1], dtype=torch.long)
        torch.testing.assert_close(got, expected)

    def test_compute_swa_physical_rows_with_short_ring(self) -> None:
        block_table = torch.tensor([[7, 11]], dtype=torch.int32)
        abs_pos = torch.tensor([16382, 16383, 16384, 16385, -1], dtype=torch.int32)

        got = compute_kv_pool_slot_mapping(
            block_table,
            abs_pos,
            pool_entries_per_block=132,
            pool_tokens_per_block=16384,
            ring_entries=132,
        )

        expected = torch.tensor([938, 939, 1468, 1469, -1], dtype=torch.long)
        torch.testing.assert_close(got, expected)

    def test_compute_rejects_reserved_or_null_blocks(self) -> None:
        block_table = torch.tensor([[0, -1]], dtype=torch.int32)
        abs_pos = torch.tensor([0, 128], dtype=torch.int32)

        got = compute_kv_pool_slot_mapping(
            block_table,
            abs_pos,
            pool_entries_per_block=128,
            pool_tokens_per_block=128,
            ring_entries=128,
        )

        expected = torch.tensor([-1, -1], dtype=torch.long)
        torch.testing.assert_close(got, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_translate_full_params_equal_entries(self) -> None:
        device = torch.device("cuda")
        req_id = torch.tensor([0], dtype=torch.int32, device=device)
        block_table = torch.tensor([[3, 4]], dtype=torch.int32, device=device)
        local_idx = torch.tensor([[0, 127, 128, -1]], dtype=torch.int32, device=device)

        got = translate_local_to_global_slots(
            req_id,
            block_table,
            local_idx,
            entries_per_block=128,
            tokens_per_block_for_block_table=128,
        )

        expected = torch.tensor([[384, 511, 512, -1]], dtype=torch.int32, device=device)
        torch.testing.assert_close(got, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_translate_swa_physical_rows_with_short_entries(self) -> None:
        device = torch.device("cuda")
        req_id = torch.tensor([0], dtype=torch.int32, device=device)
        block_table = torch.tensor([[7, 11]], dtype=torch.int32, device=device)
        local_idx = torch.tensor(
            [[16382, 16383, 16384, 16385]],
            dtype=torch.int32,
            device=device,
        )

        got = translate_local_to_global_slots(
            req_id,
            block_table,
            local_idx,
            entries_per_block=132,
            tokens_per_block_for_block_table=16384,
        )

        expected = torch.tensor(
            [[938, 939, 1468, 1469]],
            dtype=torch.int32,
            device=device,
        )
        torch.testing.assert_close(got, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_translate_compressor_entries_matrix(self) -> None:
        device = torch.device("cuda")
        req_id = torch.tensor([0], dtype=torch.int32, device=device)
        local_idx = torch.arange(128, dtype=torch.int32, device=device).view(1, 128)
        for entries_per_block in (1, 2, 32, 64):
            max_blocks = 128 // entries_per_block + 1
            block_ids = torch.arange(
                3,
                3 + max_blocks,
                dtype=torch.int32,
                device=device,
            ).view(1, max_blocks)
            with self.subTest(entries_per_block=entries_per_block):
                got = translate_local_to_global_slots(
                    req_id,
                    block_ids,
                    local_idx,
                    entries_per_block=entries_per_block,
                    tokens_per_block_for_block_table=entries_per_block,
                )
                expected_block = block_ids[0, local_idx // entries_per_block]
                expected = expected_block * entries_per_block + (
                    local_idx % entries_per_block
                )
                torch.testing.assert_close(got, expected.to(torch.int32))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_translate_swa_entries_matrix(self) -> None:
        device = torch.device("cuda")
        req_id = torch.tensor([0], dtype=torch.int32, device=device)
        base_positions = torch.tensor(
            [0, 1, 127, 128, 129, 130, 131, 132, 133, 134, 16382, 16383, 16384, 16385],
            dtype=torch.int32,
            device=device,
        )
        local_idx = torch.full((1, 128), -1, dtype=torch.int32, device=device)
        local_idx[0, : base_positions.numel()] = base_positions
        block_table = torch.tensor([[7, 11]], dtype=torch.int32, device=device)
        for entries_per_block in (128, 130, 132, 134):
            with self.subTest(entries_per_block=entries_per_block):
                got = translate_local_to_global_slots(
                    req_id,
                    block_table,
                    local_idx,
                    entries_per_block=entries_per_block,
                    tokens_per_block_for_block_table=16384,
                )
                expected = torch.full_like(local_idx, -1)
                valid = local_idx >= 0
                safe = torch.where(valid, local_idx, torch.zeros_like(local_idx))
                expected_block = block_table[0, safe // 16384]
                expected = torch.where(
                    valid,
                    expected_block * entries_per_block + (safe % entries_per_block),
                    expected,
                )
                torch.testing.assert_close(got, expected)


if __name__ == "__main__":
    unittest.main()
