"""Packed SWA transport must preserve attention and persistent-cache bytes."""

import os
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8._swa_cp_byte_sliced import (
    build_cp_byte_sliced_slot_compaction,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_cp_packed_triton import (
    build_attention_cache,
    insert_packed_rows,
    insert_packed_rows_cp_byte_sliced,
    pack_local_kv,
    packed_swa_cp_enabled,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
    gather_k_cache_slots_packed,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
    quantize_and_insert_k_cache,
    quantize_and_insert_k_cache_cp_byte_sliced,
)
from rtp_llm.models_py.modules.dsv4.prefill_workspace import PrefillWorkspace


class PackedSwaFlagTest(unittest.TestCase):
    def test_default_off_and_invalid_values(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(packed_swa_cp_enabled())
            os.environ["DSV4_CP_SWA_PACKED"] = "1"
            self.assertTrue(packed_swa_cp_enabled())
            os.environ["DSV4_CP_SWA_PACKED"] = "true"
            with self.assertRaises(ValueError):
                packed_swa_cp_enabled()

    def test_prefix_byte_preservation_keeps_existing_transport(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import attention

        common = SimpleNamespace(
            cp_on=True,
            batch_size=1,
            workspace=object(),
            cp_ctx=SimpleNamespace(cp_size=4),
            csa_meta=None,
            hca_meta=None,
        )
        kv = torch.zeros(1, 512, dtype=torch.bfloat16)
        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(os.environ, {"DSV4_CP_SWA_PACKED": "1"}, clear=True)
            )
            stack.enter_context(patch.object(attention, "is_sm120", return_value=True))
            stack.enter_context(
                patch.object(
                    torch.cuda, "is_current_stream_capturing", return_value=False
                )
            )
            for name in (
                "_cp_swa_kv_async_enabled",
                "_prefill_cp_overlap_enabled",
                "_fp8_gather_enabled",
                "_sm120_paged_cache_prefill_enabled",
                "_use_cp_cache_hit_raw_q_merge",
            ):
                stack.enter_context(patch.object(attention, name, return_value=False))
            for ratio in (0, 4, 128):
                layer = SimpleNamespace(
                    head_dim=512, window_size=128, compress_ratio=ratio
                )
                for preserve in ("0", "1"):
                    with self.subTest(compress_ratio=ratio, preserve=preserve):
                        os.environ["DSV4_SM120_PRESERVE_PREFIX_BYTES"] = preserve
                        self.assertEqual(
                            attention.AttentionFP8._use_packed_swa_cp(
                                layer, common, kv
                            ),
                            preserve == "0",
                        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PackedSwaTransportTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19073)
        self.device = torch.device("cuda")
        self.workspace = PrefillWorkspace(
            self.device, q_rows=1, q_dim=1, reserve_cp=False
        )

    def kv(self, count):
        value = torch.randn(count, 512, device=self.device)
        value *= torch.exp2(
            torch.arange(count, device=self.device).remainder(21).float() - 10
        )[:, None]
        if count:
            value[0] = 0
        if count > 1:
            value[1, :64] = torch.linspace(-1, 1, 64, device=self.device) * 448
        return value.to(torch.bfloat16)

    def test_strided_paged_cache_and_skipped_rows(self):
        for count, page, padding in (
            (0, 64, 0),
            (7, 64, 128),
            (257, 256, 512),
            (1024, 1024, 256),
        ):
            with self.subTest(count=count, page=page, padding=padding):
                kv = self.kv(count)
                rows = pack_local_kv(kv, self.workspace)
                blocks = max((count + page - 1) // page + 1, 2)
                raw = torch.full(
                    (blocks, page * 584 + padding),
                    173,
                    dtype=torch.uint8,
                    device=self.device,
                )
                expected_raw = raw.clone()
                cache = raw.as_strided((blocks, page, 584), (raw.shape[1], 584, 1))
                expected = expected_raw.as_strided(cache.shape, cache.stride())
                slots = torch.randperm(blocks * page, device=self.device)[
                    :count
                ].contiguous()
                slots[::7] = -1
                quantize_and_insert_k_cache(kv, expected, slots)
                insert_packed_rows(rows, cache, slots)
                self.assertTrue(torch.equal(raw, expected_raw))

    def test_cp_byte_slices_match_full_old_write(self):
        for page in (64, 256, 1024):
            count = min(347, page * 3)
            kv = self.kv(count)
            rows = pack_local_kv(kv, self.workspace)
            slots = torch.randperm(page * 4, device=self.device)[:count].contiguous()
            slots[::11] = -1
            compaction = build_cp_byte_sliced_slot_compaction(
                slots, page, 4, "test.swa_packed", "skip_minus_one"
            )
            for rank in range(4):
                with self.subTest(page=page, rank=rank):
                    raw = torch.full(
                        (4, (page * 584 + 256) // 4),
                        173,
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    expected = raw.clone()
                    kwargs = dict(
                        full_entries_per_block=page,
                        cp_rank=rank,
                        cp_size=4,
                        compaction=compaction,
                    )
                    quantize_and_insert_k_cache_cp_byte_sliced(
                        kv, expected, slots, **kwargs
                    )
                    insert_packed_rows_cp_byte_sliced(rows, raw, **kwargs)
                    self.assertTrue(torch.equal(raw, expected))

    def test_attention_bytes_for_cold_and_continuation_chunks(self):
        for count, prefix_count in (
            (1, 0),
            (127, 1),
            (257, 127),
            (4096, 0),
            (4096, 127),
            (3328, 127),
        ):
            with self.subTest(count=count, prefix=prefix_count):
                kv = self.kv(count)
                prefix = self.kv(prefix_count)
                rows = pack_local_kv(kv, self.workspace)
                actual = build_attention_cache(rows, self.workspace, prefix=prefix)
                combined = torch.cat((prefix, kv))
                expected = torch.empty_like(actual)
                slots = torch.arange(combined.shape[0], device=self.device)
                quantize_and_insert_k_cache(combined, expected, slots)
                actual_rows = gather_k_cache_slots_packed(actual, slots)
                expected_rows = gather_k_cache_slots_packed(expected, slots)
                self.assertTrue(torch.equal(actual_rows, expected_rows))

    def test_all_skipped_byte_slices_leave_pool_untouched(self):
        kv = self.kv(7)
        rows = pack_local_kv(kv, self.workspace)
        slots = torch.full((7,), -1, dtype=torch.int64, device=self.device)
        compaction = build_cp_byte_sliced_slot_compaction(
            slots, 256, 2, "test.swa_empty", "skip_minus_one"
        )
        raw = torch.full(
            (2, 256 * 584 // 4), 173, dtype=torch.uint8, device=self.device
        )
        expected = raw.clone()
        insert_packed_rows_cp_byte_sliced(
            rows,
            raw,
            full_entries_per_block=256,
            cp_rank=2,
            cp_size=4,
            compaction=compaction,
        )
        self.assertTrue(torch.equal(raw, expected))


if __name__ == "__main__":
    unittest.main()
