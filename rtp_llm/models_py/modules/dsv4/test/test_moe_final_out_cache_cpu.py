"""Production MoE cache ownership and geometric-capacity regression (CPU)."""

import unittest
import weakref
from contextlib import ExitStack
from unittest import mock

import torch
from moe_cache_test_support import moe_layer


class FinalOutCacheTest(unittest.TestCase):
    def setUp(self):
        self._patches = ExitStack()
        self.addCleanup(self._patches.close)
        self._patches.enter_context(
            mock.patch.object(moe_layer, "_FINAL_OUT_CACHE", {})
        )
        self._patches.enter_context(
            mock.patch.object(moe_layer, "_FINAL_OUT_RETIRED", [])
        )

    def get(self, capacity, dim=8, dtype=torch.float32):
        return moe_layer._get_or_create_final_out(
            capacity, dim, dtype, torch.device("cpu")
        )

    def test_same_capacity_reuses_buffer_without_allocating(self):
        first = self.get(4)
        with mock.patch.object(
            torch, "empty", side_effect=AssertionError("unexpected allocation")
        ):
            self.assertIs(self.get(4), first)
            self.assertIs(self.get(1), first)

    def test_replacement_retains_the_old_owner(self):
        old = self.get(4)
        old.fill_(17)
        ref = weakref.ref(old)
        ptr = old.data_ptr()
        del old
        new = self.get(64)
        self.assertIsNotNone(ref(), "graph-captured allocation lost its owner")
        self.assertEqual(ref().data_ptr(), ptr)
        self.assertNotEqual(new.data_ptr(), ptr)
        torch.testing.assert_close(ref(), torch.full((4, 8), 17.0))

    def test_incremental_growth_has_geometric_memory_bound(self):
        for requested in range(1, 258):
            current = self.get(requested)
            retired = moe_layer._FINAL_OUT_RETIRED
            total = current.numel() + sum(x.numel() for x in retired)
            self.assertGreaterEqual(current.size(0), requested)
            self.assertLess(current.size(0), 2 * requested)
            self.assertLess(total, 2 * current.numel())
        self.assertLessEqual(len(moe_layer._FINAL_OUT_RETIRED), 9)

    def test_large_jumps_preserve_the_same_bound(self):
        for requested in (3, 5, 17, 1025, 1026, 4097):
            current = self.get(requested)
            total = current.numel() + sum(
                x.numel() for x in moe_layer._FINAL_OUT_RETIRED
            )
            self.assertLess(total, 2 * current.numel())
            self.assertLess(current.size(0), 2 * requested)

    def test_cache_keys_keep_dtype_and_dimension_separate(self):
        a = self.get(4, 8, torch.float32)
        b = self.get(4, 16, torch.float32)
        c = self.get(4, 8, torch.float64)
        self.assertEqual(len(moe_layer._FINAL_OUT_CACHE), 3)
        self.assertEqual(len({a.data_ptr(), b.data_ptr(), c.data_ptr()}), 3)
        self.assertIs(self.get(2, 16), b)

    def test_empty_request_still_has_positive_capacity(self):
        self.assertEqual(self.get(0).size(0), 1)

    def test_allocation_failure_preserves_existing_ownership(self):
        old = self.get(4)
        with mock.patch.object(
            torch, "empty", side_effect=RuntimeError("synthetic allocation failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic allocation failure"):
                self.get(64)
        self.assertIs(self.get(4), old)
        self.assertEqual(moe_layer._FINAL_OUT_RETIRED, [])


if __name__ == "__main__":
    unittest.main()
