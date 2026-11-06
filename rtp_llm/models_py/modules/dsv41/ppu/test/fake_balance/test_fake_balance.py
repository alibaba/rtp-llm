"""Route invariants and dispatch integration, independent of native kernels."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu.moe import PPUMoE


def engine(rank=0, enabled=True, backend="grouped"):
    with patch.dict(os.environ, {"RTP_V41_PPU_MOE_BACKEND": backend}):
        result = PPUMoE(ep_size=8, ep_rank=rank)
    result.configure_fake_balance(enabled, top_k=6, dp_rank=rank, dp_size=8)
    return result


def routes(rows=8):
    ids = torch.full((rows, 8), -1, dtype=torch.int64)
    ids[:, :6] = torch.arange(6)
    weights = torch.zeros((rows, 8), dtype=torch.float32)
    weights[:, :6] = torch.arange(1, 7).float() / 20
    return ids, weights


class FakeBalanceTest(unittest.TestCase):
    def test_default_and_explicit_off_preserve_objects(self):
        for value in (PPUMoE(ep_size=8), engine(enabled=False)):
            ids, weights = routes()
            actual = value._apply_fake_balance(ids, weights)
            self.assertIs(actual[0], ids)
            self.assertIs(actual[1], weights)

    def test_global_load_and_distinct_experts_for_decode_batches(self):
        for rows in (1, 3, 8, 16, 32, 64, 128, 256):
            counts = torch.zeros(384, dtype=torch.int64)
            for rank in range(8):
                ids, weights = engine(rank)._apply_fake_balance(*routes(rows))
                self.assertTrue(torch.all(ids[:, 6:] == -1))
                self.assertTrue(torch.all(weights[:, 6:] == 0))
                self.assertTrue(torch.all(weights[:, :6] == 1))
                for row in ids[:, :6]:
                    self.assertEqual(len(set(row.tolist())), 6)
                counts += torch.bincount(ids[:, :6].flatten(), minlength=384)
            self.assertEqual(int(counts.sum()), 8 * rows * 6)
            # Each EP rank receives the same number of valid assignments.
            self.assertEqual(counts.reshape(8, 48).sum(1).tolist(), [rows * 6] * 8)
            self.assertLessEqual(int(counts.max() - counts.min()), 1)

    def test_masks_noncontiguous_storage_and_input_preservation(self):
        for dtype in (torch.int32, torch.int64):
            source_ids, source_weights = routes(16)
            source_ids = source_ids.to(dtype)
            source_ids[4:8] = -1
            source_weights[4:8] = 0
            ids, weights = source_ids[::2], source_weights[::2]
            before = ids.clone(), weights.clone()
            actual_ids, actual_weights = engine(3)._apply_fake_balance(ids, weights)
            self.assertEqual(actual_ids.dtype, dtype)
            self.assertTrue(torch.equal(actual_ids[2:4], before[0][2:4]))
            self.assertTrue(torch.all(actual_weights[2:4] == 0))
            self.assertTrue(torch.equal(ids, before[0]))
            self.assertTrue(torch.equal(weights, before[1]))

    def test_empty_and_idle_owners_keep_zero_valid_routes(self):
        for rows in (0, 1, 8):
            ids, weights = routes(rows)
            ids.fill_(-1)
            weights.zero_()
            actual_ids, actual_weights = engine()._apply_fake_balance(ids, weights)
            self.assertTrue(torch.all(actual_ids == -1))
            self.assertTrue(torch.all(actual_weights == 0))

    def test_override_reaches_both_backend_boundaries(self):
        for backend, method in (
            ("megamoe", "_forward_megamoe"),
            ("grouped", "_forward_deepep"),
        ):
            value = engine(backend=backend)
            ids, weights = routes()
            expected = value._apply_fake_balance(ids, weights)
            with patch.object(
                value, "_use_deepep_dispatch", return_value=True
            ), patch.object(value, method, return_value="result") as dispatch:
                self.assertEqual(
                    value.forward(torch.zeros(8, 5120), ids, weights, layer_id=0),
                    "result",
                )
            torch.testing.assert_close(dispatch.call_args.args[1], expected[0])
            torch.testing.assert_close(dispatch.call_args.args[2], expected[1])

    def test_configuration_is_explicit_validated_and_frozen(self):
        value = engine()
        value.configure_fake_balance(True, top_k=6, dp_rank=0, dp_size=8)
        with self.assertRaisesRegex(ValueError, "frozen"):
            value.configure_fake_balance(False, top_k=6, dp_rank=0, dp_size=8)
        for enabled, top_k, rank, size in (
            (1, 6, 0, 8),
            (True, 0, 0, 8),
            (True, 6, 8, 8),
        ):
            with self.assertRaises(ValueError):
                PPUMoE(ep_size=8).configure_fake_balance(
                    enabled, top_k=top_k, dp_rank=rank, dp_size=size
                )


if __name__ == "__main__":
    unittest.main()
