"""Exact hash/ownership gates; no model math or golden changes."""

from types import SimpleNamespace
from unittest import TestCase, main, skipUnless
from unittest.mock import patch

import torch

from rtp_llm.multimodal.mm_work_item import prepare_batch_feature_hashes
from rtp_llm.ops import get_multimodal_feature_hash, get_multimodal_feature_hashes


def item(defer=False, hashes=None):
    return SimpleNamespace(defer_feature_hashes=defer, feature_hashes=hashes)


class FeatureHashBatchCpuTest(TestCase):
    def test_cpu_empty_single_mixed_shapes_and_ownership(self):
        self.assertEqual(get_multimodal_feature_hashes([]), [])
        values = [
            torch.arange(39, dtype=torch.uint8).reshape(3, 13),
            torch.arange(24, dtype=torch.float32).reshape(4, 6).t(),
            torch.randn(184, 32, dtype=torch.bfloat16),
        ]
        for batch in (values[:1], values):
            got = get_multimodal_feature_hashes(batch)
            for source, result in zip(batch, got):
                self.assertTrue(
                    torch.equal(result, get_multimodal_feature_hash(source))
                )
                self.assertEqual(result.device.type, "cpu")
                self.assertEqual(result.dtype, torch.int32)
                self.assertEqual(result.untyped_storage().nbytes(), result.numel() * 4)

    def test_invalid_inputs_still_fail(self):
        for invalid in (torch.empty(0, 4), torch.empty(3, 0), torch.tensor(1.0)):
            with self.assertRaises(RuntimeError):
                get_multimodal_feature_hashes([torch.ones(2, 4), invalid])

    def test_cpu_and_deferred_items_keep_existing_path(self):
        rows = [item(), item(defer=True), item(hashes=[torch.tensor([7])])]
        outputs = [(torch.ones(2, 4), [], [])] * 3
        with patch(
            "rtp_llm.multimodal.mm_work_item.get_multimodal_feature_hashes"
        ) as op:
            prepare_batch_feature_hashes(rows, outputs)
        op.assert_not_called()
        self.assertIsNone(rows[0].feature_hashes)
        self.assertIsNone(rows[1].feature_hashes)
        self.assertEqual(rows[2].feature_hashes[0].item(), 7)


@skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class FeatureHashBatchCudaTest(TestCase):
    def check_batch(self, values):
        originals = [value.cpu().clone() for value in values]
        expected = [get_multimodal_feature_hash(value) for value in values]
        got = get_multimodal_feature_hashes(values)
        self.assertEqual(len(got), len(values))
        for source, before, old_hash, result in zip(values, originals, expected, got):
            self.assertTrue(
                torch.equal(source.cpu().view(torch.uint8), before.view(torch.uint8))
            )
            self.assertTrue(torch.equal(result, old_hash))
            self.assertTrue(torch.equal(result, get_multimodal_feature_hash(before)))
            self.assertEqual(result.device.type, "cpu")
            self.assertEqual(result.dtype, torch.int32)
            self.assertEqual(result.untyped_storage().nbytes(), result.numel() * 4)
        self.assertEqual(len({x.untyped_storage().data_ptr() for x in got}), len(got))

    def test_same_device_batches_and_realistic_rows(self):
        torch.manual_seed(42)
        for count in (1, 2, 8, 128):
            values = [
                torch.randn(
                    184 if i % 2 else 968, 4096, device="cuda", dtype=torch.bfloat16
                )
                for i in range(count)
            ]
            self.check_batch(values)

    def test_tail_bytes_noncontiguous_and_mixed_devices(self):
        values = [
            torch.arange(39, dtype=torch.uint8, device="cuda").reshape(3, 13),
            torch.arange(24, dtype=torch.float32, device="cuda").reshape(4, 6).t(),
            torch.randn(5, 19, device="cuda", dtype=torch.float16),
        ]
        # Keep non-contiguous byte comparison explicit; view(uint8) needs a contiguous last axis.
        expected = [get_multimodal_feature_hash(x.cpu()) for x in values]
        got = get_multimodal_feature_hashes(values)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(expected, got)))
        values.insert(1, torch.randn(7, 9))
        got = get_multimodal_feature_hashes(values)
        self.assertTrue(
            all(
                torch.equal(x, get_multimodal_feature_hash(v.cpu()))
                for x, v in zip(got, values)
            )
        )

    def test_current_stream_completion_and_retained_results(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            values = [torch.ones(184, 32, device="cuda") for _ in range(4)]
            for index, value in enumerate(values):
                value.mul_(index + 2)
            results = get_multimodal_feature_hashes(values)
            # No synchronize by the test before inspecting the CPU results.
            for index, result in enumerate(results):
                self.assertTrue(
                    torch.equal(
                        result,
                        get_multimodal_feature_hash(
                            torch.full((184, 32), float(index + 2))
                        ),
                    )
                )
            snapshots = [result.clone() for result in results]
            for value in values:
                value.zero_()
            get_multimodal_feature_hashes(values)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(results, snapshots)))

    def test_work_item_order_deferred_and_existing_hashes(self):
        values = [torch.randn(5 + i, 16, device="cuda") for i in range(5)]
        rows = [item(), item(defer=True), item(hashes=[torch.tensor([123])]), item()]
        outputs = [
            (values[:2], [], []),
            (values[2], [], []),
            (values[3], [], []),
            (values[4], [], []),
        ]
        prepare_batch_feature_hashes(rows, outputs)
        for value, hashed in zip(values[:2], rows[0].feature_hashes):
            self.assertTrue(torch.equal(hashed, get_multimodal_feature_hash(value)))
        self.assertIsNone(rows[1].feature_hashes)
        self.assertEqual(rows[2].feature_hashes[0].item(), 123)
        self.assertTrue(
            torch.equal(
                rows[3].feature_hashes[0], get_multimodal_feature_hash(values[4])
            )
        )


if __name__ == "__main__":
    main()
