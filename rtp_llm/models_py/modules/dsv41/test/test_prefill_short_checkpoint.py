"""Full prefill must retain canonical history across a short checkpoint chunk."""

import dataclasses
import unittest

import test_prefill as fixture
import torch
from rtp_llm.models_py.modules.dsv41.ced import ReplayMode


class ShortCheckpointGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.PrefillGpuTest.setUpClass()
        cls.helper = fixture.PrefillGpuTest()

    @classmethod
    def tearDownClass(cls):
        fixture.PrefillGpuTest.tearDownClass()

    @torch.inference_mode()
    def test_full_short_checkpoint_chunks_and_restored_continuation(self):
        boundary = self.helper.layout.reuse_unit
        self.assertEqual(boundary, 1024)
        for chunk, last_count in ((341, 1), (511, 2)):
            with self.subTest(chunk=chunk):
                cache = self.helper.cache(boundary + 3, ReplayMode.FULL)
                plan = self.helper.plan(cache, boundary + 3, chunk)
                checkpoint_extend = next(
                    item for item in plan.extends if item.checkpoint_end == boundary
                )
                self.assertEqual(len(checkpoint_extend.encoder_rows), last_count)
                runner, _ = self.helper.execute(cache, plan)
                snapshot = runner.protected
                self.assertEqual(snapshot.checkpoint.materialized_end, boundary)
                expected, _ = self.helper.rows(boundary - 3, boundary)
                for field in dataclasses.fields(expected):
                    self.helper.equal(
                        getattr(snapshot.history_rows, field.name),
                        getattr(expected, field.name),
                    )
                saved = {
                    field.name: getattr(snapshot.history_rows, field.name).clone()
                    for field in dataclasses.fields(expected)
                }
                receiver = self.helper.cache(
                    2 * boundary + 3, ReplayMode.FULL, request="continued"
                )
                continuation = self.helper.plan(
                    receiver, 2 * boundary + 3, chunk, restored=snapshot
                )
                resumed, _ = self.helper.execute(
                    receiver, continuation, restored=snapshot
                )
                self.assertEqual(
                    resumed.protected.checkpoint.materialized_end, 2 * boundary
                )
                for field in dataclasses.fields(expected):
                    self.helper.equal(
                        getattr(resumed.protected.history_rows, field.name),
                        getattr(expected, field.name),
                    )
                    self.helper.equal(
                        getattr(snapshot.history_rows, field.name), saved[field.name]
                    )
                self.assertIsNone(runner.tail)
                self.assertIsNone(resumed.tail)
                self.helper.records.append(
                    {
                        "test": self.id(),
                        "chunk": chunk,
                        "last_checkpoint_chunk_rows": last_count,
                        "protected_end": boundary,
                        "restored_protected_end": 2 * boundary,
                        "canonical_history_exact": True,
                    }
                )


if __name__ == "__main__":
    unittest.main()
