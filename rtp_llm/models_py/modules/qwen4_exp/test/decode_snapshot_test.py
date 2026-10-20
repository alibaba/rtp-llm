import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_decode_triton import (
    write_decode_key_with_undo_,
)
from rtp_llm.models_py.modules.qwen4_exp.test import (
    target_writer_fusion_test as fixtures,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DecodeSnapshotTest(unittest.TestCase):
    case = fixtures.TargetWriterFusionTest.case
    compare = fixtures.TargetWriterFusionTest.compare

    def setUp(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("SM120 fast path")
        torch.manual_seed(20261019)

    def write(self, args, fused):
        with patch.dict("os.environ", {"QWEN4_QSA_DECODE_SNAPSHOT": str(int(fused))}):
            return write_decode_key_with_undo_(*args, norm_eps=1.0e-6)

    def test_complete_writer_and_rollback_match_at_every_residue_and_page_edge(self):
        for positions in (
            [0],
            [1],
            [2],
            [3],
            [127],
            [128],
            [4093],
            [32749],
            [124, 125, 126, 127, 128, 129, 130],
        ):
            self.compare(self.case(positions, 1))
        args = self.case([5, 127, 128], 1)
        args[6][0].zero_()
        args[8][1].zero_()
        args[8][2].fill_(args[7].shape[0] + 1)
        self.compare(args)

    def test_snapshot_int64_arithmetic_and_unsupported_layout_fallback(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa.decode_snapshot_triton import (
            try_snapshot_decode,
        )

        args = self.case([0, 127, 128, 4095], 1)
        args[1].copy_(
            torch.tensor(
                [-2147483648, -1, 2147483646, 2147483647],
                device="cuda",
                dtype=torch.int32,
            )
        )
        path = "rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_decode_triton.write_decode_key_"
        with patch(path):
            baseline = self.write(args, False)
            candidate = self.write(args, True)
        for name in ("state_slots", "kv_slots", "original_state", "original_kv"):
            self.assertTrue(
                torch.equal(getattr(baseline, name), getattr(candidate, name)), name
            )
        with patch.dict("os.environ", {"QWEN4_QSA_DECODE_SNAPSHOT": "1"}):
            for positions in (args[1].long(), args[1][::2], args[1][:0]):
                self.assertIsNone(
                    try_snapshot_decode(positions, args[5], args[6], args[7], args[8])
                )

    def test_graph_replay_refreshes_all_snapshots_before_original_writer(self):
        for batch in (1, 3, 8, 32):
            args = self.case([128] * batch, 1)
            saved_kv, saved_state = args[5].clone(), args[7].clone()
            self.write(args, True)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    actual = self.write(args, True)
                for generation in range(32):
                    args[1].copy_(
                        torch.tensor(
                            [124 + (generation + row) % 12 for row in range(batch)],
                            device="cuda",
                            dtype=torch.int32,
                        )
                    )
                    args[0].normal_()
                    args[5].copy_(saved_kv)
                    args[7].copy_(saved_state)
                    expected = self.write(args, False)
                    after_kv, after_state = args[5].clone(), args[7].clone()
                    args[5].copy_(saved_kv)
                    args[7].copy_(saved_state)
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertTrue(torch.equal(args[5], after_kv))
                    self.assertTrue(torch.equal(args[7], after_state))
                    for name in (
                        "state_slots",
                        "kv_slots",
                        "original_state",
                        "original_kv",
                    ):
                        self.assertTrue(
                            torch.equal(getattr(actual, name), getattr(expected, name)),
                            name,
                        )
            finally:
                graph.reset()


if __name__ == "__main__":
    unittest.main()
