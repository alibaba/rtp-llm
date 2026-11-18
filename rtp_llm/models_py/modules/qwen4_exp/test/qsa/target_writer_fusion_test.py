import unittest
from itertools import product
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_compressor import (
    restore_indexer_cache,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_decode import (
    write_target_window_with_undo_,
)
from rtp_llm.models_py.modules.qwen4_exp.test.support.target_writer_fusion import (
    TargetWriterFixture,
)

MODULE = "rtp_llm.models_py.modules.qwen4_exp.qsa.target_writer"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TargetWriterFusionTest(TargetWriterFixture, unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() not in ((9, 0), (12, 0)):
            self.skipTest("SM90/SM120 fast path")
        torch.manual_seed(613)

    def write(self, args, fused):
        if fused:
            return write_target_window_with_undo_(*args, norm_eps=1.0e-6)
        with patch(MODULE + ".try_write_target_window", return_value=None):
            return write_target_window_with_undo_(*args, norm_eps=1.0e-6)

    def test_snapshots_writes_and_undo_are_exact(self):
        for page_size, query in product((64, 128), (1, 2, 3, 4)):
            for prefixes in (
                [4],
                [5],
                [6],
                [7],
                [page_size - 1],
                [page_size],
                [4093],
                [32749],
                [5, page_size - 1, 4093],
                list(range(page_size - 4, page_size + 3)),
            ):
                for strided in (False, True):
                    with self.subTest(
                        page_size=page_size,
                        query=query,
                        prefixes=prefixes,
                        strided=strided,
                    ):
                        self.compare(self.case(prefixes, query, strided, page_size))

    def test_dummy_and_invalid_physical_slots_match_reference(self):
        for page_size, query in product((64, 128), (1, 2, 3, 4)):
            args = self.case([5, page_size - 1, page_size], query, page_size=page_size)
            args[6][0].zero_()
            args[8][1].zero_()
            args[8][2].fill_(args[7].shape[0] + 1)
            self.compare(args)

    def test_empty_capacity_declines_without_touching_pools(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa.target_writer import (
            try_write_target_window,
        )

        for index in (5, 6, 7, 8):
            args = self.case([7], 4)
            args[index] = args[index][:0] if index in (5, 7) else args[index][:, :0]
            self.assertIsNone(try_write_target_window(*args, norm_eps=1e-6))

    def test_graph_refreshes_prefix_pages_keys_and_gamma(self):
        for page_size, query in product((64, 128), (1, 2, 3, 4)):
            args = self.case([5, page_size - 3, page_size - 2], query, True, page_size)
            self.write(args, True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                undo = self.write(args, True)
            for residue in (0, 1, 2, 3):
                args[1].copy_(
                    torch.tensor(
                        [4 + residue, page_size - 4 + residue, page_size + residue],
                        device="cuda",
                        dtype=torch.int32,
                    )
                )
                args[0].normal_()
                args[2].normal_()
                args[3].normal_()
                args[4].normal_(0, 0.1)
                args[6][:, 0], args[6][:, 1] = (
                    args[6][:, 1].clone(),
                    args[6][:, 0].clone(),
                )
                args[8][:, 0], args[8][:, 1] = (
                    args[8][:, 1].clone(),
                    args[8][:, 0].clone(),
                )
                initial_kv = args[5].clone()
                initial_state = args[7].clone()
                graph.replay()
                actual_kv = args[5].clone()
                actual_state = args[7].clone()
                restore_indexer_cache(undo)
                self.assertTrue(torch.equal(args[5], initial_kv))
                self.assertTrue(torch.equal(args[7], initial_state))
                self.write(args, False)
                self.assertTrue(torch.equal(args[5], actual_kv))
                self.assertTrue(torch.equal(args[7], actual_state))


if __name__ == "__main__":
    unittest.main()
