"""SM120 Engram integer hash and graph gate; missing hardware is a failure."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.v41_engram_hash import (
    V41EngramHasher,
    V41EngramLayout,
)


class V41EngramHashHardwareTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "Engram hardware gate requires CUDA")
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        layout = V41EngramLayout.from_config(
            dict(
                engram_layer_ids=[1, 14],
                engram_num_embeddings=[384006168, 384016682],
                engram_max_ngram_size=4,
                engram_n_heads=8,
                engram_head_dim=256,
                engram_compressed_vocab_size=99092,
                engram_vocab_size=16000000,
            ),
            pad_token_id=2,
        )
        mapping = list(range(99092)) + [i % 99092 for i in range(129280 - 99092)]
        self.reference = V41EngramHasher(layout, mapping)
        self.gpu = V41EngramHasher(layout, mapping).cuda()

    def assert_result(self, windows, result):
        reference = self.reference(windows.cpu(), backend="eager")
        torch.testing.assert_close(result.ids.cpu(), reference.ids, rtol=0, atol=0)
        torch.testing.assert_close(result.valid_rows.cpu(), reference.valid_rows)

    def test_device_hashes_are_bit_exact_for_both_integer_widths(self):
        for dtype in (torch.int32, torch.int64):
            windows = torch.randint(0, 129280, (257, 4), dtype=dtype, device="cuda")
            windows[0] = -1
            windows[1, 1] = -1
            windows[2, 2] = -1
            windows[3, 3] = -1
            windows[4, 0] = 129280
            self.assert_result(windows, self.gpu(windows))

    def test_graph_replay_updates_hashes_and_fake_row_mask(self):
        windows = torch.zeros(65, 4, dtype=torch.int32, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.gpu(windows)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self.gpu(windows)
        for current in (11, -1, 99091):
            windows.fill_(current)
            windows[1, 1] = -1
            graph.replay()
            self.assert_result(windows, output)


if __name__ == "__main__":
    unittest.main()
