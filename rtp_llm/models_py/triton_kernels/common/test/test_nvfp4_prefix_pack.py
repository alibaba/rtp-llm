import unittest

import torch

from rtp_llm.models_py.triton_kernels.common.nvfp4_prefix_pack import pack_prefix_pools


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestPrefixPack(unittest.TestCase):
    def test_bytes_empty_repeated_ids_and_strides(self):
        for padding in (0, 256):
            main = torch.randint(
                0, 256, (16, 65536 + padding), device="cuda", dtype=torch.uint8
            )[:, :65536]
            side = torch.randint(
                0, 256, (16, 17408 + padding), device="cuda", dtype=torch.uint8
            )[:, :17408]
            for dtype in (torch.int32, torch.int64):
                for values in ([], [0], [12, 0, 12, 8, 3, 15]):
                    with self.subTest(padding=padding, dtype=dtype, values=values):
                        ids = torch.tensor(values, dtype=dtype, device="cuda")
                        actual = pack_prefix_pools(main, side, ids)
                        for source, output in zip((main, side), actual):
                            self.assertTrue(output.is_contiguous())
                            self.assertTrue(
                                torch.equal(output, source.index_select(0, ids))
                            )

    def test_graph_refreshes_ids_and_bytes(self):
        main = torch.randint(0, 256, (16, 65536), device="cuda", dtype=torch.uint8)
        side = torch.randint(0, 256, (16, 17408), device="cuda", dtype=torch.uint8)
        ids = torch.tensor([1, 7, 7, 0], dtype=torch.int64, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                pack_prefix_pools(main, side, ids)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = pack_prefix_pools(main, side, ids)
        pointers = tuple(output.data_ptr() for output in outputs)
        for values in ([3, 15, 0, 3], [8, 8, 9, 11]):
            ids.copy_(torch.tensor(values, dtype=ids.dtype, device=ids.device))
            main.random_(0, 256)
            side.random_(0, 256)
            graph.replay()
            self.assertEqual(pointers, tuple(output.data_ptr() for output in outputs))
            for source, output in zip((main, side), outputs):
                self.assertTrue(torch.equal(output, source.index_select(0, ids)))

    def test_rejects_incompatible_host_geometry(self):
        main = torch.zeros((2, 65536), device="cuda", dtype=torch.uint8)
        side = torch.zeros((2, 17408), device="cuda", dtype=torch.uint8)
        ids = torch.tensor([0], device="cuda", dtype=torch.int64)
        for bad_main, bad_side, bad_ids in (
            (main[:, :-1], side, ids),
            (main, side[:, :-1], ids),
            (main, side, ids.cpu()),
            (main, side, ids.float()),
        ):
            with self.assertRaises(ValueError):
                pack_prefix_pools(bad_main, bad_side, bad_ids)


if __name__ == "__main__":
    unittest.main()
