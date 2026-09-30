"""SM120 mapped-host lookup and graph gate; missing hardware is a failure."""

import json
import struct
import tempfile
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv4.v41_engram_host import (
    HostEngramEmbedding,
    V41EngramTableSource,
    _PinnedSharedTable,
)


class V41EngramLookupHardwareTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "Engram hardware gate requires CUDA")
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        temporary = tempfile.TemporaryDirectory(dir="/dev/shm")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        # Cover all E4M3 encodings against zero, subnormal, finite and NaN scales.
        self.raw = torch.arange(256).to(torch.uint8).repeat(9, 1)
        self.scales = (
            torch.tensor(
                [[0], [1], [2], [120], [126], [127], [128], [254], [255]],
                dtype=torch.uint8,
            )
            .expand(9, 8)
            .contiguous()
        )
        weights, scales = self.raw.numpy().tobytes(), self.scales.numpy().tobytes()
        names = ["layers.1.engram.embed.weight", "layers.1.engram.embed.scale"]
        header = json.dumps(
            {
                names[0]: dict(
                    dtype="F8_E4M3", shape=[9, 256], data_offsets=[0, len(weights)]
                ),
                names[1]: dict(
                    dtype="F8_E8M0",
                    shape=[9, 8],
                    data_offsets=[len(weights), len(weights) + len(scales)],
                ),
            }
        ).encode()
        (root / "table.safetensors").write_bytes(
            struct.pack("<Q", len(header)) + header + weights + scales
        )
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {name: "table.safetensors" for name in names}})
        )
        source = V41EngramTableSource.from_checkpoint(root, 1, 9, 256, "a" * 64)
        self.table = _PinnedSharedTable(source, max_bytes=source.nbytes, directory=root)
        self.addCleanup(self.table.close)
        self.table.register_cuda(torch.device("cuda", torch.cuda.current_device()))
        self.embedding = HostEngramEmbedding(self.table)

    def assert_result(self, indices, valid_rows, output):
        indices, valid_rows = indices.cpu(), valid_rows.cpu()
        live = (indices >= 0) & (indices < 9) & valid_rows[:, None]
        expected = torch.zeros((*indices.shape, 256), dtype=torch.bfloat16)
        # This uses Torch's checkpoint dtypes, independently of the lookup codec.
        values = self.raw[indices[live]].view(torch.float8_e4m3fn).float()
        scales = self.scales[indices[live]].view(torch.float8_e8m0fnu).float()
        expected[live] = (values * scales.repeat_interleave(32, dim=-1)).bfloat16()
        actual = output.cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        zero = expected == 0
        torch.testing.assert_close(
            torch.signbit(actual[zero]), torch.signbit(expected[zero])
        )

    def test_mapped_host_lookup_special_codes_strides_tails_and_mask(self):
        for dtype in (torch.int32, torch.int64):
            indices = (
                torch.arange(17 * 48, device="cuda", dtype=dtype).reshape(17, 48) % 11
                - 1
            )[:, ::2]
            valid_rows = torch.arange(17, device="cuda") % 3 != 0
            self.assert_result(indices, valid_rows, self.embedding(indices, valid_rows))
        empty = self.embedding(
            torch.empty(0, 24, dtype=torch.int32, device="cuda"),
            torch.empty(0, dtype=torch.bool, device="cuda"),
        )
        self.assertEqual(empty.shape, (0, 24, 256))

    def test_graph_replay_updates_lookup_ids_and_dead_rows(self):
        indices = torch.zeros(17, 24, dtype=torch.int64, device="cuda")
        valid_rows = torch.ones(17, dtype=torch.bool, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.embedding(indices, valid_rows)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                output = self.embedding(indices, valid_rows)
            for row, live in ((0, True), (5, True), (8, False), (8, True), (-1, True)):
                indices.fill_(row)
                valid_rows.fill_(live)
                valid_rows[1] = False
                graph.replay()
                self.assert_result(indices, valid_rows, output)
        finally:
            torch.cuda.synchronize()
            graph.reset()


if __name__ == "__main__":
    unittest.main()
