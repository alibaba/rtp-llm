"""SM120 mapped-host lookup -> WKV -> injection; missing hardware fails."""

import json
import struct
import tempfile
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv4.v41_engram import V41Engram
from rtp_llm.models_py.modules.dsv4.v41_engram_host import (
    HostEngramEmbedding,
    V41EngramTableSource,
    _PinnedSharedTable,
)
from rtp_llm.models_py.modules.dsv4.v41_engram_inject import gated_engram_residual
from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import V41MXFP8Linear, quantize_group32


class V41EngramHardwareTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "Engram hardware gate requires CUDA")
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        directory = tempfile.TemporaryDirectory(dir="/dev/shm")
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.raw = ((torch.arange(7 * 256).reshape(7, 256) % 33 - 16).float() / 16).to(
            torch.float8_e4m3fn
        )
        self.scales = (torch.arange(56).reshape(7, 8) % 5 + 121).to(torch.uint8)
        payload, scales = (
            self.raw.view(torch.uint8).numpy().tobytes(),
            self.scales.numpy().tobytes(),
        )
        names = ["layers.1.engram.embed.weight", "layers.1.engram.embed.scale"]
        header = json.dumps(
            {
                names[0]: dict(
                    dtype="F8_E4M3", shape=[7, 256], data_offsets=[0, len(payload)]
                ),
                names[1]: dict(
                    dtype="F8_E8M0",
                    shape=[7, 8],
                    data_offsets=[len(payload), len(payload) + len(scales)],
                ),
            }
        ).encode()
        (root / "table.safetensors").write_bytes(
            struct.pack("<Q", len(header)) + header + payload + scales
        )
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {name: "table.safetensors" for name in names}})
        )
        source = V41EngramTableSource.from_checkpoint(root, 1, 7, 256, "a" * 64)
        table = _PinnedSharedTable(source, max_bytes=source.nbytes, directory=root)
        self.addCleanup(table.close)
        table.register_cuda(torch.device("cuda", torch.cuda.current_device()))
        self.embedding = HostEngramEmbedding(table)
        # Constant FP8 payload and varying block scales admit an independent WKV oracle.
        weight = torch.full(
            (25600, 6144), 0.5, dtype=torch.float8_e4m3fn, device="cuda"
        )
        self.w_scales = (
            (torch.arange(800)[:, None] + torch.arange(192)[None, :]) % 5 + 118
        ).to(torch.uint8)
        self.projection = V41MXFP8Linear(weight, self.w_scales.cuda())
        self.q = torch.ones(4, 5120, dtype=torch.bfloat16, device="cuda")
        self.k = (
            torch.tensor([1, -1, 0.5, -0.5], device="cuda", dtype=torch.bfloat16)[
                :, None
            ]
            .expand(4, 5120)
            .contiguous()
        )

    def reference(self, hidden, ids, valid):
        hidden, ids, valid = hidden.cpu(), ids.cpu(), valid.cpu()
        decoded = (
            self.raw.float()
            * torch.exp2(self.scales.float() - 127).repeat_interleave(32, dim=1)
        ).bfloat16()
        live = (ids >= 0) & (ids < 7) & valid[:, None]
        values = torch.zeros((*ids.shape, 256), dtype=torch.bfloat16)
        values[live] = decoded[ids[live]]
        values, scales = quantize_group32(values.flatten(1), backend="eager")
        sums = values.float().reshape(ids.shape[0], 192, 32).sum(-1) * 0.5
        scaled = sums * torch.exp2(scales.float() - 127)
        projected = (
            (scaled @ torch.exp2(self.w_scales.float() - 127).T)
            .repeat_interleave(32, dim=1)
            .bfloat16()
        )
        return gated_engram_residual(
            hidden,
            projected,
            self.q.cpu(),
            self.k.cpu(),
            valid,
            eps=1e-20,
            backend="eager",
        )

    def test_real_lookup_projection_injection_with_bounded_chunks_and_alias(self):
        for tokens in (1, 17):
            hidden = torch.randn(tokens, 4, 5120, device="cuda", dtype=torch.bfloat16)
            ids = (
                (torch.arange(tokens * 48, device="cuda").reshape(tokens, 48) % 9) - 1
            )[:, ::2]
            valid = torch.arange(tokens, device="cuda") % 3 != 0
            valid[-1] = True
            hidden[~valid] = -0.0
            expected = self.reference(hidden, ids, valid)
            for chunk in (1, 8, 128):
                op = V41Engram(
                    self.embedding,
                    self.projection,
                    self.q,
                    self.k,
                    eps=1e-20,
                    chunk_tokens=chunk,
                )
                output = op(hidden, ids, valid)
                torch.testing.assert_close(output.cpu(), expected, rtol=8e-3, atol=8e-3)
                torch.testing.assert_close(
                    output[~valid].view(torch.uint8), hidden[~valid].view(torch.uint8)
                )
            alias = hidden.clone()
            with torch.inference_mode():
                self.assertIs(op(alias, ids, valid, out=alias), alias)
            torch.testing.assert_close(alias.cpu(), expected, rtol=8e-3, atol=8e-3)


if __name__ == "__main__":
    unittest.main()
