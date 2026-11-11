"""Check which sparse decode kernel a service instance selects."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from rtp_llm.device.device_type import is_ppu


@unittest.skipUnless(is_ppu(), "requires a PPU")
class SparseTritonGateTest(unittest.TestCase):
    def op(self, env=None):
        from rtp_llm.models_py.modules.dsv4.ppu.attention.ppu_flash_mla import (
            PpuSparseAttnV4DecodeFp8Op,
        )

        if env is None:
            return PpuSparseAttnV4DecodeFp8Op(64, 512, 512 ** -0.5)
        with patch.dict(os.environ, {"DSV4_PPU_SPARSE_TRITON": env}):
            return PpuSparseAttnV4DecodeFp8Op(64, 512, 512 ** -0.5)

    def call(self, op, q, kv_cache):
        from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op import (
            SparseAttnV4DecodeFp8Op,
        )

        # The base implementation is the compiled FlashMLA path; report which
        # one ran without launching either kernel.
        with patch(
            "rtp_llm.models_py.kernels.ppu.attention.ppu_sparse_attention.forward",
            return_value="triton",
        ) as triton, patch.object(
            SparseAttnV4DecodeFp8Op, "_forward_flash_mla", return_value="compiled"
        ) as compiled:
            out = op._forward_flash_mla(
                q,
                kv_cache,
                torch.zeros(64),
                torch.zeros((2, 1, 128), dtype=torch.int32),
                SimpleNamespace(have_initialized=True),
            )
        return out, triton, compiled

    def test_env_selects_kernel_once_at_construction(self):
        q = torch.zeros((2, 1, 64, 512), dtype=torch.bfloat16)
        kv = torch.zeros((8, 64, 584), dtype=torch.uint8)
        for env, expected in (("1", "triton"), ("0", "compiled")):
            op = self.op(env)
            self.assertEqual(op.triton_sparse, expected == "triton")
            # A later environment change must not switch an instance that may
            # already own a captured Graph.
            with patch.dict(
                os.environ, {"DSV4_PPU_SPARSE_TRITON": "1" if env == "0" else "0"}
            ):
                out, triton, compiled = self.call(op, q, kv)
            self.assertEqual(out, expected)
            self.assertEqual(triton.called, expected == "triton")
            self.assertEqual(compiled.called, expected == "compiled")

    def test_qualified_shape_is_the_only_triton_path(self):
        kv = torch.zeros((8, 64, 584), dtype=torch.uint8)
        for shape in ((2, 1, 64, 512), (2, 1, 32, 512), (2, 1, 64, 256)):
            out, triton, _ = self.call(
                self.op("1"), torch.zeros(shape, dtype=torch.bfloat16), kv
            )
            self.assertEqual(out, "triton" if shape == (2, 1, 64, 512) else "compiled")
        # A page count outside the native set already uses the compiled path.
        out, _, _ = self.call(
            self.op("1"),
            torch.zeros((2, 1, 64, 512), dtype=torch.bfloat16),
            torch.zeros((8, 63, 584), dtype=torch.uint8),
        )
        self.assertEqual(out, "compiled")


if __name__ == "__main__":
    unittest.main()
