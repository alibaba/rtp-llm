"""Batch decoder ABI and explicit native legacy-row byte equivalence.

CPU doubles only the device provider. The PPU test covers every combination
of FP8 payload byte and UE8M0 byte, comparing complete BF16 storage against
the existing segment kernel, including NaNs, signed zero and untouched gaps.
"""

from contextlib import ExitStack
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import _quant


class _Kernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args):
            self.calls.append((grid, args))
        return launch


class CP8SWABatchKernelContracts(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(torch.no_grad())
        self.kernel = _Kernel()
        self.stack.enter_context(patch.object(_quant, "_ppu_fast_path", return_value=True))
        self.stack.enter_context(patch.object(
            _quant, "_triton_kernels", return_value=NS(dequant_v41_swa_batch=self.kernel)
        ))

    def operands(self):
        descriptors = torch.zeros(3, 6, dtype=torch.int64)
        base = torch.full((20, 520), 1, dtype=torch.bfloat16)
        out = base[:, :512]
        packed = torch.zeros(4, 528, dtype=torch.uint8)
        return descriptors, out, packed.view(torch.float8_e4m3fn)

    def test_single_two_dimensional_launch_preserves_strided_output_and_typed_anchor(self):
        descriptors, out, reference = self.operands()
        before = out.clone()
        self.assertTrue(_quant.try_dequantize_swa_batch(
            descriptors, out, 11, reference=reference,
        ))
        self.assertEqual(len(self.kernel.calls), 1)
        grid, args = self.kernel.calls[0]
        self.assertEqual(grid, (11, 3))
        self.assertIs(args[0], reference)
        self.assertIs(args[1], descriptors)
        self.assertIs(args[2], out)
        self.assertEqual(args[3:], (520, 528, 512, 32))
        self.assertTrue(torch.equal(out, before))

    def test_unsupported_abi_alias_and_grad_decline_without_launch_or_mutation(self):
        descriptors, out, reference = self.operands()
        before = out.clone()
        for table, target, maximum, anchor in (
            (descriptors.float(), out, 11, reference),
            (descriptors[:, :5], out, 11, reference),
            (descriptors.t(), out, 11, reference),
            (descriptors[:0], out, 11, reference),
            (descriptors, out.float(), 11, reference),
            (descriptors, out[:, ::2], 11, reference),
            (descriptors, out, 0, reference),
            (descriptors, out, True, reference),
            (descriptors, out, 11, reference.view(torch.uint8)),
            (descriptors, out, 11, reference[:0]),
        ):
            self.assertFalse(_quant.try_dequantize_swa_batch(
                table, target, maximum, reference=anchor,
            ))
        storage = torch.zeros(9000, dtype=torch.uint8)
        alias_out = storage[:8192].view(torch.bfloat16).view(8, 512)
        alias_anchor = storage[:528].view(torch.float8_e4m3fn).view(1, 528)
        self.assertFalse(_quant.try_dequantize_swa_batch(
            descriptors, alias_out, 1, reference=alias_anchor,
        ))
        with torch.enable_grad():
            self.assertFalse(_quant.try_dequantize_swa_batch(
                descriptors, out, 11, reference=reference,
            ))
        self.assertFalse(self.kernel.calls)
        self.assertTrue(torch.equal(out, before))


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1",
    "explicit PPU device test",
)
class CP8SWABatchKernelDevice(unittest.TestCase):
    def test_every_payload_and_scale_byte_matches_legacy_with_disjoint_offsets(self):
        from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_metadata import _upload

        with torch.no_grad():
            self.assertTrue(_quant.is_ppu())
            packed = torch.empty(256, 528, device="cuda", dtype=torch.uint8)
            packed[:, :512] = (torch.arange(512, device="cuda") % 256).to(torch.uint8)
            packed[:, 512:] = torch.arange(256, device="cuda", dtype=torch.uint8)[:, None]
            # First two descriptors cover all 256*256 payload/scale pairs.
            # The rest exercise empty-tail, empty-fresh and fresh-only access.
            tail, fresh = packed[:70], packed[70:]
            empty = packed[:0]
            segments = (
                (tail, fresh, 0, 93, 17),
                (tail, fresh, 93, 163, 131),
                (empty, packed[11:20], 0, 9, 319),
                (packed[21:48], empty, 4, 13, 347),
                (tail, fresh, 72, 17, 380),
            )
            actual = torch.empty(420, 520, device="cuda", dtype=torch.bfloat16)
            actual.view(torch.int16).fill_(0x3EED)
            expected = actual.clone()
            descriptors = []
            before = packed.clone()
            for old, new, start, count, begin in segments:
                descriptors.append((old.data_ptr(), new.data_ptr(), begin, start, old.shape[0], count))
                self.assertTrue(_quant.try_dequantize_swa_segments(
                    old, new, start, expected[begin:begin + count, :512],
                ))
            table = _upload(tuple(descriptors), actual.device, torch.int64)
            self.assertTrue(_quant.try_dequantize_swa_batch(
                table, actual[:, :512], 163, reference=fresh.view(torch.float8_e4m3fn),
            ))
            torch.cuda.synchronize()
            # Integer storage comparisons include NaN payload/sign bits and
            # negative zero. Whole-allocation comparison checks prefix/gaps
            # and row padding were never overwritten by a neighboring segment.
            self.assertTrue(torch.equal(actual.view(torch.int16), expected.view(torch.int16)))
            self.assertTrue(torch.equal(packed, before))
            self.assertTrue(torch.isnan(actual[131 + 162, :512]).any())
            self.assertTrue((actual[:17].view(torch.int16) == 0x3EED).all())
            self.assertTrue((actual[:, 512:].view(torch.int16) == 0x3EED).all())


if __name__ == "__main__":
    unittest.main()
