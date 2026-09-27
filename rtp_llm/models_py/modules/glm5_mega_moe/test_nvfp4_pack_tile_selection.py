"""CPU dispatch regressions; no CUDA allocation or kernel launch."""

import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SOURCE = Path(__file__).with_name("mega_nvfp4_input_packer_triton.py")
spec = importlib.util.spec_from_file_location("nvfp4_pack_dispatch_test", SOURCE)
packer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packer)
ENV = "GLM5_MEGA_MOE_NVFP4_PACK_BLOCK_M"


class LaunchRecorder:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


class NVFP4PackTileSelectionTest(unittest.TestCase):
    def dispatch(self, tokens, hidden=6144, topk=4, override=None):
        strides = (8192, 8, 9, 4096, 128, 1, 12, 13)
        tensors = [
            SimpleNamespace(stride=lambda axis, stride=stride: stride)
            for stride in strides
        ]
        gsf, packed = LaunchRecorder(), LaunchRecorder()
        triton = SimpleNamespace(
            cdiv=lambda value, divisor: (value + divisor - 1) // divisor,
            next_power_of_2=lambda value: 1 << (value - 1).bit_length(),
        )
        with patch.dict(os.environ):
            os.environ.pop(ENV, None)
            if override is not None:
                os.environ[ENV] = override
            with (
                patch.object(
                    packer, "_validate_inputs", return_value=(tokens, hidden, topk)
                ),
                patch.object(packer, "_row_gsf_kernel", gsf, create=True),
                patch.object(packer, "_pack_nvfp4_inputs_kernel", packed, create=True),
                patch.object(packer, "triton", triton),
            ):
                packer.fused_pack_mega_nvfp4_inputs(*tensors)
        if tokens == 0:
            self.assertEqual(gsf.calls, [])
            self.assertEqual(packed.calls, [])
            return None
        self.assertEqual(len(gsf.calls), 1)
        self.assertEqual(len(packed.calls), 1)
        grid, args, kwargs = packed.calls[0]
        self.assertEqual(args[:8], tuple(tensors))
        self.assertEqual(args[8:11], (tokens, hidden, topk))
        self.assertEqual(args[11:], tuple(strides[i] for i in (0, 1, 2, 3, 4, 6, 7)))
        self.assertEqual(gsf.calls[0][0], (tokens,))
        self.assertEqual(
            gsf.calls[0][1], (tensors[0], tensors[5], tokens, hidden, strides[0])
        )
        self.assertEqual(
            gsf.calls[0][2],
            {"BLOCK_D": triton.next_power_of_2(hidden), "num_warps": 8},
        )
        self.assertEqual(kwargs["num_warps"], 4)
        self.assertEqual(kwargs["BLOCK_TOPK"], triton.next_power_of_2(topk))
        self.assertEqual(
            grid, (triton.cdiv(tokens, kwargs["BLOCK_M"]), triton.cdiv(hidden, 64))
        )
        return kwargs["BLOCK_M"]

    def test_measured_decode_shapes(self):
        for tokens in (80, 96, 112, 128):
            with self.subTest(tokens=tokens):
                self.assertEqual(self.dispatch(tokens), 16)

    def test_other_small_shapes_unchanged(self):
        for tokens in (1, 3, 5, 16, 64, 79, 81, 95, 97, 111, 113, 127, 129, 1023):
            with self.subTest(tokens=tokens):
                self.assertEqual(self.dispatch(tokens), 4)
        self.assertEqual(self.dispatch(80, hidden=4096), 4)
        self.assertEqual(self.dispatch(128, topk=8), 4)

    def test_prefill_threshold_unchanged(self):
        for hidden in (4096, 6144):
            for tokens, expected in ((1023, 4), (1024, 16), (1025, 16)):
                self.assertEqual(self.dispatch(tokens, hidden=hidden), expected)

    def test_override_precedence(self):
        for tokens in (1, 80, 96, 112, 128, 1024):
            for tile in (1, 2, 4, 8, 16):
                self.assertEqual(self.dispatch(tokens, override=str(tile)), tile)

    def test_invalid_override_and_empty_input(self):
        for value in ("0", "32", "invalid"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.dispatch(80, override=value)
                self.assertIsNone(self.dispatch(0, override=value))


if __name__ == "__main__":
    unittest.main()
