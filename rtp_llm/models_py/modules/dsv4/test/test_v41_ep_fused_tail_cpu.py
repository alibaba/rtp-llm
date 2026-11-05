"""CPU tests for the E8 fused EP combine tail wrapper: argument validation
mirroring both parent kernels' checks, empty-batch short-circuit. Kernel
numerics are proven bitwise on GPU (r13_mb_eptail microbench) and by the
parity sha gate."""

import importlib.util
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "_ep_mxfp8_combine_under_test",
        ROOT / "moe" / "_nccl_ep_mxfp8_combine.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mc = load_module()

HIDDEN = 5120
WORLD = 4
ROWS = 6
PAYLOAD_COLS = HIDDEN + HIDDEN // 32


def payload(rows=ROWS):
    return torch.randint(0, 256, (WORLD * rows, PAYLOAD_COLS), dtype=torch.uint8)


class WrapperValidationTest(unittest.TestCase):
    def test_rejects_non_uint8_payload(self):
        p = torch.zeros((WORLD * ROWS, PAYLOAD_COLS), dtype=torch.int8)
        with self.assertRaises(ValueError):
            mc.mxfp8_peer_sum_shared_epilogue(
                p,
                torch.zeros(ROWS, HIDDEN, dtype=torch.bfloat16),
                ROWS,
                HIDDEN,
                WORLD,
                torch.bfloat16,
            )

    def test_rejects_wrong_payload_shape(self):
        p = torch.zeros((WORLD * ROWS + 1, PAYLOAD_COLS), dtype=torch.uint8)
        with self.assertRaises(ValueError):
            mc.mxfp8_peer_sum_shared_epilogue(
                p,
                torch.zeros(ROWS, HIDDEN, dtype=torch.bfloat16),
                ROWS,
                HIDDEN,
                WORLD,
                torch.bfloat16,
            )

    def test_rejects_shared_shape_mismatch(self):
        p = payload()
        with self.assertRaises(ValueError):
            mc.mxfp8_peer_sum_shared_epilogue(
                p,
                torch.zeros(ROWS + 1, HIDDEN, dtype=torch.bfloat16),
                ROWS,
                HIDDEN,
                WORLD,
                torch.bfloat16,
            )

    def test_rejects_non_unit_stride_shared(self):
        p = payload()
        strided = torch.zeros(ROWS, 2 * HIDDEN, dtype=torch.bfloat16)[:, ::2]
        self.assertEqual(tuple(strided.shape), (ROWS, HIDDEN))
        self.assertNotEqual(strided.stride(1), 1)
        with self.assertRaises(ValueError):
            mc.mxfp8_peer_sum_shared_epilogue(
                p,
                strided,
                ROWS,
                HIDDEN,
                WORLD,
                torch.bfloat16,
            )

    def test_empty_rows_short_circuit(self):
        p = torch.zeros((0, PAYLOAD_COLS), dtype=torch.uint8)
        out = mc.mxfp8_peer_sum_shared_epilogue(
            p,
            torch.zeros(0, HIDDEN, dtype=torch.bfloat16),
            0,
            HIDDEN,
            WORLD,
            torch.bfloat16,
        )
        self.assertEqual(tuple(out.shape), (0, HIDDEN))
        self.assertEqual(out.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
