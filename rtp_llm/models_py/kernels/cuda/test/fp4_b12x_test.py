import unittest
from unittest.mock import Mock

import torch
from rtp_llm.models_py.kernels.cuda.fp4_b12x import (
    B12xMinMPlan,
    _select_tactic,
    create_b12x_min_m_plan,
)


class B12xPlanContractTest(unittest.TestCase):
    def setUp(self):
        self.runner = Mock()
        self.plan = B12xMinMPlan(
            64,
            (),
            (),
            (),
            torch.device("cpu"),
            256,
            128,
            True,
            self.runner,
            torch.empty(0),
        )
        self.operands = [
            torch.empty((3, 64), dtype=torch.uint8),
            torch.empty((256, 64), dtype=torch.uint8).T,
            torch.empty((128, 8), dtype=torch.uint8),
            torch.empty((256, 8), dtype=torch.uint8).T,
            torch.ones((), dtype=torch.float32),
        ]

    def test_preserves_complete_tactic_tail(self):
        default = ((128, 128), (1, 1), False, False, "sm120", None)
        smaller = ((64, 128), *default[1:])
        self.assertEqual(_select_tactic(default, (default, smaller)), (64, smaller))
        self.assertEqual(_select_tactic(smaller, (default, smaller)), (64, None))
        for valid in [
            (),
            (smaller,),
            (default, ((64, 64), *default[1:])),
            (default, ((64, 128), (1, 1), False, True, "sm120", None)),
        ]:
            with self.subTest(valid=valid), self.assertRaises(ValueError):
                _select_tactic(default, valid)

    def test_rejects_invalid_operands_before_launch(self):
        invalid = [
            (0, torch.empty((3, 72), dtype=torch.uint8)),
            (0, torch.empty((3, 64), dtype=torch.float32)),
            (0, torch.empty((64, 3), dtype=torch.uint8).T),
            (0, torch.empty((3, 64), dtype=torch.uint8, device="meta")),
            (1, torch.empty((255, 64), dtype=torch.uint8).T),
            (1, torch.empty((64, 256), dtype=torch.uint8)),
            (2, torch.empty((128, 7), dtype=torch.uint8)),
            (2, torch.empty((128, 8), dtype=torch.float32)),
            (3, torch.empty((255, 8), dtype=torch.uint8).T),
            (4, torch.ones(2, dtype=torch.float32)),
            (4, torch.ones((), dtype=torch.float16)),
        ]
        for index, tensor in invalid:
            with self.subTest(index=index, shape=tensor.shape, dtype=tensor.dtype):
                operands = list(self.operands)
                operands[index] = tensor
                with self.assertRaisesRegex(ValueError, "b12x"):
                    self.plan(*operands)
                self.runner.assert_not_called()

    def test_rejects_m_and_output_dtype_before_launch(self):
        for shape in [(0, 64), (64, 64), (3, 2, 32), ()]:
            operands = list(self.operands)
            operands[0] = torch.empty(shape, dtype=torch.uint8)
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "BF16"):
                self.plan(*operands)
        with self.assertRaisesRegex(ValueError, "BF16"):
            self.plan(*self.operands, out_dtype=torch.float16)
        self.runner.assert_not_called()

    def test_launch_passes_current_operands(self):
        output = self.plan(*self.operands)
        self.assertEqual(output.shape, (3, 256))
        self.assertEqual(output.dtype, torch.bfloat16)
        inputs = self.runner.call_args.kwargs["inputs"]
        for actual, expected in zip(inputs[:5], self.operands):
            self.assertIs(actual, expected)
        self.assertIs(inputs[6], output)

    def test_cpu_factory_does_not_import_optional_runtime(self):
        weight = torch.empty((256, 64), dtype=torch.uint8)
        self.assertIsNone(
            create_b12x_min_m_plan(weight, self.operands[3].T, self.operands[4])
        )


if __name__ == "__main__":
    unittest.main()
