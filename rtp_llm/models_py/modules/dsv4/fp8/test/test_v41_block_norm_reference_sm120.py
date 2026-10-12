"""Explicit selection, output ownership and restoration for block norm replay."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_block_norm_reference import (
    block_norm_reference,
)


@unittest.skipUnless(torch.cuda.is_available(), "Requires SM120")
class BlockNormReferenceTest(unittest.TestCase):
    def setUp(self):
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        self.norm = RMSNorm(torch.ones(5120, device="cuda", dtype=torch.bfloat16))
        self.other = RMSNorm(self.norm.weight.clone())
        self.x = torch.arange(5120, device="cuda").bfloat16().reshape(1, -1) / 5120

    def test_selected_output_contract_and_unselected_forward(self):
        sentinel = torch.zeros_like(self.x)
        with patch.object(RMSNorm, "forward", return_value=sentinel) as original:
            with block_norm_reference(True, norms=[self.norm]):
                default = self.norm(self.x)
                self.assertIs(self.other(self.x), sentinel)
                original.assert_called_once_with(self.other, self.x, output=None)
                output = torch.empty_like(self.x)
                self.assertIs(self.norm(self.x, output=output), output)
                self.assertTrue(torch.equal(default, output))
                inplace = self.x.clone()
                self.assertIs(self.norm(inplace, output=inplace), inplace)
                self.assertTrue(torch.equal(default, inplace))
            self.assertIs(self.norm(self.x), sentinel)

    def test_selectors_reject_duplicates_and_unsupported_norms(self):
        for norms in (
            (),
            (self.norm, self.norm),
            (object(),),
            (RMSNorm(torch.ones(5120, dtype=torch.bfloat16)),),
            (RMSNorm(torch.ones(5120, device="cuda")),),
            (RMSNorm(torch.ones(512, device="cuda", dtype=torch.bfloat16)),),
        ):
            with self.subTest(norms=norms), self.assertRaisesRegex(
                ValueError, "normalization|instances"
            ):
                with block_norm_reference(True, norms=norms):
                    pass
        with self.assertRaisesRegex(ValueError, "bool"):
            with block_norm_reference(1, norms=[self.norm]):
                pass

    def test_rows_and_output_geometry_preserve_weights(self):
        weight = self.norm.weight.clone()
        with block_norm_reference(True, norms=[self.norm]):
            for value in (self.x.float(), self.x.cpu(), self.x.reshape(-1)):
                with self.subTest(shape=value.shape), self.assertRaisesRegex(
                    ValueError, "token rows"
                ):
                    self.norm(value)
            for output in (
                torch.empty_like(self.x, dtype=torch.float32),
                torch.empty_like(self.x, device="cpu"),
                torch.empty(1, 5121, device="cuda", dtype=torch.bfloat16),
                self.norm.weight.reshape(1, -1),
            ):
                with self.subTest(shape=output.shape), self.assertRaisesRegex(
                    ValueError, "output"
                ):
                    self.norm(self.x, output=output)
        self.assertTrue(torch.equal(self.norm.weight, weight))

    def test_disabled_nested_and_exception_restoration(self):
        original = RMSNorm.forward
        with block_norm_reference(False):
            self.assertIs(RMSNorm.forward, original)
        with block_norm_reference(True, norms=[self.norm]):
            outer = RMSNorm.forward
            expected = self.norm(self.x)
            with block_norm_reference(False):
                self.assertIs(RMSNorm.forward, outer)
            with block_norm_reference(True, norms=[self.other]):
                self.assertTrue(torch.equal(self.norm(self.x), expected))
                self.assertTrue(torch.equal(self.other(self.x), expected))
            self.assertIs(RMSNorm.forward, outer)
            with self.assertRaisesRegex(RuntimeError, "scope-control"):
                with block_norm_reference(True, norms=[self.other]):
                    raise RuntimeError("scope-control")
            self.assertIs(RMSNorm.forward, outer)
        self.assertIs(RMSNorm.forward, original)

    def test_explicit_fixture_width_and_inplace_output(self):
        norm = RMSNorm(torch.ones(512, device="cuda", dtype=torch.bfloat16))
        value = self.x[:, :512].contiguous().clone()
        original = RMSNorm.forward
        with block_norm_reference(True, norms=[norm], dim=512):
            expected = norm(value)
            inplace = value.clone()
            self.assertIs(norm(inplace, output=inplace), inplace)
            self.assertTrue(torch.equal(expected, inplace))
            with self.assertRaisesRegex(ValueError, "token rows"):
                norm(self.x)
        self.assertIs(RMSNorm.forward, original)
        with self.assertRaisesRegex(ValueError, "checkpoint norms"):
            with block_norm_reference(True, norms=[norm]):
                pass
        for width in (True, 512.0, 1024):
            with self.subTest(width=width), self.assertRaisesRegex(ValueError, "width"):
                with block_norm_reference(True, norms=[norm], dim=width):
                    pass


if __name__ == "__main__":
    unittest.main()
