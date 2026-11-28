"""Accelerator acceptance, distinct from the CPU FT byte/layout checks."""

import unittest

import torch

from rtp_llm.models_py.new_models.deepseek_v3.test import (
    test_mla_ft_roundtrip as fixtures,
)


class MlaFtGpu(unittest.TestCase):
    def test_block_fp8_serving(self):
        self.assertTrue(torch.cuda.is_available(), "requires an allocated H20")
        fixtures.MlaFtRoundTrip().check_block_writer("cuda:0")


if __name__ == "__main__":
    unittest.main()
