import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.distributed.fp8_collective_projection import (
    Fp8Activation,
    Fp8CollectiveProjection,
)


class Projection:
    K = 512
    N = 16
    scale_ue8m0 = True

    def quantize_input(self, _):
        raise AssertionError("prequantized input must not be quantized again")

    def forward_quantized(self, values, scales, *, out):
        self.last_scales = scales.clone()
        out.copy_(values.float().mean(dim=1, keepdim=True).expand_as(out))


class Fp8CollectiveProjectionTest(unittest.TestCase):
    def test_prequantized_input_reaches_consumer_without_requantization(self):
        values = torch.full((4, 512), 2, dtype=torch.float8_e4m3fn)
        scale_wire = torch.full((1, 4), 0x7F7F7F7F, dtype=torch.int32)
        activation = Fp8Activation(values, scale_wire)
        projection = Projection()
        collective = object.__new__(Fp8CollectiveProjection)
        collective.enable_ag = True
        collective.group = SimpleNamespace(group_name="test")
        collective.device = torch.device("cpu")
        collective.max_m = 8
        collective.hidden_size = 512
        collective.world_size = 2

        def pipeline(inputs, consume, outputs, group_name, *, ag_out_needed):
            self.assertEqual(group_name, "test")
            self.assertFalse(ag_out_needed)
            self.assertEqual(inputs[0].dtype, torch.uint8)
            for rank in range(2):
                consume(inputs, rank)

        with patch(
            "rtp_llm.models_py.distributed.fp8_collective_projection.symm._pipelined_multi_all_gather_and_consume",
            side_effect=pipeline,
        ):
            output = collective.all_gather_gemm(activation, projection)
        self.assertEqual(output.shape, (8, 16))
        torch.testing.assert_close(output, torch.full_like(output, 2), rtol=0, atol=0)
        self.assertEqual(projection.last_scales.shape, (4, 1))


if __name__ == "__main__":
    unittest.main()
