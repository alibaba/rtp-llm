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
    def test_collective_capacity_checks_padded_batch(self):
        collective = object.__new__(Fp8CollectiveProjection)
        collective.world_size = 8
        collective.max_m = 65536
        self.assertTrue(collective.can_run_ag(8192))
        self.assertFalse(collective.can_run_ag(8193))
        self.assertTrue(collective.can_run_rs(65536))
        self.assertFalse(collective.can_run_rs(65544))

    def test_decode_reduce_scatter_is_limited_to_target_verify(self):
        collective = object.__new__(Fp8CollectiveProjection)
        collective.enable_rs = True
        collective.decode_staging = True
        collective.world_size = 8
        collective.max_m = 128
        inputs = SimpleNamespace(is_prefill=False, is_mtp_draft_update=False)
        verify = SimpleNamespace(is_target_verify=True)
        self.assertTrue(collective.eligible_rs(128, inputs, verify, True))
        self.assertFalse(collective.eligible_rs(128, inputs, verify, False))
        self.assertFalse(collective.eligible_rs(128, inputs, SimpleNamespace(is_target_verify=False), True))
        inputs.is_mtp_draft_update = True
        self.assertFalse(collective.eligible_rs(128, inputs, verify, True))
        inputs.is_mtp_draft_update = False
        with self.assertRaisesRegex(RuntimeError, "exceeds its initialized capacity"):
            collective.eligible_rs(136, inputs, verify, True)

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
