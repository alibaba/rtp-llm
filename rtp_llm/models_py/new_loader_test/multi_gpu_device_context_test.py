import unittest
from unittest import mock

import torch
from rtp_llm.models_py.layers import activation
from rtp_llm.models_py.layers.norm import RMSNorm, RMSResNorm


class MultiGpuDeviceContextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.device_count() < 2:
            raise RuntimeError(
                "multi_gpu_device_context_test must be scheduled with two GPUs"
            )

    def setUp(self):
        self.original_device = torch.cuda.current_device()
        self.input_device = 1 if self.original_device == 0 else 0

    def tearDown(self):
        torch.cuda.set_device(self.original_device)

    def test_fused_silu_uses_non_current_input_device(self):
        gate_up = torch.randn(
            2, 256, dtype=torch.bfloat16, device=f"cuda:{self.input_device}"
        )
        torch.cuda.set_device(self.original_device)

        with mock.patch.object(activation, "_SILU_FUSED_ENABLED", True):
            output = activation.silu_and_mul(gate_up)
        torch.cuda.synchronize(self.input_device)

        gate, up = gate_up.chunk(2, dim=-1)
        expected = (torch.nn.functional.silu(gate.float()) * up.float()).to(
            gate_up.dtype
        )
        self.assertEqual(torch.cuda.current_device(), self.original_device)
        self.assertEqual(output.device, gate_up.device)
        torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)

    def test_rmsnorm_uses_non_current_input_device(self):
        input_device = torch.device("cuda", self.input_device)
        layer = RMSNorm(256, params_dtype=torch.bfloat16).to(input_device)
        inputs = torch.randn(4, 256, dtype=torch.bfloat16, device=input_device)
        variance = inputs.float().pow(2).mean(-1, keepdim=True)
        expected = (
            layer.weight * inputs.float() * torch.rsqrt(variance + layer.eps)
        ).to(inputs.dtype)
        torch.cuda.set_device(self.original_device)

        output = layer(inputs)
        torch.cuda.synchronize(self.input_device)

        self.assertEqual(torch.cuda.current_device(), self.original_device)
        self.assertEqual(output.device, input_device)
        torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)

    def test_rms_res_norm_uses_non_current_input_device(self):
        input_device = torch.device("cuda", self.input_device)
        layer = RMSResNorm(256, params_dtype=torch.bfloat16).to(input_device)
        hidden_states = torch.randn(4, 256, dtype=torch.bfloat16, device=input_device)
        residual = torch.randn(4, 256, dtype=torch.bfloat16, device=input_device)
        expected_residual = hidden_states + residual
        residual_fp32 = expected_residual.float()
        variance = residual_fp32.pow(2).mean(-1, keepdim=True)
        expected_output = (
            layer.weight.float() * residual_fp32 * torch.rsqrt(variance + layer.eps)
        ).to(hidden_states.dtype)
        torch.cuda.set_device(self.original_device)

        output, residual_out = layer(hidden_states, residual)
        torch.cuda.synchronize(self.input_device)

        self.assertEqual(torch.cuda.current_device(), self.original_device)
        self.assertEqual(output.device, input_device)
        self.assertEqual(residual_out.device, input_device)
        torch.testing.assert_close(output, expected_output, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(
            residual_out, expected_residual, rtol=2e-2, atol=2e-2
        )


if __name__ == "__main__":
    unittest.main()
