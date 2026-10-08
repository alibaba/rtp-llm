"""Numerical contract for the strided shared-expert SiTU producer."""

import pytest
import torch

from rtp_llm.models_py.triton_kernels.common import activation


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_situ_and_mul_consumes_gate_up_views_without_reordering():
    torch.manual_seed(492)
    packed = torch.randn(17, 2, 6144, device="cuda", dtype=torch.bfloat16) * 4
    gate, up = packed[:, 0], packed[:, 1]
    gate_fp32, up_fp32 = gate.float(), up.float()
    expected = (
        (4.0 * torch.tanh(gate_fp32 / 4.0) * torch.sigmoid(gate_fp32))
        * (25.0 * torch.tanh(up_fp32 / 25.0))
    ).to(torch.bfloat16)

    assert hasattr(activation, "situ_and_mul"), "strided SiTU producer is unavailable"
    observed = activation.situ_and_mul(gate, up, beta=4.0, linear_beta=25.0)

    assert observed.shape == gate.shape
    assert observed.dtype == torch.bfloat16
    torch.testing.assert_close(observed, expected, rtol=0.02, atol=0.125)
