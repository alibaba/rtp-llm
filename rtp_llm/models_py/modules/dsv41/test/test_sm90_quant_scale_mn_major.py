"""Direct TMA scale layout: exact values, per-group alignment, dynamic replay."""

import pytest
import torch

from rtp_llm.models_py.modules.dsv41 import _sm90_quant_scale_fused as fused
from rtp_llm.models_py.modules.dsv41.sm90_dense import unpack_ue8m0_input_scales


@pytest.mark.parametrize("rows", [0, 1, 3, 7, 17, 128])
@pytest.mark.parametrize("groups", [1, 8])
def test_unpack_all_exponents_and_group_alignment(rows, groups):
    raw = torch.arange(groups * rows * 128, device="cuda", dtype=torch.int32)
    raw = raw.to(torch.uint8).reshape(groups * rows, 128)
    packed = raw.view(torch.int32).T.contiguous().T
    actual = fused.unpack_ue8m0_to_fp32(packed, mn_major=True, n_groups=groups)
    expected = unpack_ue8m0_input_scales(packed).reshape(groups, rows, 128)
    if groups == 1:
        actual = actual.unsqueeze(0)
    assert actual.stride(-2) == 1
    assert actual.stride(-1) == max(4, (rows + 3) // 4 * 4)
    if groups > 1:
        assert actual.stride(0) % 4 == 0
    # Compare bits, including exponent 255's signaling-NaN payload.
    torch.testing.assert_close(
        actual.contiguous().view(torch.int32),
        expected.contiguous().view(torch.int32),
        rtol=0,
        atol=0,
    )


def _packed_quant_reference(x):
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8

    q, packed = sgl_per_token_group_quant_fp8(
        x.contiguous(),
        group_size=32,
        eps=torch.finfo(torch.float32).tiny,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    return q, unpack_ue8m0_input_scales(packed)[:, : x.shape[1] // 32]


@pytest.mark.parametrize("rows", [1, 7, 16, 17, 96, 97, 127, 128, 191, 251, 256, 257])
@pytest.mark.parametrize("policy", ["legacy", "v2"])
def test_quant_exact_dynamic_graph(rows, policy, monkeypatch):
    monkeypatch.setenv("DSV4_FP8_QUANT_KERNEL", policy)
    x = torch.randn(rows, 640, device="cuda", dtype=torch.bfloat16)
    q, scales = fused.quantize_g32(x, mn_major=True)
    assert scales.stride() == (1, (rows + 3) // 4 * 4)
    for _ in range(3):
        fused.quantize_g32(x, out_q=q, out_scale=scales, mn_major=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused.quantize_g32(x, out_q=q, out_scale=scales, mn_major=True)
    for multiplier in (0.001, 64.0):
        x.copy_(torch.randn_like(x) * multiplier)
        graph.replay()
        reference_q, reference_scales = fused.quantize_g32(x)
        torch.testing.assert_close(
            q.view(torch.uint8), reference_q.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(scales, reference_scales, rtol=0, atol=0)
        # The packed CUDA producer and Torch UE8M0 decode are independent of
        # this Triton kernel; retaining the eager self-check alone can hide
        # a graph/eager bug shared by both native-scale calls.
        original_q, original_scales = _packed_quant_reference(x)
        torch.testing.assert_close(
            q.view(torch.uint8), original_q.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(
            scales.contiguous().view(torch.int32),
            original_scales.contiguous().view(torch.int32),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("rows", [1, 7, 17, 128])
def test_woa_scales_mn_major_exact(rows):
    x = torch.randn(rows, 64, 512, device="cuda", dtype=torch.bfloat16)
    freqs = torch.polar(
        torch.ones(rows, 32, device="cuda"), torch.randn(rows, 32, device="cuda")
    )
    q, scales = fused.quantize_wo_a_g32(
        x, freqs, n_groups=8, heads_per_group=8, mn_major=True
    )
    reference_q, reference_scales = fused.quantize_wo_a_g32(
        x, freqs, n_groups=8, heads_per_group=8
    )
    torch.testing.assert_close(
        q.contiguous().view(torch.uint8),
        reference_q.contiguous().view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(scales, reference_scales, rtol=0, atol=0)
    for group in range(8):
        assert scales[:, group, :].stride() == (1, (rows + 3) // 4 * 4)
        assert scales[:, group, :].data_ptr() % 16 == 0
