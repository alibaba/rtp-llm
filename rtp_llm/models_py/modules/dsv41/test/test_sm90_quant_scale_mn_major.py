"""Direct TMA scale layout: exact values, alignment, dynamic replay."""

import pytest
import torch

from rtp_llm.models_py.modules.dsv41 import _sm90_quant_scale_fused as fused
from rtp_llm.models_py.modules.dsv41.sm90_dense import unpack_ue8m0_input_scales


@pytest.mark.parametrize("rows", [0, 1, 3, 7, 17, 128])
def test_unpack_all_exponents_and_alignment(rows):
    raw = torch.arange(rows * 128, device="cuda", dtype=torch.int32)
    raw = raw.to(torch.uint8).reshape(rows, 128)
    packed = raw.view(torch.int32).T.contiguous().T
    actual = fused.unpack_ue8m0_to_fp32(packed, mn_major=True)
    expected = unpack_ue8m0_input_scales(packed).reshape(rows, 128)
    assert actual.stride(-2) == 1
    assert actual.stride(-1) == max(4, (rows + 3) // 4 * 4)
    # Compare bits, including exponent 255's signaling-NaN payload.
    torch.testing.assert_close(
        actual.contiguous().view(torch.int32),
        expected.contiguous().view(torch.int32),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("rows", [1, 7, 17, 128])
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


@pytest.mark.parametrize("rows,n,k", [(1, 256, 128), (7, 320, 640), (17, 256, 512), (128, 1792, 5120)])
def test_dense_matches_row_major_and_graph_has_no_scale_transpose(rows, n, k):
    import deep_gemm
    from rtp_llm.models_py.modules.dsv41.sm90_deepgemm_dense import Sm90DeepGEMMMXFP8Linear
    from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear

    torch.manual_seed(5141)
    weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.ones((n + 31) // 32, k // 32, device="cuda").to(torch.float8_e8m0fnu)
    linear = V41MXFP8Linear(weight, scales)
    adapter = linear._sm90_linear
    assert isinstance(adapter, Sm90DeepGEMMMXFP8Linear)
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    q, packed = linear._quantize_input(x)
    out = torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        adapter.forward_quantized(q, packed, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        adapter.forward_quantized(q, packed, out=out)
    for multiplier in (0.03125, 16.0):
        new_q, new_packed = linear._quantize_input(torch.randn_like(x) * multiplier)
        q.copy_(new_q)
        packed.copy_(new_packed)
        graph.replay()
        reference = torch.empty_like(out)
        deep_gemm.fp8_gemm_nt(
            (q, unpack_ue8m0_input_scales(packed)[:, :k // 32]),
            (weight, scales.float()), reference, recipe=(1, 32, 32), compiled_dims="",
        )
        torch.testing.assert_close(out, reference, rtol=0, atol=0)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
        adapter.forward_quantized(q, packed, out=out)
    kernels = [event.name for event in profile.events() if event.device_type == torch.autograd.DeviceType.CUDA]
    assert any("sm90_fp8_gemm_1d2d_impl" in name for name in kernels), kernels
    assert not any("transpose" in name for name in kernels), kernels
