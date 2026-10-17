import pytest
import torch

from rtp_llm.models_py.modules.factory.fused_moe.utils.sm120.dispatch_quant_pack import (
    dispatch_payload_layout,
    quant_pack_dispatch_payload,
)


@pytest.mark.parametrize("hidden", [128, 640, 1024])
@pytest.mark.parametrize("n_pad", [32, 33])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_partial_scale_pack_preserves_payload(hidden, n_pad, padded, strided):
    n_valid = n_pad - int(padded)
    topk = 3
    width, packed, _, _, _ = dispatch_payload_layout(hidden, topk)
    storage = torch.full((n_pad + 2, width), 165, dtype=torch.uint8, device="cuda")
    payload = storage[1:-1]
    x = torch.randn(n_valid, hidden * (2 if strided else 1), device="cuda").bfloat16()
    if strided:
        x = x[:, ::2]
    weights = torch.rand(n_valid, topk, device="cuda")
    indices = torch.arange(n_valid * topk, device="cuda").reshape(n_valid, topk)
    actual_x, scales, actual_weights, actual_indices = quant_pack_dispatch_payload(
        x, weights, indices, payload, n_valid
    )
    torch.cuda.synchronize()

    # Guard rows catch a tail group writing beyond the final payload row.
    assert torch.all(storage[0] == 165)
    assert torch.all(storage[-1] == 165)
    torch.testing.assert_close(actual_weights[:n_valid], weights, rtol=0, atol=0)
    torch.testing.assert_close(actual_indices[:n_valid], indices.int(), rtol=0, atol=0)

    groups = hidden // 128
    grouped = x.float().reshape(n_valid, groups, 128)
    exponent = torch.ceil(torch.log2((grouped.abs().amax(-1) / 448).clamp_min(1e-4)))
    expected = (grouped / torch.exp2(exponent).unsqueeze(-1)).clamp(-448, 448)
    expected = expected.to(torch.float8_e4m3fn).reshape(n_valid, hidden)
    torch.testing.assert_close(
        actual_x[:n_valid].float(), expected.float(), rtol=0, atol=0
    )
    scale_bytes = scales.contiguous().view(torch.uint8)
    torch.testing.assert_close(
        scale_bytes[:n_valid, :groups], (exponent + 127).to(torch.uint8), rtol=0, atol=0
    )
    assert torch.all(scale_bytes[:, groups : packed * 4] == 0)
    if padded:
        assert torch.all(actual_x[n_valid:].float() == 0)
        assert torch.all(scales[n_valid:] == 0)
        assert torch.all(actual_weights[n_valid:] == 0)
        assert torch.all(actual_indices[n_valid:] == -1)
