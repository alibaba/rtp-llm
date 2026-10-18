from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from flashinfer import mxfp8_quantize

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.sm120_decode import (
    Sm120DecodeExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.shared_expert import (
    combine_routed_and_shared,
)


@pytest.mark.parametrize("world", [4, 8])
@pytest.mark.parametrize("rows", [1, 4, 17])
@pytest.mark.parametrize("return_mode", ["mxfp8", "bf16", "fp32"])
def test_eager_preserves_peer_sum_until_shared_add(
    world, rows, return_mode, monkeypatch
):
    monkeypatch.setenv("DSV4_MOE_RETURN_DTYPE", return_mode)
    monkeypatch.setenv("MOE_SHARED_EXPERT_BF16_ADD", "0")
    monkeypatch.setenv("MOE_STRICT_FUSED", "1")
    hidden = 128
    x = torch.ones((rows, hidden), dtype=torch.bfloat16, device="cuda")
    weights = torch.ones((rows, 1), device="cuda")
    indices = torch.zeros((rows, 1), dtype=torch.int64, device="cuda")
    # Exact in both wire formats, but their FP32 sum is not BF16-representable.
    # This isolates premature rounding from transport quantization differences.
    partials = torch.zeros((world, rows, hidden), dtype=x.dtype, device=x.device)
    partials[0].fill_(1)
    partials[1].fill_(2**-9)
    partials = partials.flatten(0, 1)
    quantized, scales = mxfp8_quantize(partials, is_sf_swizzled_layout=False)
    payload = torch.cat(
        [quantized.view(torch.uint8), scales.reshape(world * rows, hidden // 32)],
        dim=1,
    )

    executor = Sm120DecodeExecutor.__new__(Sm120DecodeExecutor)
    torch.nn.Module.__init__(executor)
    executor.cfg = SimpleNamespace(n_routed_experts=world, local_expert_start=0)
    wire_payload = (
        payload
        if return_mode == "mxfp8"
        else partials.to(torch.float32 if return_mode == "fp32" else torch.bfloat16)
    )
    executor._compute_received = Mock(return_value=wire_payload)

    def gather(out, src, **kwargs):
        out.view(world, *src.shape).copy_(src)

    def exchange(out, src, **kwargs):
        # Exercise the production transport callers and CUDA combine kernels
        # on one GPU; only remote exchange and expert computation are replaced.
        if out.shape == wire_payload.shape:
            out.copy_(wire_payload)
        else:
            out.copy_(src)

    with (
        patch.object(torch.distributed, "get_world_size", return_value=world),
        patch.object(torch.distributed, "get_rank", return_value=0),
        patch.object(torch.distributed, "all_gather_into_tensor", side_effect=gather),
        patch.object(torch.distributed, "all_to_all_single", side_effect=exchange),
    ):
        eager = executor._forward_all_to_all(x, weights, indices)
    expected = partials.view(world, rows, hidden).float().sum(0)
    assert eager.dtype == torch.float32
    torch.testing.assert_close(eager, expected, rtol=0, atol=0)
    shared = torch.full_like(x, 2**-8)
    expected_output = (expected + shared.float()).to(x.dtype)
    early_rounded = (expected.to(x.dtype).float() + shared.float()).to(x.dtype)
    assert not torch.equal(expected_output, early_rounded)
    torch.testing.assert_close(
        combine_routed_and_shared(eager, shared, x.dtype),
        expected_output,
        rtol=0,
        atol=0,
    )
