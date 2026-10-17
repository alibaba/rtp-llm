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
def test_eager_and_graph_preserve_peer_sum_until_shared_add(world, rows, monkeypatch):
    monkeypatch.delenv("DSV4_MOE_RETURN_DTYPE", raising=False)
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
    executor._experts = Mock()
    executor._experts.forward_masked.side_effect = (
        lambda *args, out, **kwargs: out.copy_(partials)
    )
    executor._compute_received = Mock(return_value=payload)

    def gather(out, src, **kwargs):
        out.view(world, *src.shape).copy_(src)

    def exchange(out, src, **kwargs):
        # Exercise the production transport callers and CUDA combine kernels
        # on one GPU; only remote exchange and expert computation are replaced.
        if out.shape == payload.shape:
            out.copy_(payload)
        elif out.dtype == torch.bfloat16:
            out.copy_(partials)
        else:
            out.copy_(src)

    with (
        patch.object(torch.distributed, "get_world_size", return_value=world),
        patch.object(torch.distributed, "get_rank", return_value=0),
        patch.object(torch.distributed, "all_gather_into_tensor", side_effect=gather),
        patch.object(torch.distributed, "all_to_all_single", side_effect=exchange),
    ):
        eager = executor._forward_all_to_all(x, weights, indices)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            executor._forward_fixed_ep(x, weights, indices, pad_floor=rows)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = executor._forward_fixed_ep(x, weights, indices, pad_floor=rows)
        graph.replay()

    expected = partials.view(world, rows, hidden).float().sum(0)
    assert eager.dtype == captured.dtype == torch.float32
    torch.testing.assert_close(eager, expected, rtol=0, atol=0)
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    shared = torch.full_like(x, 2**-8)
    expected_output = (expected + shared.float()).to(x.dtype)
    early_rounded = (expected.to(x.dtype).float() + shared.float()).to(x.dtype)
    assert not torch.equal(expected_output, early_rounded)
    for routed in (eager, captured):
        torch.testing.assert_close(
            combine_routed_and_shared(routed, shared, x.dtype),
            expected_output,
            rtol=0,
            atol=0,
        )
