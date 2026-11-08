from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors import sm120_decode
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.sm120_decode import (
    Sm120DecodeExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.shared_expert import (
    combine_routed_and_shared,
)


@pytest.mark.parametrize("world", [4, 8])
@pytest.mark.parametrize("rows", [1, 4, 17])
def test_eager_preserves_peer_sum_until_shared_add(world, rows, monkeypatch):
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
    # The return wire is fp32 unconditionally, so it carries the expert
    # partials unquantized (see sm120_decode._compute_received).
    payload = partials.float()

    executor = Sm120DecodeExecutor.__new__(Sm120DecodeExecutor)
    torch.nn.Module.__init__(executor)
    executor.cfg = SimpleNamespace(n_routed_experts=world, local_expert_start=0)
    executor._experts = Mock()
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


@pytest.mark.parametrize(
    "counts", [(0, 0, 0, 0), (4, 4, 4, 4), (5, 4, 4, 4), (9, 0, 3, 8)]
)
def test_oversize_rounds_are_bounded_and_symmetric(counts, monkeypatch):
    world, dim, bound = len(counts), 128, 16
    monkeypatch.setattr(sm120_decode, "_A2A_PAYLOAD_TOKEN_BOUND", bound)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda *_: world)
    rounds_by_rank = []
    for rank, rows in enumerate(counts):
        monkeypatch.setattr(torch.distributed, "get_rank", lambda *_: rank)
        executor = Sm120DecodeExecutor.__new__(Sm120DecodeExecutor)
        torch.nn.Module.__init__(executor)
        executor.cfg = SimpleNamespace(n_routed_experts=world * 4)
        executor._chunk_tokens = 6
        executor._compute_received = lambda recv, n, x, *_: torch.ones(
            n, dim, dtype=torch.float32, device=x.device
        )
        rounds = []

        def gather(out, src, **kwargs):
            out.copy_(torch.tensor(counts, device=out.device).reshape_as(out))

        def exchange(out, src, **kwargs):
            if out.dtype == torch.uint8:
                received = kwargs["output_split_sizes"]
                assert sum(received) <= bound
                assert kwargs["input_split_sizes"] == [received[rank]] * world
                rounds.append(tuple(received))
                out.zero_()
            else:
                out.fill_(1)

        monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", gather)
        monkeypatch.setattr(torch.distributed, "all_to_all_single", exchange)
        x = torch.ones(rows, dim, dtype=torch.bfloat16, device="cuda")
        weights = torch.ones(rows, 1, device="cuda")
        indices = torch.zeros(rows, 1, dtype=torch.int64, device="cuda")
        actual = executor._forward_all_to_all(x, weights, indices)
        torch.testing.assert_close(
            actual, torch.full_like(actual, world), rtol=0, atol=0
        )
        assert sum(part[rank] for part in rounds) == rows
        rounds_by_rank.append(rounds)
    assert all(rounds == rounds_by_rank[0] for rounds in rounds_by_rank)
