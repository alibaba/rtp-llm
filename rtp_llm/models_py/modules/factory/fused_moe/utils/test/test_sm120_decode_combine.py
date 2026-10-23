import gc
import os
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from flashinfer import mxfp8_quantize

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors import sm120_decode
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.sm120_decode import (
    Sm120DecodeExecutor,
    Sm120DecodeExperts,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.shared_expert import (
    combine_routed_and_shared,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.sm120.workspace import (
    Sm120MoeWorkspace,
)
from rtp_llm.utils.model_weight import W


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
        with torch.cuda.stream(stream), patch.dict(
            os.environ, {"RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD": "1"}
        ):
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


@pytest.mark.parametrize("rows", [512, 513, 256 * 4])
def test_replicated_graph_capture_crosses_masked_boundary(rows, monkeypatch):
    torch.manual_seed(41)
    experts, dim, inter, world = 4, 512, 512, 4
    cfg = SimpleNamespace(
        n_routed_experts=experts,
        dim=dim,
        moe_inter_dim=inter,
        moe_w1_layout="gate_up",
        swiglu_limit=10.0,
    )
    layer_weights = {}
    for weight, scale, out_dim, in_dim in (
        (W.moe_w1, W.moe_s1, 2 * inter, dim),
        (W.moe_w2, W.moe_s2, dim, inter),
    ):
        layer_weights[weight] = torch.randint(
            -128, 127, (experts, out_dim, in_dim // 2), dtype=torch.int8, device="cuda"
        )
        layer_weights[scale] = torch.full(
            (experts, out_dim, in_dim // 32), 120, dtype=torch.uint8, device="cuda"
        ).view(torch.float8_e8m0fnu)
    executor = Sm120DecodeExecutor.__new__(Sm120DecodeExecutor)
    torch.nn.Module.__init__(executor)
    executor.cfg = SimpleNamespace(
        ep_size=world, attn_tp_size=world, local_expert_start=0, n_local_experts=experts
    )
    executor._chunk_tokens = 4096
    executor._validate_group = Mock()
    executor._experts = Sm120DecodeExperts(cfg, layer_weights)
    monkeypatch.setattr(
        executor._experts,
        "forward_contiguous",
        Mock(side_effect=AssertionError("CPU routing readback")),
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda *_: world)
    # Keep real expert kernels and capture; replace only remote reduction.
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *_a, **_k: None)
    x = torch.randn(rows, dim, dtype=torch.bfloat16, device="cuda") * 0.2
    weights = torch.full((rows, 2), 0.5, device="cuda")
    indices = torch.arange(rows * 2, device="cuda").reshape(rows, 2) % (experts * world)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), patch.dict(
        os.environ, {"RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD": "1"}
    ):
        for _ in range(3):
            executor(x, weights, indices)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = executor(x, weights, indices)
    for shift in (1, 3):
        x.mul_(0.75)
        indices.add_(shift).remainder_(experts * world)
        expected = executor(x, weights, indices).clone()
        graph.replay()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        assert torch.isfinite(captured).all()
        assert captured.abs().sum() > 0


@pytest.mark.parametrize(
    "counts", [(0, 0, 0, 0), (4, 4, 4, 4), (5, 4, 4, 4), (9, 0, 3, 8)]
)
def test_oversize_rounds_are_bounded_and_symmetric(counts, monkeypatch):
    world, dim, bound = len(counts), 128, 16
    monkeypatch.setenv("DSV4_MOE_RETURN_DTYPE", "fp32")
    monkeypatch.setattr(sm120_decode, "_A2A_PAYLOAD_TOKEN_BOUND", bound)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda *_: world)
    rounds_by_rank = []
    for rank, rows in enumerate(counts):
        monkeypatch.setattr(torch.distributed, "get_rank", lambda *_: rank)
        executor = Sm120DecodeExecutor.__new__(Sm120DecodeExecutor)
        torch.nn.Module.__init__(executor)
        executor.cfg = SimpleNamespace(n_routed_experts=world * 4)
        executor._chunk_tokens = 6
        executor._forward_fixed_ep = Mock(
            side_effect=AssertionError("unbounded fixed-EP fallback")
        )
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


def test_workspace_lifetime_stream_isolation_and_graph_generations():
    owner = Sm120MoeWorkspace()
    cfg = SimpleNamespace(sm120_workspace=owner)

    def experts(config):
        result = Sm120DecodeExperts.__new__(Sm120DecodeExperts)
        torch.nn.Module.__init__(result)
        result.cfg = config
        return result

    first, second = experts(cfg), experts(cfg)
    other_model = experts(SimpleNamespace())
    device = torch.device("cuda", torch.cuda.current_device())

    def allocate(expert, rows):
        return expert._ensure_masked_workspace(rows, 2, 2, 512, 512, 128, device)

    eager = allocate(first, 3)
    assert allocate(second, 3) is eager
    assert allocate(other_model, 3) is not eager
    with torch.cuda.stream(torch.cuda.Stream()):
        assert allocate(first, 3) is not eager

    with patch.dict(os.environ, {"RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD": "1"}):
        small = allocate(first, 3)
    assert small is not eager
    graphs, outputs = [], []
    for rows in (3, 129):
        with patch.dict(os.environ, {"RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD": "1"}):
            allocate(first, rows)
        graph = torch.cuda.CUDAGraph()
        out = torch.empty(rows, 512, device=device)
        with torch.cuda.graph(graph):
            ws = allocate(second, rows)
            ws["gather"].fill_(rows)
            out.copy_(ws["gather"][:rows])
        graphs.append(graph)
        outputs.append(out)
    # A later capacity and intervening eager call must not redirect old graphs.
    eager["gather"].fill_(-1)
    for graph, output, rows in zip(graphs, outputs, (3, 129)):
        graph.replay()
        torch.testing.assert_close(output, torch.full_like(output, rows))
    with patch.dict(os.environ, {"RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD": "1"}):
        assert allocate(first, 3) is small
    torch.cuda.synchronize()
    graphs.clear()
    del graph
    tensor_ref = weakref.ref(small["gather"])
    del first, second, cfg, owner, small, ws, eager
    gc.collect()
    assert tensor_ref() is None
