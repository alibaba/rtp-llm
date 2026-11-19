"""Executor numerics against unpacked FP4 weights and ordinary matmuls."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.sm120_decode import (
    Sm120DecodeExperts,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.sm120_grouped_fp4 import (
    Sm120GroupedFp4Executor,
)
from rtp_llm.utils.model_weight import W


@pytest.mark.parametrize("capturing", [False, True])
def test_prequantized_input_rejected_before_capture_host_sync(capturing):
    executor = Sm120GroupedFp4Executor.__new__(Sm120GroupedFp4Executor)
    torch.nn.Module.__init__(executor)
    x = torch.zeros((1, 128), dtype=torch.float8_e4m3fn)
    scales = torch.ones((1, 4), dtype=torch.uint8)
    with (
        patch("torch.cuda.is_current_stream_capturing", return_value=capturing),
        patch.dict(os.environ, {"RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD": "1"}),
        patch.object(executor, "forward_eager") as eager,
    ):
        with pytest.raises(ValueError, match="requires unquantized input"):
            executor.forward(x, torch.ones(1, 1), torch.zeros(1, 1), scales)
        eager.assert_not_called()


def _weights(experts, dim, inter, layout):
    weights, dense = {}, []
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
    )
    for w_key, s_key, out_dim, in_dim in (
        (W.moe_w1, W.moe_s1, 2 * inter, dim),
        (W.moe_w2, W.moe_s2, dim, inter),
    ):
        packed = torch.randint(
            0, 256, (experts, out_dim, in_dim // 2), dtype=torch.uint8, device="cuda"
        )
        scales = torch.randint(
            119, 124, (experts, out_dim, in_dim // 32), dtype=torch.uint8, device="cuda"
        )
        codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
        expanded_scales = torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
        dense.append(table[codes] * expanded_scales)
        weights[w_key] = packed.view(torch.int8)
        weights[s_key] = scales.view(torch.float8_e8m0fnu)
    if layout == "up_gate":
        for key in (W.moe_w1, W.moe_s1):
            value = weights[key]
            weights[key] = torch.cat(
                (
                    value[:, inter:].view(torch.uint8),
                    value[:, :inter].view(torch.uint8),
                ),
                dim=1,
            ).view(value.dtype)
    return weights, dense


def _reference(x, weights, ids, dense, limit):
    w13, w2 = dense
    result = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(w13.size(0)):
        rows, slots = torch.where((ids == expert) & (weights != 0))
        if rows.numel() == 0:
            continue
        gate, up = F.linear(x[rows].float(), w13[expert]).chunk(2, -1)
        if limit > 0:
            gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
        hidden = F.silu(gate) * up
        output = F.linear(hidden, w2[expert]) * weights[rows, slots, None]
        result.index_add_(0, rows, output)
    return result


@pytest.mark.parametrize("layout", ["gate_up", "up_gate"])
@pytest.mark.parametrize("rows", [1, 3, 129, 4097])
@pytest.mark.parametrize("backend", ["contiguous", "grouped"])
@pytest.mark.parametrize("limit", [0.0, 1.5])
def test_executor_matches_unpacked_reference(layout, rows, backend, limit):
    torch.manual_seed(1729)
    experts, dim, inter, topk = 16, 512, 512, 6
    cfg = SimpleNamespace(
        n_routed_experts=experts,
        n_activated_experts=topk,
        dim=dim,
        moe_inter_dim=inter,
        moe_w1_layout=layout,
        swiglu_limit=limit,
        max_tokens_per_rank=rows,
    )
    layer_weights, dense = _weights(experts, dim, inter, layout)
    x = torch.randn(rows, dim, dtype=torch.bfloat16, device="cuda") * 0.2
    ids = torch.arange(topk, device="cuda").expand(rows, -1).clone()
    weights = torch.rand(rows, topk, device="cuda") / topk
    # Expert zero sees every token; ten experts see none. Masked slots must
    # contribute nothing, including when a valid expert id has zero weight.
    ids[:, -1] = -1
    weights[:, -2] = 0
    if rows > 1:
        weights[1] = 0
    expected = _reference(x, weights, ids, dense, cfg.swiglu_limit)
    if backend == "grouped":
        executor = Sm120GroupedFp4Executor.__new__(Sm120GroupedFp4Executor)
        torch.nn.Module.__init__(executor)
        executor.cfg = cfg
        executor.setup_weights(layer_weights)
        forward = executor.forward
    else:
        executor = Sm120DecodeExperts(cfg, layer_weights)
        forward = getattr(executor, "forward_" + backend)

    def check(actual):
        assert torch.isfinite(actual).all()
        # This is an unquantized reference; the executor rounds activations
        # to FP8 and GEMM outputs to BF16. Bound relative aggregate error.
        error = (actual.float() - expected).norm() / expected.norm().clamp_min(1e-10)
        assert error < 0.08, (backend, rows, layout, error.item())
        if rows > 1:
            torch.testing.assert_close(
                actual[1], torch.zeros_like(actual[1]), rtol=0, atol=0
            )

    check(forward(x, weights, ids))
    if backend != "contiguous":
        with patch.dict(os.environ, {"RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD": "1"}):
            forward(x, weights, ids)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = forward(x, weights, ids)
        graph.replay()
        check(captured)


def test_sm120_decode_graph_experts():
    from flashinfer import mxfp8_quantize

    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.sm120_graph_decode import (
        Sm120GraphDecodeExperts,
        _fixed_payload_views,
        pack_fixed_payload,
        require_sm120_decode_graph_providers,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.utils.sm120.graph_decode import (
        Sm120DecodeMoeGeometry,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.utils.sm120.workspace import (
        Sm120MoeWorkspace,
    )
    from rtp_llm.ops.compute_ops import DecodeExecutionMode

    require_sm120_decode_graph_providers()
    torch.manual_seed(1729)
    device = torch.device("cuda", torch.cuda.current_device())
    geometry = Sm120DecodeMoeGeometry(33, 4, 512, 512, 16, 2)
    rows, dim, inter = geometry.rows, geometry.hidden, geometry.inter
    experts, topk = geometry.local_experts, geometry.topk
    cfg = SimpleNamespace(
        n_routed_experts=experts,
        n_activated_experts=topk,
        dim=dim,
        moe_inter_dim=inter,
        moe_w1_layout="gate_up",
        swiglu_limit=1.5,
    )
    layer_weights, dense = _weights(experts, dim, inter, cfg.moe_w1_layout)
    executor = Sm120GraphDecodeExperts(cfg, layer_weights)
    plan = SimpleNamespace(mode=DecodeExecutionMode.GRAPH)
    pools = (Sm120MoeWorkspace(), Sm120MoeWorkspace())
    for pool in pools:
        pool.reserve_decode_graph_workspace(geometry, device)
    capture_ws, eager_ws = (
        pool.decode_graph_workspace(plan, geometry, device) for pool in pools
    )
    x = torch.randn(rows, dim, dtype=torch.bfloat16, device=device) * 0.2
    # Every row targets a single expert; expert capacity must permit all N
    # assignments. Duplicate IDs exercise the bound for hash routing.
    ids = torch.zeros(rows, topk, dtype=torch.int64, device=device)
    weights = torch.full((rows, topk), 0.5, device=device)
    valid = torch.ones(rows, dtype=torch.bool, device=device)

    def run_masked(ws):
        pack_fixed_payload(x, weights, ids, valid, ws.gathered, geometry.total_experts)
        q, scales, local_weights, local_ids = _fixed_payload_views(
            ws.gathered, dim, topk
        )
        executor._forward_coalesced(
            q, local_weights, local_ids, scales, geometry, 0, ws
        )
        return ws.partial

    run_masked(capture_ws)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run_masked(capture_ws)

    for live, expert in ((rows, 0), (1, 1), (0, 2), (rows - 1, 2)):
        x.copy_(torch.randn_like(x) * 0.2)
        ids.fill_(expert)
        valid.copy_(torch.arange(rows, device=device) < live)
        graph.replay()
        fixed = run_masked(eager_ws)
        torch.testing.assert_close(output, fixed, rtol=0, atol=0)
        torch.testing.assert_close(
            output[live:], torch.zeros_like(output[live:]), rtol=0, atol=0
        )
        if live == 0:
            continue

        mxfp8, mxfp8_scale = mxfp8_quantize(x[:live], is_sf_swizzled_layout=False)
        scales = mxfp8_scale.view(torch.uint8).reshape(live, -1)
        restored = (
            (
                mxfp8.float().reshape(live, -1, 32)
                * torch.exp2(scales.float() - 127).unsqueeze(-1)
            )
            .reshape(live, dim)
            .to(torch.bfloat16)
        )
        contiguous = executor.forward_contiguous(
            mxfp8, weights[:live], ids[:live], input_scale=scales
        )
        independent = _reference(
            restored, weights[:live], ids[:live], dense, cfg.swiglu_limit
        )
        # FP8 rounding is measured separately from the masked-versus-
        # contiguous kernel tolerance and the bitwise replay check.
        compute_error = (output[:live] - independent).norm() / independent.norm()
        layout_error = (output[:live] - contiguous).norm() / contiguous.norm()
        assert compute_error < 0.08
        assert layout_error < 0.01
