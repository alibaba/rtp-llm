"""Exact H20 cast-elision contracts and opt-in paired CUDA Graph timing."""

import json
import os
import statistics
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from rtp_llm.models_py.modules.dsv41 import sm90_dense
from rtp_llm.models_py.modules.dsv41._sm90_quant_scale_fused import unpack_ue8m0_to_fp32
from rtp_llm.models_py.modules.dsv41.moe import _sm90_deepep_v2 as transport
from rtp_llm.models_py.modules.dsv41.moe._shared_expert_triton import fused_moe_epilogue
from rtp_llm.models_py.modules.dsv41.moe.gate import fused_sqrtsoftplus_gate


def _exact(a, b):
    torch.testing.assert_close(
        a.contiguous().view(torch.uint8),
        b.contiguous().view(torch.uint8),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("h20", [False, True])
@pytest.mark.parametrize("rows", [1, 17, 32, 64, 128, 256, 257])
def test_dense_uses_exact_scale_decoder_for_card(monkeypatch, h20, rows):
    linear = object.__new__(sm90_dense.Sm90MXFP8Linear)
    linear.K = 5120
    linear.weight = torch.empty(0, device="cuda")
    linear._forward_with_scales = lambda q, scales, out: scales
    q = torch.empty((rows, 5120), dtype=torch.float8_e4m3fn, device="cuda")
    # Column-major packed scales, including all 256 exponent byte patterns.
    words = torch.randint(
        -(2**31), 2**31 - 1, (40, rows), device="cuda", dtype=torch.int32
    ).T
    monkeypatch.setattr(sm90_dense, "is_h20", lambda device: h20)
    with patch.object(
        sm90_dense,
        "unpack_ue8m0_input_scales",
        wraps=sm90_dense.unpack_ue8m0_input_scales,
    ) as old:
        actual = linear.forward_quantized(q, words)
        assert old.call_count == int(not h20 or rows > 256)
    _exact(actual, sm90_dense.unpack_ue8m0_input_scales(words))


@pytest.mark.parametrize("h20", [False, True])
@pytest.mark.parametrize("debug", [False, True])
def test_transport_dtype_and_epilogue_changed_input_graph(monkeypatch, h20, debug):
    from rtp_llm.models_py.modules.dsv4 import _record_tensor

    monkeypatch.setattr(_record_tensor, "ENABLED", debug)
    # Stub only transport: this is dtype/epilogue evidence, not collective evidence.
    x = torch.randn((128, 5120), device="cuda", dtype=torch.bfloat16)
    shared = torch.randn_like(x)
    indices = torch.zeros((128, 6), device="cuda", dtype=torch.int64)
    weights = torch.full((128, 6), 1 / 6, device="cuda")
    handle = SimpleNamespace(psum_num_recv_tokens_per_scaleup_rank=[128])
    buffer = SimpleNamespace(
        dispatch=lambda value, **kw: (value, indices, weights, handle, None),
        combine=lambda value, h: (value, None, None),
    )
    strategy = SimpleNamespace(
        cfg=SimpleNamespace(n_routed_experts=384, n_local_experts=48),
        _comm_buffer=buffer,
        _decode_graph_max_rows=256,
        _v2_num_sms=0,
        _local=None,
    )
    monkeypatch.setattr(transport, "is_h20", lambda device: h20)

    def run():
        routed = transport.forward_elastic_static(
            strategy,
            x,
            weights,
            indices,
            local_forward=lambda local, value, *args, **kw: value,
        )
        assert routed.dtype == (torch.bfloat16 if h20 and not debug else torch.float32)
        return fused_moe_epilogue(routed, shared, torch.bfloat16)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    for _ in range(4):
        x.normal_()
        shared.normal_()
        graph.replay()
        _exact(output, (x.float() + shared.float()).bfloat16())


@pytest.mark.skipif(
    os.environ.get("DSV41_H20_CAST_BENCH") != "1", reason="opt-in timing"
)
@torch.no_grad()
def test_paired_cast_chain_benchmark():
    from rtp_llm.device.device_type import is_h20

    assert is_h20(), "H20 measurements require an actual H20"

    def measure(name, rows, width, old, new):
        a, b = old(), new()
        for left, right in zip(
            a if isinstance(a, tuple) else (a,), b if isinstance(b, tuple) else (b,)
        ):
            _exact(left, right)
        graphs = []
        for fn in (old, new):
            for _ in range(3):
                fn()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(50):
                    fn()
            graphs.append(graph)
        samples = [[], []]
        for iteration in range(5):
            for arm in (0, 1) if iteration % 2 == 0 else (1, 0):
                start, stop = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                start.record()
                for _ in range(20):
                    graphs[arm].replay()
                stop.record()
                stop.synchronize()
                samples[arm].append(start.elapsed_time(stop))  # ms / 1000 calls -> us
        print(
            "H20_CAST_BENCH "
            + json.dumps(
                dict(
                    name=name,
                    rows=rows,
                    width=width,
                    old_us=samples[0],
                    new_us=samples[1],
                    speedup=statistics.median(samples[0])
                    / statistics.median(samples[1]),
                )
            ),
            flush=True,
        )

    for rows in (1, 4, 8, 16, 32, 64, 128, 256):
        for width in (1280, 2304, 4096, 5120, 8192):
            words = torch.randint(
                -(2**31),
                2**31 - 1,
                (width // 128, rows),
                device="cuda",
                dtype=torch.int32,
            ).T
            measure(
                "scales",
                rows,
                width,
                lambda: sm90_dense.unpack_ue8m0_input_scales(words),
                lambda: unpack_ue8m0_to_fp32(words),
            )
        routed = torch.randn((rows, 5120), device="cuda", dtype=torch.bfloat16)
        shared = torch.randn_like(routed)
        measure(
            "combine_epilogue",
            rows,
            5120,
            lambda: fused_moe_epilogue(routed.float(), shared, torch.bfloat16),
            lambda: fused_moe_epilogue(routed, shared, torch.bfloat16),
        )
        if rows > 32:
            scores = torch.randn((rows, 384), device="cuda", dtype=torch.bfloat16)
            bias = torch.randn((384,), device="cuda")

            def gate(value):
                return fused_sqrtsoftplus_gate(
                    value, bias, topk=6, route_scale=2.5, norm_eps=1e-12
                )

            measure(
                "gate_epilogue",
                rows,
                384,
                lambda: gate(scores.float()),
                lambda: gate(scores),
            )
