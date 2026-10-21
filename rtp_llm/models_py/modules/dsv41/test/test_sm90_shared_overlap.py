"""Native G32 shared-expert fork/join, including changed-input graph replay."""

from unittest.mock import patch

import pytest
import torch

from rtp_llm.models_py.modules.dsv41.moe.shared_expert import (
    MXFP8SharedExpertExecutor,
    W13SharedExpert,
)


def _shared(dim=512, intermediate=256):
    weights = {}
    for name, n, k in (("w13", intermediate * 2, dim), ("w2", dim, intermediate)):
        weights[name + "_w"] = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        weights[name + "_s"] = torch.full(
            (n // 32, k // 32), 2**-6, device="cuda"
        ).to(torch.float8_e8m0fnu)
    return W13SharedExpert(dim, intermediate, weights, swiglu_limit=10.0)


@pytest.fixture(autouse=True)
def _sm90():
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
    torch.manual_seed(241017)


@pytest.mark.parametrize("rows", [1, 8, 32])
def test_native_shared_two_stream_graph_changed_inputs(rows):
    shared = _shared()
    with patch.dict(
        "os.environ", {"DSV4_SHARED_EXPERT_MODE": "overlap", "MOEDBG": "0"}
    ):
        executor = MXFP8SharedExpertExecutor()
        executor.prepare(shared)
        assert executor._stream is not None
        x = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
        routed_input = torch.randn_like(x)

        def run():
            executor.start(shared, x)
            assert executor._active_stream is executor._stream
            # A separate consumer on the calling stream; join must precede
            # the read of the shared-expert result, in eager and replay.
            routed = routed_input.float().square()
            y = executor.finish()
            assert y.dtype == torch.bfloat16
            return (routed + y.float()).bfloat16()

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run()
        for scale in (0.125, 1.0, 2.0):
            x.normal_().mul_(scale)
            routed_input.normal_()
            expected = (routed_input.float().square() + shared(x).float()).bfloat16()
            eager = run()
            graph.replay()
            torch.testing.assert_close(eager, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert executor._out is None and executor._input is None


def test_explicit_sequential_empty_and_large_batch():
    shared = _shared()
    for mode in ("sequential", "overlap"):
        with patch.dict("os.environ", {"DSV4_SHARED_EXPERT_MODE": mode, "MOEDBG": "0"}):
            executor = MXFP8SharedExpertExecutor()
            executor.prepare(shared)
            for rows in (0, 33):
                x = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
                executor.start(shared, x)
                assert executor._active_stream is None
                actual = executor.finish()
                assert actual.shape == x.shape
                if rows:
                    torch.testing.assert_close(actual, shared(x), rtol=0, atol=0)


def test_native_shared_overlapping_layers_preserve_allocator_lifetimes():
    shared = _shared()
    with patch.dict(
        "os.environ", {"DSV4_SHARED_EXPERT_MODE": "overlap", "MOEDBG": "0"}
    ):
        executor = MXFP8SharedExpertExecutor()
        executor.prepare(shared)
        for _ in range(16):
            x = torch.randn(1, 512, device="cuda", dtype=torch.bfloat16)
            expected = shared(x)
            executor.start(shared, x)
            actual = executor.finish()
            # Drop the input and reuse same-sized blocks on both streams.
            del x
            junk = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16)
            with torch.cuda.stream(executor._stream):
                other_junk = torch.randn_like(junk)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
