"""Compare the MTP frontend with the native RTP unfused operations."""

import functools
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

from rtp_llm.models_py.kernels.cuda import mtp_frontend
from rtp_llm.models_py.modules.base.common.embedding import Embedding
from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm


@functools.lru_cache(maxsize=1)
def native():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA required")
    op = mtp_frontend.native(enabled=True)
    assert op is not None, "Native frontend must compile for this acceptance test"
    return op


def fixture(rows, magnitude=1.0, eps=1e-06):
    torch.manual_seed(91)
    weight = (torch.randn(17, 6144, device="cuda") * magnitude).bfloat16()
    hidden = (torch.randn(rows, 6144, device="cuda") * magnitude).bfloat16()
    emb = Embedding(
        SimpleNamespace(), SimpleNamespace(get_attn_tp_size=lambda: 1), weight
    )
    en = RMSNorm(torch.randn(6144, device="cuda", dtype=torch.bfloat16), eps=eps)
    hn = RMSNorm(torch.randn(6144, device="cuda", dtype=torch.bfloat16), eps=eps)
    ids = (torch.arange(rows, device="cuda", dtype=torch.int32) % 17).contiguous()
    positions = (torch.arange(rows, device="cuda", dtype=torch.int32) % 3).contiguous()
    return (emb, hidden, en, hn, ids, positions)


def reference(emb, hidden, en, hn, ids, positions):
    e = emb(ids)
    e = torch.where((positions == 0).unsqueeze(-1), torch.zeros_like(e), e)
    return torch.cat((en(e), hn(hidden)), dim=-1)


def test_native_precision(native, rows, magnitude, eps):
    args = fixture(rows, magnitude, eps)
    with torch.no_grad():
        (output, residual) = mtp_frontend.try_mtp_frontend(native, *args)
        expected = reference(*args)
        torch.testing.assert_close(output, expected, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0, atol=0)


def test_graph_replay_changed_inputs(native):
    args = fixture(4)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(stream):
        for _ in range(3):
            mtp_frontend.try_mtp_frontend(native, *args)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            (result, residual) = mtp_frontend.try_mtp_frontend(native, *args)
        for value in [0, 16, 3]:
            args[4].fill_(value)
            args[5].fill_(value)
            args[1].fill_(value / 8)
            graph.replay()
            expected = reference(*args)
            stream.synchronize()
            torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_unsupported_falls_back(native, case):
    args = list(fixture(0 if case == "empty" else 4))
    if case == "ids64":
        args[4] = args[4].long()
    if case == "positions64":
        args[5] = args[5].long()
    if case == "positions2d":
        args[5] = args[5].view(-1, 1)
    if case == "norm2d":
        args[2].weight = args[2].weight.view(1, -1)
    if case == "eps":
        args[3].variance_epsilon *= 2
    if case == "tp":
        args[0].tp_size = 2
    if case == "hook":
        args[2].register_forward_hook(lambda *args: None)
    if case == "strided":
        args[1] = torch.empty(4, 12288, device="cuda", dtype=torch.bfloat16)[:, ::2]
    with torch.no_grad():
        assert mtp_frontend.try_mtp_frontend(native, *args) is None


def test_grad_mode_falls_back(native):
    args = fixture(4)
    with torch.enable_grad():
        assert mtp_frontend.try_mtp_frontend(native, *args) is None


def test_nonfinite_precision(native, value):
    args = fixture(4)
    args[1].fill_(value)
    with torch.no_grad():
        result, residual = mtp_frontend.try_mtp_frontend(native, *args)
        torch.testing.assert_close(result, reference(*args), rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0, atol=0)


def test_additional_fallback(native, case):
    args = list(fixture(4))
    if case == "unaligned":
        args[1] = torch.empty(4 * 6144 + 1, device="cuda", dtype=torch.bfloat16)[1:].view(4, 6144)
    elif case == "cpu_positions":
        args[-1] = args[-1].cpu()
    elif case == "fp16_hidden":
        args[1] = args[1].half()
    elif case == "compiled_embedding":
        args[0]._compiled_call_impl = lambda *a: None
    with torch.no_grad():
        if case == "global_hook":
            with patch.dict(torch.nn.modules.module._global_forward_hooks, {999: lambda *a: None}):
                assert mtp_frontend.try_mtp_frontend(native, *args) is None
        else:
            assert mtp_frontend.try_mtp_frontend(native, *args) is None


def test_disabled_does_not_load_or_invoke_fusion():
    mtp_frontend.native.cache_clear()
    with patch("torch.utils.cpp_extension.load", side_effect=AssertionError("disabled extension loaded")):
        op = mtp_frontend.native(enabled=False)
    assert op is None
    # None inputs deliberately fail if the disabled fast path inspects tensors.
    assert mtp_frontend.try_mtp_frontend(op, None, None, None, None, None, None) is None
    mtp_frontend.native.cache_clear()


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    suite.addTest(unittest.FunctionTestCase(test_disabled_does_not_load_or_invoke_fusion))

    def add(function, *args, cuda=True):
        def run():
            if cuda and not torch.cuda.is_available():
                raise unittest.SkipTest("CUDA required")
            function(native(), *args)

        suite.addTest(
            unittest.FunctionTestCase(run, description=function.__name__ + repr(args))
        )

    for rows in [1, 3, 4, 16, 33, 64]:
        for magnitude in [0.0, 1e-10, 1.0, 1e10, 1e20]:
            for eps in [1e-6, 1e-5]:
                add(test_native_precision, rows, magnitude, eps)
    add(test_graph_replay_changed_inputs)
    for case in [
        "empty",
        "ids64",
        "positions64",
        "eps",
        "tp",
        "hook",
        "strided",
        "positions2d",
        "norm2d",
    ]:
        add(test_unsupported_falls_back, case)
    add(test_grad_mode_falls_back)
    for value in (float("inf"), float("-inf"), float("nan")):
        add(test_nonfinite_precision, value)
    for case in ("unaligned", "cpu_positions", "fp16_hidden", "compiled_embedding", "global_hook"):
        add(test_additional_fallback, case)
    return suite


if __name__ == "__main__":
    unittest.main()
