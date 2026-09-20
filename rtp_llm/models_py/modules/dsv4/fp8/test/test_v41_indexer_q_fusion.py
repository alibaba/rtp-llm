"""Byte-exact V4.1 indexer Q fusion and changing CUDA graph inputs."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _v41_decode_indexer as indexer
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_indexer_q_triton as fused


def make_case(batch, span, seed=41, weights_dtype=torch.bfloat16):
    torch.manual_seed(seed)
    q = torch.randn(batch, span, 32, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(batch, span, 32, device="cuda", dtype=weights_dtype)
    angles = torch.randn(4096, 32, device="cuda", dtype=torch.float32)
    freqs = torch.polar(torch.ones_like(angles), angles)
    positions = torch.randint(0, 4096, (batch * span,), device="cuda")
    return dict(q=q, weights=weights, freqs_cis=freqs, positions=positions)


def baseline(case):
    """The exact original eight-operation chain, independent of dispatch."""
    q, weights = case["q"], case["weights"]
    weights = weights.float() * (q.shape[-1] * q.shape[-2]) ** -0.5
    freqs = case["freqs_cis"][case["positions"]]
    payload, sf = indexer.prepare_indexer_q(q, weights, freqs)
    return payload, sf, weights


def candidate(case):
    result = fused.try_fused_indexer_q(**case)
    if result is None:
        raise AssertionError("expected fused Q preparation, got fallback")
    return result


def assert_identical(actual, expected):
    for value, reference, dtype in zip(
        actual, expected, (torch.int8, torch.int32, torch.float32)
    ):
        assert value.dtype == dtype
        assert value.shape == reference.shape
        assert value.is_contiguous()
        # Integer views distinguish negative zero as well as quantization bits.
        torch.testing.assert_close(
            value.view(torch.uint8), reference.view(torch.uint8), rtol=0, atol=0
        )


def _metadata():
    def tensor(shape, dtype):
        return SimpleNamespace(
            device=torch.device("cuda"),
            shape=shape,
            ndim=len(shape),
            dtype=dtype,
            requires_grad=False,
            is_contiguous=lambda: True,
        )

    return dict(
        q=tensor((4, 6, 32, 128), torch.bfloat16),
        weights=tensor((4, 6, 32), torch.bfloat16),
        freqs_cis=tensor((4096, 32), torch.complex64),
        positions=tensor((24,), torch.int64),
    )


class V41IndexerQFusionCPU(unittest.TestCase):
    def test_default_enabled_and_supported_metadata(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            torch.cuda, "get_device_capability", return_value=(10, 3)
        ):
            self.assertTrue(fused.is_supported(**_metadata()))
            case = _metadata()
            case["weights"].dtype = torch.float32
            self.assertTrue(fused.is_supported(**case))

    def test_disable_and_cpu_short_circuit_without_cuda_queries(self):
        with patch.dict(os.environ, {"DSV41_FUSED_INDEXER_Q": "0"}), patch.object(
            torch.cuda, "get_device_capability", side_effect=AssertionError
        ):
            self.assertFalse(fused.is_supported(**_metadata()))
        with patch.dict(os.environ, {"DSV41_FUSED_INDEXER_Q": "1"}), patch.object(
            torch.cuda, "get_device_capability", side_effect=AssertionError
        ):
            q = torch.empty(1, 1, 32, 128, dtype=torch.bfloat16)
            self.assertIsNone(
                fused.try_fused_indexer_q(
                    q,
                    torch.empty(1, 1, 32, dtype=torch.bfloat16),
                    torch.empty(7, 32, dtype=torch.complex64),
                    torch.zeros(1, dtype=torch.int64),
                )
            )

    def test_layout_dtype_shape_architecture_and_grad_gates(self):
        with patch.dict(os.environ, {"DSV41_FUSED_INDEXER_Q": "1"}), patch.object(
            torch.cuda, "get_device_capability", return_value=(10, 3)
        ):
            for key, field, value in (
                ("q", "dtype", torch.float32),
                ("q", "shape", (4, 6, 64, 128)),
                ("q", "shape", (0, 6, 32, 128)),
                ("weights", "dtype", torch.float16),
                ("weights", "shape", (4, 6, 31)),
                ("freqs_cis", "dtype", torch.complex128),
                ("freqs_cis", "shape", (4096, 16)),
                ("positions", "dtype", torch.float32),
                ("positions", "shape", (4, 6)),
                ("positions", "device", torch.device("cpu")),
            ):
                case = _metadata()
                setattr(case[key], field, value)
                with self.subTest(key=key, field=field, value=value):
                    self.assertFalse(fused.is_supported(**case))
            for key in _metadata():
                case = _metadata()
                case[key].is_contiguous = lambda: False
                self.assertFalse(fused.is_supported(**case))
            self.assertFalse(fused.is_supported(**_metadata(), rope_head_dim=32))
            case = _metadata()
            case["q"].requires_grad = True
            with torch.enable_grad():
                self.assertFalse(fused.is_supported(**case))
            with torch.no_grad():
                self.assertTrue(fused.is_supported(**case))
        with patch.dict(os.environ, {"DSV41_FUSED_INDEXER_Q": "1"}), patch.object(
            torch.cuda, "get_device_capability", return_value=(9, 0)
        ):
            self.assertFalse(fused.is_supported(**_metadata()))

    def test_disabled_wrapper_preserves_gather_and_weight_arithmetic(self):
        q = torch.empty(1, 2, 32, 128, dtype=torch.bfloat16)
        weights = torch.randn(1, 2, 32, dtype=torch.bfloat16)
        freqs = torch.randn(7, 32, dtype=torch.complex64)
        positions = torch.tensor([6, -2])
        payload, sf = object(), object()
        with patch.dict(os.environ, {"DSV41_FUSED_INDEXER_Q": "0"}), patch.object(
            indexer, "prepare_indexer_q", return_value=(payload, sf)
        ) as old:
            result = indexer.prepare_indexer_q_and_weights(q, weights, freqs, positions)
        self.assertIs(result[0], payload)
        self.assertIs(result[1], sf)
        torch.testing.assert_close(result[2], weights.float() / 64, rtol=0, atol=0)
        self.assertIs(old.call_args.args[0], q)
        self.assertIs(old.call_args.args[1], result[2])
        torch.testing.assert_close(
            old.call_args.args[2], freqs[positions], rtol=0, atol=0
        )

    def test_execution_errors_propagate_without_silent_fallback(self):
        with patch.object(
            fused, "try_fused_indexer_q", side_effect=RuntimeError("kernel failure")
        ), patch.object(indexer, "prepare_indexer_q") as old:
            with self.assertRaisesRegex(RuntimeError, "kernel failure"):
                indexer.prepare_indexer_q_and_weights(**_metadata())
            old.assert_not_called()


class V41IndexerQFusionCUDA(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("V4.1 fused MXFP4 indexer Q requires SM100 family")
        self.env = patch.dict(os.environ, {"DSV41_FUSED_INDEXER_Q": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)

    @torch.no_grad()
    def test_real_shapes_and_input_immutability(self):
        for batch, span in (
            (1, 1),
            (2, 1),
            (4, 1),
            (1, 6),
            (2, 6),
            (4, 6),
            (1, 7),
            (1, 31),
            (1, 64),
            (1, 128),
            (1, 1024),
        ):
            for dtype in (torch.bfloat16, torch.float32):
                with self.subTest(batch=batch, span=span, weights_dtype=dtype):
                    case = make_case(batch, span, weights_dtype=dtype)
                    copies = {key: value.clone() for key, value in case.items()}
                    assert_identical(candidate(case), baseline(case))
                    assert_identical(
                        indexer.prepare_indexer_q_and_weights(**case), baseline(case)
                    )
                    for key, value in case.items():
                        torch.testing.assert_close(value, copies[key], rtol=0, atol=0)

    @torch.no_grad()
    def test_fp4_midpoints_scale_boundaries_and_signed_zero(self):
        case = make_case(1, 6)
        case["freqs_cis"].fill_(complex(1, 0))
        case["positions"].copy_(
            torch.tensor([0, 1, 4095, -1, -4096, 17], device="cuda")
        )
        midpoints = torch.tensor(
            [0.0, -0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
            device="cuda",
            dtype=torch.bfloat16,
        )
        values = torch.cat(
            (
                torch.nextafter(midpoints, torch.full_like(midpoints, -torch.inf)),
                midpoints,
                torch.nextafter(midpoints, torch.full_like(midpoints, torch.inf)),
            )
        )
        groups = case["q"].view(-1, 32)
        groups.zero_()
        groups[:, : values.numel()] = values
        groups[:, -1] = 6.0
        groups[1::4].neg_()
        six = torch.tensor(6.0, device="cuda", dtype=torch.bfloat16)
        groups[2::8, -1] = torch.nextafter(six, torch.full_like(six, torch.inf))
        groups[3::8, -1] = torch.nextafter(six, torch.full_like(six, -torch.inf))
        groups[4::8].zero_()
        groups[5::8].fill_(-0.0)
        groups[6::8].mul_(torch.finfo(torch.bfloat16).tiny)
        groups[7::8].mul_(2.0**-130)
        case["weights"].view(-1)[:2] = torch.tensor([0.0, -0.0], device="cuda")
        assert_identical(candidate(case), baseline(case))

    @torch.no_grad()
    def test_noncontiguous_falls_back_byte_exact(self):
        case = make_case(2, 6)
        padded = torch.empty(2, 6, 32, 256, device="cuda", dtype=torch.bfloat16)
        padded[..., ::2].copy_(case["q"])
        case["q"] = padded[..., ::2]
        self.assertIsNone(fused.try_fused_indexer_q(**case))
        assert_identical(indexer.prepare_indexer_q_and_weights(**case), baseline(case))

    @torch.no_grad()
    def test_paged_scoring_matches_legacy_contract(self):
        from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_decode_indexer import (
            make_case as make_scoring_case,
        )

        if not indexer.is_supported(torch.device("cuda"), 128):
            self.skipTest("DeepGEMM paged MXFP4 scoring unavailable")
        prepared = make_case(4, 6)
        for logical, ratio in ((128, 1), (64, 2)):
            with self.subTest(logical=logical, ratio=ratio):
                case, _ = make_scoring_case(4, 6, 2053, 128, logical, ratio)
                case["q"] = prepared["q"]
                case["weights"] = prepared["weights"].float() / 64
                case["freqs_cis"] = prepared["freqs_cis"][prepared["positions"]]
                legacy = indexer.score_decode_indexer(**case)
                case["weights"] = prepared["weights"]
                case["freqs_cis"] = prepared["freqs_cis"]
                case["positions"] = prepared["positions"]
                actual = indexer.score_decode_indexer(**case)
                self.assertIsNotNone(actual)
                torch.testing.assert_close(actual, legacy, rtol=0, atol=0)
                with patch.dict(os.environ, {"DSV41_FUSED_INDEXER_Q": "0"}):
                    disabled = indexer.score_decode_indexer(**case)
                torch.testing.assert_close(disabled, legacy, rtol=0, atol=0)

    @torch.no_grad()
    def test_cuda_graph_replay_reads_all_current_inputs(self):
        for batch, span in ((1, 1), (2, 1), (4, 1), (1, 6), (2, 6), (4, 6)):
            with self.subTest(batch=batch, span=span):
                case = make_case(batch, span)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        baseline(case)
                        candidate(case)
                torch.cuda.current_stream().wait_stream(stream)
                graph_old, graph_new = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph_old, stream=stream):
                    old = baseline(case)
                with torch.cuda.graph(graph_new, stream=stream):
                    new = candidate(case)
                for replay in range(4):
                    updated = make_case(batch, span, seed=91 + replay)
                    for key, value in case.items():
                        value.copy_(updated[key])
                    graph_old.replay()
                    graph_new.replay()
                    assert_identical(new, old)
                    assert_identical(new, baseline(case))


if __name__ == "__main__":
    unittest.main()
