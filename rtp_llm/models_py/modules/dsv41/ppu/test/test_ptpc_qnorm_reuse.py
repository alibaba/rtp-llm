"""Component gates for Q-LoRA RMSNorm's shared whole-row PTPC producer.

The oracle is the frozen ATen norm/quant expression plus native GEMM. No
checkpoint, complete model, cache publication, or service is exercised.
"""

from collections import Counter
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu.linear import PPUResidentBF16Linear
from rtp_llm.models_py.modules.dsv41.ppu.test.test_ptpc_input_reuse import (
    ENV,
    assert_bits,
    make_linear,
    make_norm,
    make_projection,
    old_linear,
    old_norm,
    old_projection,
    old_quant,
    quantizations,
)


ROWS = (1, 3, 4, 8, 24, 32)


def real_projection():
    """Use the actual WqA/WqB/index-query dimensions with resident weights."""
    op, hidden = make_projection("cuda")
    op.wq_b = make_linear(1280, 32768, "cuda", 4180)
    op.index_wq = make_linear(1280, 4096, "cuda", 4181)
    return op, hidden


def cpu_projection():
    op, hidden = make_projection()
    op.wq_a = make_linear(hidden, 1280, "cpu", 4182)
    op.q_norm = make_norm(1280, "cpu", 4183)
    op.wq_b = make_linear(1280, 32, "cpu", 4184)
    op.index_wq = make_linear(1280, 16, "cpu", 4185)
    return op, hidden


def projection(op, value, *, indexed, reuse=True, fuse=True):
    qr, q, kv, pair = op.project_query_kv(
        value, prepare_indexer=indexed, reuse_ptpc=reuse, fuse_qr_norm=fuse
    )
    result = (qr, q, kv)
    if indexed:
        result += (op.project_indexer_query(qr, quantized_input=pair),)
    return result


def contains_identity(value, targets, visited=None):
    """Inspect Python ownership without reading tensor values or device state."""
    if any(value is target for target in targets):
        return True
    visited = set() if visited is None else visited
    if id(value) in visited:
        return False
    visited.add(id(value))
    if isinstance(value, dict):
        return any(contains_identity(v, targets, visited) for v in value.values())
    if isinstance(value, (tuple, list)):
        return any(contains_identity(v, targets, visited) for v in value)
    return False


class _QNormReuseBase(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, ENV)
        environment.start()
        self.addCleanup(environment.stop)
        torch.manual_seed(4170)

    def assert_outputs(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            assert_bits(self, left, right)

    def assert_pair(self, normalized, pair):
        self.assertIsNotNone(pair, "compatible device Q-LoRA must prepare PTPC")
        self.assertEqual(len(pair), 2)
        self.assertEqual(pair[0].dtype, torch.float8_e4m3fn)
        self.assertEqual(pair[1].dtype, torch.float32)
        self.assertEqual(tuple(pair[0].shape), tuple(normalized.shape))
        self.assertEqual(tuple(pair[1].shape), (*normalized.shape[:-1], 1))
        self.assertTrue(pair[0].is_contiguous())
        self.assertTrue(pair[1].is_contiguous())
        for actual, expected in zip(pair, old_quant(normalized)):
            assert_bits(self, actual, expected)

    def assert_not_retained(self, op, pair):
        if pair is None:
            return
        for name in ("q_norm", "wq_a", "wq_b", "wkv", "index_wq"):
            self.assertFalse(
                contains_identity(vars(getattr(op, name)), (pair, *pair)), name
            )


class PTPCQNormReuseCPU(_QNormReuseBase):
    @torch.no_grad()
    def test_cpu_fallback_keeps_norm_formula_and_one_call(self):
        norm = make_norm(1280, "cpu", 4171)
        for rows in ROWS:
            for value in (
                torch.randn(rows, 1280, dtype=torch.bfloat16),
                torch.zeros(rows, 1280, dtype=torch.bfloat16),
            ):
                with self.subTest(rows=rows, zero=not bool(value.count_nonzero())):
                    before = norm.calls
                    normalized, pair = norm.forward_with_ptpc_input(value)
                    self.assertEqual(norm.calls - before, 1)
                    assert_bits(self, normalized, old_norm(value, norm))
                    self.assertIsNone(pair)

    @torch.no_grad()
    def test_cpu_unsupported_empty_and_legacy_norms_fall_back_once(self):
        for mode in (ENV["RTP_V41_PPU_QUANT_MODE"], "legacy"):
            with patch.dict(os.environ, {"RTP_V41_PPU_QUANT_MODE": mode}):
                for value in (
                    torch.empty(0, 1280, dtype=torch.bfloat16),
                    torch.randn(1280, dtype=torch.bfloat16),
                    torch.randn(3, 2560, dtype=torch.bfloat16)[:, ::2],
                    torch.randn(3, 1296, dtype=torch.bfloat16),
                ):
                    norm = make_norm(value.shape[-1], "cpu", 4172)
                    with self.subTest(mode=mode, shape=value.shape):
                        before = norm.calls
                        normalized, pair = norm.forward_with_ptpc_input(value)
                        self.assertEqual(norm.calls - before, 1)
                        assert_bits(self, normalized, old_norm(value, norm))
                        self.assertIsNone(pair)

    @torch.no_grad()
    def test_cpu_projection_counts_and_fallback_consumers_are_unchanged(self):
        for indexed in (False, True):
            op, hidden = cpu_projection()
            value = torch.randn(3, hidden, dtype=torch.bfloat16)
            for fusion, expected_quant in (("1", 2), ("0", 4 if indexed else 3)):
                with self.subTest(indexed=indexed, fusion=fusion):
                    before = op.q_norm.calls
                    with patch.dict(
                        os.environ, {"RTP_V41_PPU_PTPC_FUSION": fusion}
                    ), quantizations() as calls:
                        actual = projection(op, value, indexed=indexed, reuse=True)
                    self.assertEqual(len(calls), expected_quant)
                    self.assertEqual(op.q_norm.calls - before, 1)
                    self.assert_outputs(
                        actual, old_projection(op, value, indexed=indexed)
                    )
            # Either legacy QR consumer declines the combined producer.
            for name in (("wq_b", "index_wq") if indexed else ("wq_b",)):
                op, hidden = cpu_projection()
                source = getattr(op, name)
                setattr(
                    op,
                    name,
                    PPUResidentBF16Linear(source.weight.float().to(torch.bfloat16)),
                )
                before = op.q_norm.calls
                with patch.object(
                    op.q_norm,
                    "forward_with_ptpc_input",
                    side_effect=AssertionError(
                        "mixed consumers must decline before norm"
                    ),
                ):
                    actual = projection(op, value, indexed=indexed, reuse=True)
                self.assertEqual(op.q_norm.calls - before, 1)
                self.assert_outputs(actual, old_projection(op, value, indexed=indexed))

    @torch.no_grad()
    def test_default_prepared_path_does_not_opt_in_to_qnorm_producer(self):
        op, hidden = cpu_projection()
        value = torch.randn(3, hidden, dtype=torch.bfloat16)
        with patch.object(
            op.q_norm,
            "forward_with_ptpc_input",
            side_effect=AssertionError("default phase must keep prepared-only norm"),
        ), quantizations() as calls:
            qr, q, kv, pair = op.project_query_kv(
                value, prepare_indexer=True, reuse_ptpc=True
            )
            index = op.project_indexer_query(qr, quantized_input=pair)
        self.assertEqual(len(calls), 2)
        self.assertEqual(op.q_norm.calls, 1)
        self.assert_outputs((qr, q, kv, index), old_projection(op, value, indexed=True))

    @torch.no_grad()
    def test_actual_decode_entry_points_pass_explicit_opt_in(self):
        from rtp_llm.models_py.modules.dsv41.ppu.dspark_attention import (
            forward_graph_query,
            forward_query,
        )
        from rtp_llm.models_py.modules.dsv41.ppu.graph_attention_layer import (
            forward_graph_attention,
        )

        class ProjectionReached(Exception):
            pass

        observed = []

        def stop_at_projection(value, **kwargs):
            observed.append((tuple(value.shape), kwargs))
            raise ProjectionReached

        geometry = SimpleNamespace(hidden_size=64)
        attn = SimpleNamespace(
            geometry=geometry,
            layer_id=0,
            compress_ratio=0,
            is_index_source=False,
            project_query_kv=stop_at_projection,
        )
        positions = torch.arange(4)
        table = torch.zeros(4, 1)
        state = SimpleNamespace(
            cache=SimpleNamespace(max_seq_len=4),
            verify=False,
            idle=False,
            metadata=SimpleNamespace(positions=positions, valid_mask=positions >= 0),
            freqs=[table],
            shared={},
        )
        for verify in (False, True):
            state.verify = verify
            with self.assertRaises(ProjectionReached):
                forward_graph_attention(
                    attn, torch.zeros(4, 64, dtype=torch.bfloat16), state
                )
            shape, kwargs = observed.pop()
            self.assertEqual(shape, (4, 64))
            self.assertEqual(kwargs["reuse_ptpc"], verify)
            self.assertEqual(kwargs.get("fuse_qr_norm", False), verify)
        runtime = SimpleNamespace(
            current=state, geometry=geometry, freqs_for=lambda layer_id: table
        )
        model = SimpleNamespace(
            shared_runtime=runtime,
            shared_weight_views=SimpleNamespace(geometry=geometry),
            v4=SimpleNamespace(layers=[SimpleNamespace(attn=attn)]),
        )
        value = torch.zeros(1, 3, 64, dtype=torch.bfloat16)
        args = (model, 0, value, positions[:3], torch.zeros(1), torch.ones(1))
        for call in (
            lambda: forward_graph_query(*args),
            lambda: forward_query(*args, None),
        ):
            with self.assertRaises(ProjectionReached):
                call()
            shape, kwargs = observed.pop()
            self.assertEqual(shape, (3, 64))
            self.assertTrue(kwargs["reuse_ptpc"])
            self.assertTrue(kwargs["fuse_qr_norm"])


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_PTPC_DEVICE") == "1", "explicit PPU component test"
)
class PTPCQNormReuseDevice(_QNormReuseBase):
    @torch.no_grad()
    def test_real_projections_pair_bytes_and_shared_native_consumers(self):
        op, hidden = real_projection()
        for rows in ROWS:
            for indexed in (False, True):
                with self.subTest(rows=rows, indexed=indexed):
                    value = torch.randn(
                        rows, hidden, device="cuda", dtype=torch.bfloat16
                    )
                    expected = old_projection(op, value, indexed=indexed)
                    observed = []
                    original = op.q_norm.forward_with_ptpc_input

                    def record(raw):
                        result = original(raw)
                        observed.append(result)
                        return result

                    before = op.q_norm.calls
                    with patch.object(
                        op.q_norm, "forward_with_ptpc_input", new=record
                    ), patch.object(
                        op.wq_b, "forward_quantized", wraps=op.wq_b.forward_quantized
                    ) as wq_b, patch.object(
                        op.index_wq,
                        "forward_quantized",
                        wraps=op.index_wq.forward_quantized,
                    ) as index_wq, quantizations() as calls:
                        actual = projection(op, value, indexed=indexed, reuse=True)
                    self.assertEqual(op.q_norm.calls - before, 1)
                    self.assertEqual(
                        len(calls), 1, "only shared hidden-x quant remains"
                    )
                    self.assertEqual(len(observed), 1)
                    normalized, pair = observed[0]
                    assert_bits(self, normalized, expected[0])
                    self.assert_pair(expected[0], pair)
                    self.assert_outputs(actual, expected)
                    self.assertEqual(wq_b.call_count, 1)
                    self.assertIs(wq_b.call_args.args[0], pair[0])
                    self.assertIs(wq_b.call_args.args[1], pair[1])
                    self.assertEqual(index_wq.call_count, int(indexed))
                    if indexed:
                        self.assertIs(index_wq.call_args.args[0], pair[0])
                        self.assertIs(index_wq.call_args.args[1], pair[1])
                    self.assert_not_retained(op, pair)

    @torch.no_grad()
    def test_graph_recomputes_pair_when_input_and_gamma_change(self):
        op, hidden = real_projection()
        for rows in ROWS:
            for indexed in (False, True):
                with self.subTest(rows=rows, indexed=indexed):
                    value = torch.randn(
                        rows, hidden, device="cuda", dtype=torch.bfloat16
                    )
                    projection(op, value, indexed=indexed, reuse=True)
                    observed = []
                    original = op.q_norm.forward_with_ptpc_input

                    def record(raw):
                        result = original(raw)
                        observed.append(result)
                        return result

                    graph = torch.cuda.CUDAGraph()
                    with patch.object(op.q_norm, "forward_with_ptpc_input", new=record):
                        with torch.cuda.graph(graph):
                            actual = projection(op, value, indexed=indexed, reuse=True)
                    self.assertEqual(len(observed), 1)
                    normalized, pair = observed[0]
                    self.assertIsNotNone(pair)
                    for magnitude in (0.0, 0.125, 2.0):
                        value.copy_(torch.randn_like(value) * magnitude)
                        op.q_norm.weight.copy_(torch.randn_like(op.q_norm.weight))
                        graph.replay()
                        expected = old_projection(op, value, indexed=indexed)
                        self.assert_outputs(actual, expected)
                        assert_bits(self, normalized, expected[0])
                        self.assert_pair(expected[0], pair)
                        self.assert_not_retained(op, pair)
                    _, next_pair = op.q_norm.forward_with_ptpc_input(
                        old_linear(op.wq_a, value)
                    )
                    self.assertIsNotNone(next_pair)
                    self.assertIsNot(next_pair, pair)
                    self.assertIsNot(next_pair[0], pair[0])
                    self.assertIsNot(next_pair[1], pair[1])

    @torch.no_grad()
    def test_device_zero_empty_strided_and_rollback_contracts(self):
        norm = make_norm(1280, "cuda", 4173)
        zero = torch.zeros(3, 1280, device="cuda", dtype=torch.bfloat16)
        normalized, pair = norm.forward_with_ptpc_input(zero)
        assert_bits(self, normalized, old_norm(zero, norm))
        self.assert_pair(normalized, pair)
        for value in (
            torch.randn(3, 2560, device="cuda", dtype=torch.bfloat16)[:, ::2],
            torch.randn(1280, device="cuda", dtype=torch.bfloat16),
            torch.randn(3, 1296, device="cuda", dtype=torch.bfloat16),
        ):
            fallback = make_norm(value.shape[-1], "cuda", 4174)
            before = fallback.calls
            normalized, pair = fallback.forward_with_ptpc_input(value)
            self.assertEqual(fallback.calls - before, 1)
            assert_bits(self, normalized, old_norm(value, fallback))
            self.assertIsNone(pair)
        for mode, fusion in (
            ("legacy", "1"), (ENV["RTP_V41_PPU_QUANT_MODE"], "0")
        ):
            with patch.dict(
                os.environ,
                {
                    "RTP_V41_PPU_QUANT_MODE": mode,
                    "RTP_V41_PPU_PTPC_FUSION": fusion,
                },
            ):
                fallback = make_norm(1280, "cuda", 4174)
                before = fallback.calls
                normalized, pair = fallback.forward_with_ptpc_input(zero)
                self.assertEqual(fallback.calls - before, 1)
                assert_bits(self, normalized, old_norm(zero, fallback))
                self.assertIsNone(pair)
        for name in ("wq_b", "index_wq"):
            op, hidden = make_projection("cuda")
            source = getattr(op, name)
            setattr(
                op,
                name,
                PPUResidentBF16Linear(source.weight.float().to(torch.bfloat16)),
            )
            value = torch.randn(4, hidden, device="cuda", dtype=torch.bfloat16)
            before = op.q_norm.calls
            with patch.object(
                op.q_norm,
                "forward_with_ptpc_input",
                side_effect=AssertionError("mixed consumers must not prepare QR"),
            ), quantizations() as calls:
                actual = projection(op, value, indexed=True)
            self.assertEqual(op.q_norm.calls - before, 1)
            self.assertEqual(len(calls), 2)
            self.assert_outputs(actual, old_projection(op, value, indexed=True))
        op, hidden = make_projection("cuda")
        value = torch.randn(4, hidden, device="cuda", dtype=torch.bfloat16)
        for fusion, fuse in (("1", False), ("0", True)):
            before = op.q_norm.calls
            with patch.dict(
                os.environ, {"RTP_V41_PPU_PTPC_FUSION": fusion}
            ), quantizations() as calls:
                actual = projection(op, value, indexed=True, fuse=fuse)
            self.assertEqual(op.q_norm.calls - before, 1)
            self.assertEqual(len(calls), 2 if fusion == "1" else 4)
            self.assert_outputs(actual, old_projection(op, value, indexed=True))
        empty = zero[:0]
        with patch(
            "rtp_llm.models_py.modules.dsv41.ppu.ptpc_norm_quant._kernels",
            side_effect=AssertionError("empty norm must not launch a kernel"),
        ):
            normalized, pair = norm.forward_with_ptpc_input(empty)
        assert_bits(self, normalized, empty)
        if pair is not None:
            self.assert_pair(normalized, pair)
        op, hidden = make_projection("cuda")
        with patch.object(op.wq_a, "_gemm") as wqa, patch.object(
            op.wq_b, "_gemm"
        ) as wqb, patch.object(op.wkv, "_gemm") as wkv, patch.object(
            op.index_wq, "_gemm"
        ) as index_wq:
            actual = projection(
                op,
                torch.empty(0, hidden, device="cuda", dtype=torch.bfloat16),
                indexed=True,
                reuse=True,
            )
        self.assertTrue(all(not tensor.numel() for tensor in actual))
        for gemm in (wqa, wqb, wkv, index_wq):
            gemm.assert_not_called()

    @torch.no_grad()
    def test_m3_m4_replay_removes_only_one_quantizer(self):
        op, hidden = make_projection("cuda")
        with tempfile.TemporaryDirectory(prefix="ptpc-qnorm-reuse-") as temporary:
            root = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", temporary))
            root.mkdir(parents=True, exist_ok=True)
            for rows, indexed in ((3, False), (4, True)):
                value = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
                expected = old_projection(op, value, indexed=indexed)
                measured = {}
                for baseline in (True, False):
                    name = "prepared-only" if baseline else "combined"
                    path = root / f"qnorm-{name}-M{rows}-indexed{int(indexed)}.json"
                    counts, actual = profile_projection(
                        op, value, indexed, baseline, path
                    )
                    self.assert_outputs(actual, expected)
                    measured[name] = counts
                baseline = measured["prepared-only"]
                candidate = measured["combined"]
                base_quant = sum(v for k, v in baseline.items() if "quantize_rows" in k)
                next_quant = sum(
                    v for k, v in candidate.items() if "quantize_rows" in k
                )
                self.assertEqual((base_quant, next_quant), (2, 1))
                self.assertEqual(
                    Counter(
                        {k: v for k, v in baseline.items() if "quantize_rows" not in k}
                    ),
                    Counter(
                        {k: v for k, v in candidate.items() if "quantize_rows" not in k}
                    ),
                )
                for counts in (baseline, candidate):
                    gemms = sum(v for k, v in counts.items() if "fp8_dense_gemm" in k)
                    self.assertEqual(gemms, 4 if indexed else 3, counts)
                print(
                    "PTPC_QNORM_REUSE_PROFILE="
                    + json.dumps(
                        {
                            "rows": rows,
                            "indexed": indexed,
                            "prepared_only_kernels": dict(baseline),
                            "combined_kernels": dict(candidate),
                            "quant_kernels": [base_quant, next_quant],
                            "nonquant_multiset_unchanged": True,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )


def profile_projection(op, value, indexed, baseline, path):
    """Record real replay kernels, including the untouched norm/GEMM chain."""
    if path.exists():
        raise FileExistsError(path)
    for _ in range(3):
        projection(op, value, indexed=indexed, reuse=True, fuse=not baseline)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = projection(op, value, indexed=indexed, reuse=True, fuse=not baseline)
    graph.replay()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        graph.replay()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    counts = Counter(
        event["name"]
        for event in events
        if event.get("cat") == "kernel" and event.get("ph") == "X"
    )
    if not counts:
        raise AssertionError("profiler must contain actual device kernels")
    return counts, output


if __name__ == "__main__":
    unittest.main()
