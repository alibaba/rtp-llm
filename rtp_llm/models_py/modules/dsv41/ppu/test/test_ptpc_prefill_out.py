"""Prefill Indexer projection into request slices of one contiguous slab."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import linear as linear_module
from rtp_llm.models_py.modules.dsv41.ppu import prefill
from rtp_llm.models_py.modules.dsv41.ppu.linear import (
    PPUResidentBF16Linear,
    PPUPTPCLinear,
)


def make_linear(n=128, k=512, device="cpu"):
    weight = torch.randn(n, k, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    scale = torch.rand(n, 1, dtype=torch.float32, device=device) * 0.01 + 0.001
    return PPUPTPCLinear(weight, scale)


def assert_bits(test, actual, expected):
    test.assertEqual(actual.dtype, expected.dtype)
    test.assertEqual(actual.shape, expected.shape)
    if actual.numel():
        test.assertTrue(torch.equal(
            actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
        ))


def reference_quant(value):
    values = value.float()
    maximum = values.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    scale = maximum * 0.0022321429569274187
    payload = (values * (448.0 / maximum)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return payload.contiguous(), scale.contiguous()


def request_projection(op, values, counts, direct):
    if not direct:
        return torch.cat([op(rows) for rows in values.split(counts)])
    output = values.new_empty((values.shape[0], op.weight.shape[0]))
    for rows, target in zip(values.split(counts), output.split(counts)):
        op.forward_into(rows, target)
    return output


class PTPCPrefillOutCPU(unittest.TestCase):
    def test_exact_offset_leading_shapes_strided_input_and_zero_rows(self):
        torch.manual_seed(4141)
        op = make_linear(n=16, k=128)
        for shape in ((128,), (7, 128), (2, 3, 128), (0, 128), (2, 0, 128)):
            with self.subTest(shape=shape):
                value = torch.randn((*shape[:-1], 256), dtype=torch.bfloat16)[..., ::2]
                expected = op(value)
                rows = value.numel() // value.shape[-1]
                backing = torch.full((rows + 2, 16), 17.0, dtype=torch.bfloat16)
                output = backing[1:1 + rows].reshape(expected.shape)
                self.assertIs(op.forward_into(value, output), output)
                assert_bits(self, output, expected)
                self.assertTrue(torch.equal(
                    backing[[0, -1]], torch.full((2, 16), 17.0, dtype=torch.bfloat16)
                ))
        value = torch.zeros(4, 128, dtype=torch.bfloat16)
        output = torch.full((4, 16), 17.0, dtype=torch.bfloat16)
        assert_bits(self, op.forward_into(value, output), op(value))

    def test_rejects_invalid_outputs_before_quantization_or_mutation(self):
        op = make_linear(n=16, k=128)
        value = torch.randn(4, 128, dtype=torch.bfloat16)
        invalid = (
            (torch.full((4, 16), 17.0), TypeError, "must be BF16"),
            (torch.full((4, 15), 17.0, dtype=torch.bfloat16), ValueError, "output shape"),
            (torch.full((4, 32), 17.0, dtype=torch.bfloat16)[:, ::2], ValueError, "contiguous"),
            (torch.empty(4, 16, device="meta", dtype=torch.bfloat16), ValueError, "share the input"),
            (torch.full((4, 16), 17.0, dtype=torch.bfloat16, requires_grad=True), ValueError, "gradients"),
        )
        for output, error, message in invalid:
            with self.subTest(message=message):
                before = output.detach().clone()
                with patch.object(op, "quantize_input", side_effect=AssertionError("quantized invalid output")):
                    with self.assertRaisesRegex(error, message):
                        op.forward_into(value, output)
                if output.device.type != "meta":
                    assert_bits(self, output, before)
        with patch.object(op, "quantize_input", side_effect=AssertionError("quantized invalid output")):
            with self.assertRaisesRegex(TypeError, "strided BF16"):
                op.forward_into(value, None)

    def test_rejects_input_weight_and_scale_aliases_before_quantization(self):
        # One wide channel scale also supplies enough bytes for a BF16 output
        # view, allowing all three operand overlap contracts to be exercised.
        op = make_linear(n=128, k=128)
        value = torch.randn(1, 128, dtype=torch.bfloat16)
        aliases = (
            value,
            op.weight.view(torch.bfloat16).reshape(-1)[:128].view(1, 128),
            op.scale.view(torch.bfloat16).reshape(-1)[:128].view(1, 128),
        )
        for output in aliases:
            before = output.clone()
            with patch.object(op, "quantize_input", side_effect=AssertionError("quantized aliased output")):
                with self.assertRaisesRegex(ValueError, "must not overlap"):
                    op.forward_into(value, output)
            assert_bits(self, output, before)

    @torch.no_grad()
    def test_request_order_payload_and_scale_are_unchanged(self):
        torch.manual_seed(4142)
        op = make_linear()
        counts = (3, 0, 5, 2)
        values = torch.randn(sum(counts), 512, dtype=torch.bfloat16)
        values[3:8].zero_()
        expected = request_projection(op, values, counts, False)
        quantize = linear_module.quantize_ptpc_activation
        seen = []

        def record(rows):
            result = quantize(rows)
            seen.append((rows.clone(), tuple(part.clone() for part in result)))
            return result

        with patch.object(linear_module, "quantize_ptpc_activation", side_effect=record):
            actual = request_projection(op, values, counts, True)
        assert_bits(self, actual, expected)
        self.assertEqual([rows.shape[0] for rows, _ in seen], list(counts))
        for (rows, actual_pair), expected_rows in zip(seen, values.split(counts)):
            assert_bits(self, rows, expected_rows)
            for actual_part, expected_part in zip(actual_pair, reference_quant(expected_rows)):
                assert_bits(self, actual_part, expected_part)

    @torch.no_grad()
    def test_tiled_prefill_uses_ptpc_module_and_preserves_legacy(self):
        torch.manual_seed(4143)
        counts = (3, 5, 2)
        values = torch.randn(sum(counts), 512, dtype=torch.bfloat16)
        positions = torch.cat([torch.arange(count) for count in counts])
        requests, offset = [], 0
        for rid, count in enumerate(counts):
            rows = slice(offset, offset + count)
            requests.append((rid, 0, count, 0, rows, rows))
            offset += count
        ops = SimpleNamespace(try_encode_rotary_kv=lambda value, *args, **kwargs: value)
        freqs = torch.ones(8, 32, dtype=torch.complex64)
        for op in (make_linear(), PPUResidentBF16Linear(torch.randn(128, 512, dtype=torch.bfloat16))):
            layer = SimpleNamespace(
                index_wk=op, compressor_norm=torch.ones(512, dtype=torch.bfloat16),
                index_k_norm=torch.ones(128, dtype=torch.bfloat16),
            )
            latent = prefill._rms_norm(values, layer.compressor_norm, 1e-6)
            expected = prefill._rms_norm(
                torch.cat([op(rows) for rows in latent.split(counts)]), layer.index_k_norm, 1e-6
            )
            for direct in ("0", "1"):
                with self.subTest(ptpc=isinstance(op, PPUPTPCLinear), direct=direct):
                    with patch.dict(os.environ, {prefill._DIRECT_PRODUCER_WRITE_ENV: direct}):
                        produced = list(prefill._tiled_producer_inputs(
                            requests, ops, layer, 1e-6, freqs, values, None, positions, 1, "global_kv_1"
                        ))
                    self.assertEqual([request[0] for request, _ in produced], [0, 1, 2])
                    for (_, packed), expect in zip(produced, expected.split(counts)):
                        assert_bits(self, packed[2], expect)

    @torch.no_grad()
    def test_single_request_prefill_uses_ptpc_module(self):
        op = make_linear()
        layer = SimpleNamespace(
            index_wk=op, compressor_norm=torch.ones(512, dtype=torch.bfloat16),
            index_k_norm=torch.ones(128, dtype=torch.bfloat16),
        )
        values = torch.randn(5, 512, dtype=torch.bfloat16)
        positions = torch.arange(5)
        freqs = torch.ones(8, 32, dtype=torch.complex64)
        writes = []
        slots = SimpleNamespace(global_slots={2: positions}, indexer_slots={2: positions})
        cache = SimpleNamespace(slots={0: slots}, insert_rows=lambda *args: writes.append(args))
        ops = SimpleNamespace(encode_kv=lambda value, **kwargs: value)
        prefill._produce_global(
            ops=ops, layer=layer, geometry=SimpleNamespace(ratio=lambda layer: 1, norm_eps=1e-6),
            freqs=freqs, values=values, scores=None, positions=positions,
            request_ids=torch.zeros(5, dtype=torch.long), cache=cache, layer_id=2,
            prefill_layout=((0,), (5,)),
        )
        latent = prefill._rms_norm(values, layer.compressor_norm, 1e-6)
        expected = prefill._rms_norm(op(latent), layer.index_k_norm, 1e-6)
        expected = prefill._rope_tail(expected, freqs[positions])
        self.assertEqual([write[1] for write in writes], ["global_kv_1", "indexer_kv"])
        assert_bits(self, writes[1][3], expected)


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_PTPC_DEVICE") == "1", "explicit PPU device test"
)
class PTPCPrefillOutDevice(unittest.TestCase):
    @torch.no_grad()
    def test_native_direct_write_removes_one_cat_launch(self):
        artifact_root = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if artifact_root:
            directory = Path(artifact_root) / "ptpc-prefill-out"
            report = profile_request_projection(directory)
        else:
            with tempfile.TemporaryDirectory(prefix="ptpc-prefill-out-") as temporary:
                report = profile_request_projection(Path(temporary))
        for name in ("direct0", "direct1"):
            self.assertEqual(report[name]["native_gemm_kernel_count"], 8)
        self.assertEqual(report["direct0"]["cat_gpu_kernel_count"], 1)
        self.assertEqual(report["direct1"]["cat_gpu_kernel_count"], 0)
        self.assertEqual(
            report["direct0"]["gpu_kernel_count"] - report["direct1"]["gpu_kernel_count"], 1
        )
        print("PTPC_PREFILL_OUT_PROFILE=" + json.dumps(report, sort_keys=True))

    @torch.no_grad()
    def test_native_request_slices_and_changed_input_graph_exact(self):
        torch.manual_seed(4144)
        for n, k in ((128, 512), (1280, 512)):
            op = make_linear(n, k, "cuda")
            counts = (18, 64, 0, 32)
            values = torch.randn(sum(counts), k, dtype=torch.bfloat16, device="cuda")
            backing = torch.full((values.shape[0] + 2, n), 17.0, dtype=torch.bfloat16, device="cuda")
            output = backing[1:-1]

            def candidate():
                for rows, target in zip(values.split(counts), output.split(counts)):
                    op.forward_into(rows, target)
                return output

            expected = request_projection(op, values, counts, False)
            quantize = op.quantize_input
            seen = []

            def record(rows):
                pair = quantize(rows)
                seen.append((rows, pair))
                return pair

            with patch.object(op, "quantize_input", side_effect=record):
                assert_bits(self, candidate(), expected)
            self.assertEqual([rows.shape[0] for rows, _ in seen], list(counts))
            for rows, pair in seen:
                for actual_part, expected_part in zip(pair, reference_quant(rows)):
                    assert_bits(self, actual_part, expected_part)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = candidate()
            for magnitude in (0.0, 0.125, 2.0):
                values.copy_(torch.randn_like(values) * magnitude)
                graph.replay()
                assert_bits(self, captured, request_projection(op, values, counts, False))
                assert_bits(self, backing[[0, -1]], torch.full((2, n), 17.0, dtype=torch.bfloat16, device="cuda"))


@torch.no_grad()
def profile_request_projection(directory):
    """Component trace: DIRECT0/1 with eight native request GEMMs.

    Write complete traces and report actual device events plus CPU cat
    calls; whole-model ATen/CUB/copy classification remains the E2E gate.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(4145)
    op = make_linear(1280, 512, "cuda")
    counts = (512,) * 8
    values = torch.randn(sum(counts), 512, dtype=torch.bfloat16, device="cuda")
    expected = request_projection(op, values, counts, False)
    actual = request_projection(op, values, counts, True)
    if not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
        raise AssertionError("direct request outputs differ before profiling")
    report = {"shape": {"requests": 8, "rows": 512, "k": 512, "n": 1280}, "exact": True}
    for name, direct in (("direct0", False), ("direct1", True)):
        trace = directory / (name + ".json")
        if trace.exists():
            raise FileExistsError(trace)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as profiler:
            result = request_projection(op, values, counts, direct)
            torch.cuda.synchronize()
        profiler.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())["traceEvents"]
        kernels = [event for event in events if event.get("cat") == "kernel" and event.get("ph") == "X"]
        if not kernels:
            raise AssertionError("component trace contains no actual GPU kernel events")
        report[name] = {
            "trace": str(trace), "gpu_kernel_count": len(kernels),
            "gpu_kernel_sum_us": sum(event["dur"] for event in kernels),
            "native_gemm_kernel_count": sum(
                event["name"] in ("fp8_dense_gemm", "fp8_deep_gemm") for event in kernels
            ),
            "cat_gpu_kernel_count": sum("CatArrayBatchedCopy" in event["name"] for event in kernels),
            "aten_cat_calls": sum(event.count for event in profiler.key_averages() if event.key == "aten::cat"),
        }
        if not torch.equal(result.view(torch.uint8), expected.view(torch.uint8)):
            raise AssertionError("profiled request projection output differs")
        del result
    if report["direct0"]["aten_cat_calls"] != 1 or report["direct1"]["aten_cat_calls"] != 0:
        raise AssertionError("request projection cat contract changed")
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    unittest.main()
