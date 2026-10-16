"""Independent gates for the isolated SGLang SM90 group32 DeepGEMM port.

Parent owns GPU locks/Bazel execution. Benchmarking is opt-in and always runs
the unchanged numerical gate first. No end-to-end or best-kernel claim follows
from this component test. A failed old Humming baseline cannot veto a correct
candidate and is never timed as a performance comparator.
"""

import hashlib
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rtp_llm.models_py.modules.dsv41.sm90_dense import (
    Sm90MXFP8Linear,
    unpack_ue8m0_input_scales,
)
from rtp_llm.models_py.modules.dsv41.test.test_sm90_dense_tuning import (
    SHAPES,
    _Candidate,
    _cpu_exact_coordinate,
    _independent_scales,
    _metrics,
    _oracle,
)
from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear


def _hash(tensor):
    # contiguous()/reshape() may preserve a singleton dimension's non-unit
    # stride (e.g. packed [1,1] TMA scales with stride [1,4]). dtype-view to
    # uint8 requires unit innermost stride even when numel == 1. Allocate a
    # canonical flat destination explicitly; never modify the source payload.
    canonical = torch.empty(tensor.numel(), dtype=tensor.dtype, device=tensor.device)
    canonical.copy_(tensor.reshape(-1))
    return hashlib.sha256(canonical.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


@pytest.fixture(autouse=True)
def _require_sm90():
    assert torch.cuda.is_available(), "Dedicated target requires parent-locked CUDA GPU"
    assert torch.cuda.get_device_capability() == (9, 0), "Dedicated target requires SM90"
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


class _DeepGemmCandidate:
    def __init__(self, weight, scales):
        from rtp_llm.models_py.kernels.cuda.sm90_dense_deepgemm import (
            align_activation_scales,
            fp8_gemm_nt,
        )
        from rtp_llm.models_py.modules.dsv41.sm90_dense_deepgemm import (
            Sm90DeepGemmMXFP8Linear,
        )

        self.adapter = Sm90DeepGemmMXFP8Linear(weight, scales)
        self.weight = weight
        self.scales = scales.float().contiguous()
        self.gemm = fp8_gemm_nt
        self.align = align_activation_scales
        self.K = weight.shape[1]

    def quantize(self, x):
        return V41MXFP8Linear._quantize_input(self, x)

    def prepare_scales(self, packed):
        return self.align(unpack_ue8m0_input_scales(packed))

    def prepared(self, q, unpacked, out):
        self.gemm(q.reshape(-1, self.K), unpacked, self.weight, self.scales,
                  out.reshape(-1, self.weight.shape[0]))
        return out

    def quantized(self, q, packed, out):
        return self.adapter.forward_quantized(q, packed, out=out)

    def full_chain(self, x, out):
        return self.quantized(*self.quantize(x), out)


class _ExactScaleBf16Candidate:
    """Diagnostic only: decode the SAME g32 payload to BF16, never original x.

    For the bounded exponent distribution used here these products are exactly
    BF16-representable. The subsequent GEMM still needs the independent gate.
    Production forward uses Torch kernels and is NOT a promotion candidate.
    """

    def __init__(self, weight, scales):
        self.K = weight.shape[1]
        expanded = scales.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
        self.weight = (weight.float() * expanded[:weight.shape[0], :self.K]).bfloat16()
        self.weight_t = self.weight.T

    def quantize(self, x):
        return V41MXFP8Linear._quantize_input(self, x)

    def decode_activation(self, q, packed):
        rows = q.numel() // self.K
        scales = unpack_ue8m0_input_scales(packed)[:, :self.K // 32]
        return (q.reshape(rows, self.K // 32, 32).float()
                * scales.unsqueeze(-1)).reshape(rows, self.K).bfloat16()

    def prepared(self, a, out):
        torch.mm(a, self.weight_t, out=out.reshape(a.shape[0], self.weight.shape[0]))
        return out

    def quantized(self, q, packed, out):
        return self.prepared(self.decode_activation(q, packed), out)

    def full_chain(self, x, out):
        return self.quantized(*self.quantize(x), out)


def _check_bits(actual, expected):
    torch.testing.assert_close(actual.contiguous().view(torch.int16),
                               expected.contiguous().view(torch.int16), rtol=0, atol=0)


def _numerical_failure_details(method, x, q, packed, graph_result, oracle, weight, scales):
    """Failure-only evidence; never changes the acceptance tolerance or inputs."""
    rows, k = q.numel() // q.shape[-1], q.shape[-1]
    flat_q = q.reshape(rows, k)
    actual = graph_result.reshape_as(oracle)
    difference = (actual.float() - oracle.float()).abs()
    excess = difference - (.02 + .015 * oracle.float().abs())
    excess = torch.where(torch.isfinite(excess), excess, float("inf"))
    count = int((excess > 0).sum())
    worst = torch.topk(excess.flatten(), min(3, count)).indices.cpu().tolist()
    coordinates = [divmod(index, oracle.shape[1]) for index in worst]
    details = dict(input_hashes=dict(q=_hash(q), packed=_hash(packed)),
                   violation_count=count, anchors=[])
    eager_quantized = torch.empty_like(graph_result)
    method.quantized(q, packed, eager_quantized)
    eager_full = torch.empty_like(graph_result)
    method.full_chain(x, eager_full)
    details["eager_quantized_oracle"] = _metrics(eager_quantized, oracle)
    details["eager_full_chain_oracle"] = _metrics(eager_full, oracle)
    details["graph_vs_eager_quantized"] = _metrics(graph_result, eager_quantized)
    details["graph_vs_eager_full_chain"] = _metrics(graph_result, eager_full)
    for row, column in coordinates:
        anchor = _cpu_exact_coordinate(flat_q, packed, weight, scales, row, column)
        exact = anchor["cpu_fsum_fp64"]
        details["anchors"].append(dict(
            row=row, column=column, graph=float(actual[row, column]),
            eager_quantized=float(eager_quantized.reshape_as(oracle)[row, column]),
            eager_full_chain=float(eager_full.reshape_as(oracle)[row, column]),
            cuda_fp32_oracle=float(oracle[row, column]),
            cpu_fsum_fp64=exact, cpu_torch_dot_fp64=anchor["cpu_torch_dot_fp64"],
            cpu_fsum_rounded_bf16=anchor["cpu_fsum_rounded_bf16"],
            cuda_oracle_minus_cpu_rounded=float(oracle[row, column]) - anchor["cpu_fsum_rounded_bf16"],
            graph_minus_cpu_exact=float(actual[row, column]) - exact,
        ))
    # Full independent CPU FP64 GEMM is bounded to small diagnostic fixtures.
    # Large-model-shape cases retain the scalar math.fsum + FP64 anchors above.
    if rows * weight.numel() <= 16 * 1024**2:
        cpu_q, cpu_packed = flat_q.cpu(), packed.cpu()
        a = (cpu_q.reshape(rows, k // 32, 32).double()
             * _independent_scales(cpu_packed, k // 32).double().unsqueeze(-1)).reshape(rows, k)
        b_scale = scales.cpu().float().double().repeat_interleave(32, 0).repeat_interleave(32, 1)
        b = weight.cpu().float().double() * b_scale[:weight.shape[0], :k]
        cpu_fp64 = (a @ b.T).bfloat16().to(oracle.device)
        details["cpu_fp64_full_oracle"] = dict(
            cuda_fp32_vs_cpu_fp64=_metrics(oracle, cpu_fp64),
            graph_vs_cpu_fp64=_metrics(graph_result, cpu_fp64))
    try:
        humming = Sm90MXFP8Linear(weight, scales)
        humming_result = humming.forward_quantized(q, packed)
        details["humming_same_q"] = dict(oracle=_metrics(humming_result, oracle),
            graph_difference=_metrics(graph_result, humming_result),
            anchors=[dict(row=r, column=c, value=float(humming_result.reshape_as(oracle)[r, c]))
                     for r, c in coordinates])
    except Exception as exc:
        details["humming_same_q"] = dict(decision="BLOCKED", failure=str(exc))
    if isinstance(method, _DeepGemmCandidate):
        from rtp_llm.models_py.kernels.cuda.sm90_dense_deepgemm import CONFIGS

        prepared = method.prepare_scales(packed)
        observations = {}
        for config in CONFIGS:
            try:
                value = torch.empty_like(oracle)
                method.gemm(flat_q, prepared, method.weight, method.scales, value, config=config)
                observations[config] = dict(oracle=_metrics(value, oracle),
                    graph_difference=_metrics(value, actual),
                    anchors=[dict(row=r, column=c, value=float(value[r, c])) for r, c in coordinates])
            except Exception as exc:
                observations[config] = dict(decision="BLOCKED", failure=str(exc))
        details["deepgemm_all_configs_same_q"] = observations
    return details


def _changed_input_graph(method, x, out, weight, scales, *, records=None):
    """Capture real quantization + scale conversion + GEMM, not precomputed q."""
    saved = x.clone()
    for _ in range(3):
        method.full_chain(x, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        method.full_chain(x, out)
    original_q, original_s = method.quantize(saved)
    # Caller attaches this list to its persisted record BEFORE entering the
    # helper, so a failed assertion cannot discard the failing attempt metrics.
    if records is None:
        records = []
    try:
        torch.manual_seed(514100 + x.numel() // x.shape[-1])
        for multiplier in (16.0, 0.03125):
            x.copy_(torch.randn_like(x) * multiplier)
            q, packed = method.quantize(x)
            payload_changed = int((q.view(torch.uint8) != original_q.view(torch.uint8)).sum())
            scale_words_changed = int((packed != original_s).sum())
            assert payload_changed and scale_words_changed
            graph.replay()
            result = out.clone()
            expected = _oracle(q, packed, weight, scales)
            metrics = _metrics(result, expected)
            records.append(dict(multiplier=multiplier, payload_bytes_changed=payload_changed,
                                scale_words_changed=scale_words_changed, oracle=metrics))
            method.full_chain(x, out)
            records[-1]["graph_vs_eager"] = _metrics(result, out)
            if metrics["tolerance_violations"]:
                records[-1]["numerical_diagnostic"] = _numerical_failure_details(
                    method, x, q, packed, result, expected, weight, scales)
            # Graph faithfulness is checked before independent accuracy: an
            # eager/graph-identical numerical failure is not a graph bug.
            _check_bits(out, result)
            records[-1]["graph_vs_eager_bitwise"] = True
            torch.testing.assert_close(result.reshape_as(expected), expected, rtol=.015, atol=.02)
            for _ in range(3):
                graph.replay()
                _check_bits(out, result)
            records[-1].update(graph_vs_eager_bitwise=True, graph_repeatability=True)
    finally:
        x.copy_(saved)
    return records


def _measure(fn, flush, path, *, prepared):
    """Same 3-warmup/5-event-sample protocol as dense tuning; Chrome only."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    end.synchronize()
    spans = []
    for _ in range(5):
        if flush is not None:
            flush.zero_()
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        end.synchronize()
        spans.append(start.elapsed_time(end))
    if flush is not None:
        flush.zero_()
        torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profiler:
        fn()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path))
    counts, durations = Counter(), Counter()
    trace = json.loads(path.read_text())
    for event in trace["traceEvents"]:
        if event.get("cat") == "kernel" and event.get("ph") == "X":
            counts[event["name"]] += 1
            durations[event["name"]] += event["dur"]
    assert counts, "Missing GPU kernel evidence"
    if prepared:
        assert sum(counts.values()) == 1, counts
    return dict(stream_span_ms=spans, median_ms=statistics.median(spans),
                profile=dict(source="Chrome cat=kernel ph=X only", trace=str(path),
                             pure_kernel_launch_count=sum(counts.values()),
                             kernel_sum_us_per_iter=sum(durations.values()),
                             kernels=[dict(name=n, count=counts[n], us=durations[n])
                                      for n in sorted(counts)]))


def _run(rows, shape_name, n, k, *, benchmark=False, leading_shape=None):
    directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    assert directory, "Persist acceptance and timing evidence under Bazel outputs"
    case = Path(directory) / f"dense_deepgemm_{shape_name}_m{rows}_{'bench' if benchmark else 'gate'}"
    case.mkdir(parents=True, exist_ok=False)
    record = dict(M=rows, N=n, K=k, shape_name=shape_name, seed=5141,
                  decision="NOT_RUN", tolerance=dict(rtol=.015, atol=.02),
                  scope="Synthetic-weight dense component only; not model correctness/performance",
                  arithmetic="Original E4M3 A group1x32 / B block32x32; lossless FP32 scale conversion",
                  timing_scope="full_chain includes BF16 quantization, packed-scale conversion/alignment and GEMM; prepared_gemm excludes all preparation")
    try:
        # Keep the original failing stress fixture's RNG order unchanged.
        torch.manual_seed(5141)
        weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.pow(2.0, torch.randint(-3, 3, ((n + 31) // 32, k // 32),
                                             device="cuda").float()).to(torch.float8_e8m0fnu)
        raw_weight, raw_scales = weight.clone(), scales.clone()
        method = _DeepGemmCandidate(weight, scales)
        shape = leading_shape or (rows,)
        x = torch.randn(*shape, k, device="cuda", dtype=torch.bfloat16)
        q, packed = method.quantize(x)
        prepared = method.prepare_scales(packed)
        assert prepared.dtype == torch.float32
        assert prepared.stride(0) == 1 and prepared.stride(1) % 4 == 0
        torch.testing.assert_close(prepared, _independent_scales(packed, k // 32), rtol=0, atol=0)
        oracle = _oracle(q, packed, raw_weight, raw_scales)
        record["hashes"] = dict(x=_hash(x), q=_hash(q), packed=_hash(packed),
                                weight=_hash(raw_weight), scale=_hash(raw_scales))
        original_q, original_packed = q.view(torch.uint8).clone(), packed.clone()
        out = torch.full((*shape, n), float("nan"), device="cuda", dtype=torch.bfloat16)
        start = time.perf_counter()
        assert method.quantized(q, packed, out) is out
        torch.cuda.synchronize()
        record["first_call_compile_wall_ms"] = (time.perf_counter() - start) * 1000
        result = out.clone()
        record["candidate_oracle"] = _metrics(result, oracle)
        record["candidate_output_sha256"] = _hash(result)
        if rows == 4096 and shape_name in ("shared_w13", "shared_w2"):
            row, column = (2844, 2709) if shape_name == "shared_w13" else (990, 124)
            anchor = _cpu_exact_coordinate(q, packed, raw_weight, raw_scales, row, column)
            record["known_failure_anchor"] = dict(row=row, column=column,
                candidate=float(result[row, column]), oracle=float(oracle[row, column]),
                cpu_fsum_fp64=anchor["cpu_fsum_fp64"], cpu_torch_dot_fp64=anchor["cpu_torch_dot_fp64"],
                cpu_fsum_rounded_bf16=anchor["cpu_fsum_rounded_bf16"])
        # Persist failure metrics even when the first independent gate rejects.
        (case / "result.json").write_text(json.dumps(record, indent=2))
        torch.testing.assert_close(result.reshape_as(oracle), oracle, rtol=.015, atol=.02)
        for _ in range(3):
            out.fill_(float("nan"))
            assert method.quantized(q, packed, out) is out
            _check_bits(out, result)
        allocated = method.adapter.forward_quantized(q, packed)
        assert allocated.shape == out.shape
        _check_bits(allocated, result)
        method.full_chain(x, out)
        _check_bits(out, result)
        method.prepared(q, prepared, out)
        _check_bits(out, result)
        record["changed_input_graph"] = []
        _changed_input_graph(method, x, out, raw_weight, raw_scales,
                             records=record["changed_input_graph"])
        torch.testing.assert_close(q.view(torch.uint8), original_q, rtol=0, atol=0)
        torch.testing.assert_close(packed, original_packed, rtol=0, atol=0)
        torch.testing.assert_close(weight.view(torch.uint8), raw_weight.view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(scales.view(torch.uint8), raw_scales.view(torch.uint8), rtol=0, atol=0)
        record.update(repeatability=True, full_chain_bitwise=True, prepared_bitwise=True,
                      changed_input_graph_pass=True, input_payload_unchanged=True)
        methods = [("deepgemm_g32", method, prepared)]
        # Old Humming is diagnostic, never the correctness oracle. Only compare
        # performance if it passes the SAME independent fixture and graph gate.
        baseline_adapter = Sm90MXFP8Linear(raw_weight, raw_scales)
        baseline = _Candidate(SimpleNamespace(_sm90_linear=baseline_adapter,
                                              _quantize_input=method.quantize))
        baseline.quantized(q, packed, out)
        baseline_result = out.clone()
        metrics = _metrics(out, oracle)
        record["humming_diagnostic"] = dict(oracle=metrics, benchmark_eligible=False)
        if metrics["tolerance_violations"] == 0:
            try:
                for _ in range(3):
                    baseline.quantized(q, packed, out)
                    _check_bits(out, baseline_result)
                baseline.full_chain(x, out)
                _check_bits(out, baseline_result)
                # The captured function is the baseline's actual full_chain.
                baseline.quantize = method.quantize
                record["humming_diagnostic"]["graph"] = []
                _changed_input_graph(baseline, x, out, raw_weight, raw_scales,
                                     records=record["humming_diagnostic"]["graph"])
                record["humming_diagnostic"]["benchmark_eligible"] = True
                methods.append(("humming_valid_baseline", baseline, unpack_ue8m0_input_scales(packed)))
            except AssertionError as exc:
                # Rejected comparator must not veto a correct DG candidate.
                record["humming_diagnostic"]["additional_gate_failure"] = str(exc)
                record["speedup_vs_humming"] = "FORBIDDEN: baseline failed additional accuracy/invariance gate"
        else:
            record["speedup_vs_humming"] = "FORBIDDEN: baseline failed independent accuracy gate"
        if benchmark:
            flush = torch.empty(256 * 1024**2, device="cuda", dtype=torch.uint8)
            measurements = {name: {} for name, _, _ in methods}
            for repeat in range(3):
                ordered = methods if repeat % 2 == 0 else list(reversed(methods))
                for name, candidate, candidate_scales in ordered:
                    for scope, fn in (
                        ("full_chain", lambda candidate=candidate: candidate.full_chain(x, out)),
                        ("prepared_gemm", lambda candidate=candidate, s=candidate_scales: candidate.prepared(q, s, out)),
                    ):
                        modes = measurements[name].setdefault(scope, {"hot": [], "cold": []})
                        for mode in ("hot", "cold"):
                            modes[mode].append(_measure(fn, flush if mode == "cold" else None,
                                case / f"{name}_{scope}_{mode}_r{repeat}.trace.json", prepared=scope == "prepared_gemm"))
            record.update(measurements=measurements, l2_flush_bytes=flush.numel())
        record["decision"] = "BOUNDED_SUPPORT; component only, no production promotion"
    except Exception as exc:
        record.update(decision="REJECT", failure=str(exc))
        raise
    finally:
        (case / "result.json").write_text(json.dumps(record, indent=2))
        print("DENSE_DEEPGEMM_GATE " + json.dumps(record), flush=True)


@pytest.mark.parametrize("rows,n,k,leading_shape", [
    (1, 256, 128, None), (7, 256, 512, None),
    (15, 320, 640, (3, 5)), (129, 256, 512, None),
])
def test_dense_deepgemm_fixture(rows, n, k, leading_shape):
    _run(rows, f"fixture_n{n}_k{k}", n, k, leading_shape=leading_shape)


@pytest.mark.parametrize("rows", [1, 16, 512, 4096], ids=["m1", "m16", "m512", "m4096"])
@pytest.mark.parametrize("shape_name", ["merged_qkv", "tp8_q_b", "shared_w13", "shared_w2"])
def test_dense_deepgemm_actual(rows, shape_name):
    _run(rows, shape_name, *SHAPES[shape_name])


@pytest.mark.skipif(os.environ.get("DSV41_DENSE_DEEPGEMM_BENCHMARK", "0") != "1",
                    reason="Benchmark opt-in; correctness targets remain mandatory")
@pytest.mark.parametrize("rows", [1, 16, 512, 4096], ids=["m1", "m16", "m512", "m4096"])
@pytest.mark.parametrize("shape_name", ["shared_w13", "shared_w2"])
def test_dense_deepgemm_benchmark(rows, shape_name):
    _run(rows, shape_name, *SHAPES[shape_name], benchmark=True)


@pytest.mark.parametrize("rows", [1, 16, 512, 4096], ids=["m1", "m16", "m512", "m4096"])
@pytest.mark.parametrize("shape_name", ["shared_w13", "shared_w2"])
def test_dense_bf16_fallback(rows, shape_name):
    """Independent test selection: DG rejection must not prevent this evidence.

    Run without pytest -x when evaluating both backends. BF16 passing does not
    turn a failing DG test into a pass or imply native-forward/Torch-free gates.
    """
    n, k = SHAPES[shape_name]
    directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    assert directory
    case = Path(directory) / f"dense_bf16_fallback_{shape_name}_m{rows}"
    case.mkdir(parents=True, exist_ok=False)
    record = dict(M=rows, N=n, K=k, seed=5141, decision="NOT_RUN",
                  tolerance=dict(rtol=.015, atol=.02), backend="exactscale_bf16_diagnostic",
                  scope="Independent component; NOT native-forward-compatible and not production promotion",
                  arithmetic="Original g32 quantized payload and scale decoded to BF16; NOT original unquantized x",
                  timing_scope="full_chain = BF16 quantization + packed-scale decode + g32 activation decode + BF16 GEMM; prepared_gemm = already-decoded A/B BF16 GEMM, not FP8 inputs")
    try:
        torch.manual_seed(5141)
        weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.pow(2.0, torch.randint(-3, 3, ((n + 31) // 32, k // 32),
                                             device="cuda").float()).to(torch.float8_e8m0fnu)
        torch.cuda.synchronize()
        begin = time.perf_counter()
        method = _ExactScaleBf16Candidate(weight, scales)
        torch.cuda.synchronize()
        record.update(static_weight_decode_wall_ms=(time.perf_counter() - begin) * 1000,
                      static_decoded_weight_bytes=method.weight.numel() * method.weight.element_size(),
                      original_weight_bytes=weight.numel() * weight.element_size(),
                      memory_scope="Test retains raw FP8 and decoded BF16 copies; decoded BF16 alone is 2x payload bytes")
        x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
        q, packed = method.quantize(x)
        oracle = _oracle(q, packed, weight, scales)
        prepared = method.decode_activation(q, packed)
        # Demonstrate exact dequantization for THIS fixture independently of
        # the implementation's unpack helper; do not infer all exponent safety.
        exact_a = (q.reshape(rows, k // 32, 32).float()
                   * _independent_scales(packed, k // 32).unsqueeze(-1)).reshape(rows, k)
        exact_b = weight.float() * scales.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
        torch.testing.assert_close(prepared.float(), exact_a, rtol=0, atol=0)
        torch.testing.assert_close(method.weight.float(), exact_b, rtol=0, atol=0)
        record["hashes"] = dict(x=_hash(x), q=_hash(q), packed=_hash(packed),
                                weight=_hash(weight), scale=_hash(scales))
        out = torch.full((rows, n), float("nan"), device="cuda", dtype=torch.bfloat16)
        assert method.quantized(q, packed, out) is out
        result = out.clone()
        record["candidate_oracle"] = _metrics(result, oracle)
        (case / "result.json").write_text(json.dumps(record, indent=2))
        torch.testing.assert_close(result, oracle, rtol=.015, atol=.02)
        for _ in range(3):
            out.fill_(float("nan"))
            method.quantized(q, packed, out)
            _check_bits(out, result)
        method.full_chain(x, out)
        _check_bits(out, result)
        method.prepared(prepared, out)
        _check_bits(out, result)
        record["changed_input_graph"] = []
        _changed_input_graph(method, x, out, weight, scales, records=record["changed_input_graph"])
        if os.environ.get("DSV41_DENSE_DEEPGEMM_BENCHMARK", "0") == "1":
            flush = torch.empty(256 * 1024**2, device="cuda", dtype=torch.uint8)
            measurements = {}
            for scope, fn in (("full_chain", lambda: method.full_chain(x, out)),
                              ("prepared_gemm", lambda: method.prepared(prepared, out))):
                modes = measurements.setdefault(scope, {"hot": [], "cold": []})
                for repeat in range(3):
                    for mode in ("hot", "cold"):
                        modes[mode].append(_measure(fn, flush if mode == "cold" else None,
                            case / f"bf16_{scope}_{mode}_r{repeat}.trace.json", prepared=scope == "prepared_gemm"))
            record.update(measurements=measurements, l2_flush_bytes=flush.numel())
        record["decision"] = "BOUNDED_SUPPORT; diagnostic only, original quantization retained"
    except Exception as exc:
        record.update(decision="REJECT", failure=str(exc))
        raise
    finally:
        (case / "result.json").write_text(json.dumps(record, indent=2))
        print("DENSE_BF16_FALLBACK_GATE " + json.dumps(record), flush=True)


def test_dense_deepgemm_changed_input_diagnostic():
    """Expected-failure diagnostic, NOT an acceptance or timing test.

    The original acceptance test remains failed. This selected test only proves
    that the failure reproduced and its same-input arithmetic evidence exists.
    If a future kernel fixes it, this diagnostic must be revised, not silently
    treated as continued evidence of the old failure.
    """
    directory = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
    case = directory / "dense_deepgemm_diagnostic_n256_k512_m7_gate"
    with pytest.raises(AssertionError):
        _run(7, "diagnostic_n256_k512", 256, 512)
    record = json.loads((case / "result.json").read_text())
    assert record["decision"] == "REJECT"
    assert record["candidate_oracle"]["tolerance_violations"] == 0
    attempts = record["changed_input_graph"]
    assert attempts and attempts[-1]["oracle"]["tolerance_violations"] > 0
    diagnostic = attempts[-1]["numerical_diagnostic"]
    assert diagnostic["anchors"]
    assert "cpu_fp64_full_oracle" in diagnostic
    assert set(diagnostic["deepgemm_all_configs_same_q"]) == {"m64n64", "m64n128", "m128n128"}
    summary = dict(decision="DIAGNOSTIC_ONLY; original acceptance REJECT retained",
                   numerical_gate_pass=False, evidence=str(case / "result.json"))
    (case / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2))
    print("DENSE_DEEPGEMM_DIAGNOSTIC_ONLY " + json.dumps(summary), flush=True)
