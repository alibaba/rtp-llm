"""Bounded Humming0.1.16 dense tuning; parent owns all GPU execution.

No weight repack, quantization change, Stream-K, FP16 accumulation or production
dispatch mutation. Original dense tolerance is rtol=.015/atol=.02, NOT MoE's.
"""

import copy
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import re
import statistics
import struct
import time
import zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rtp_llm.models_py.modules.dsv41.sm90_dense import unpack_ue8m0_input_scales
from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear


# SOURCE_FACT plus rank0 Input Dims from named traces in round6_dense_notes.md.
# CP8 attention is NOT TP8: wq_b and wo_b have different N/K.
SHAPES = {
    "merged_qkv": (1792, 5120),
    "tp8_q_b": (4096, 1280),
    "cp8_q_b": (32768, 1280),
    "o_a_group": (1024, 4096),
    "tp8_o_b": (5120, 1024),
    "cp8_o_b": (5120, 8192),
    "shared_w13": (4608, 5120),
    "shared_w2": (5120, 2304),
}

@pytest.fixture(autouse=True)
def _require_sm90():
    # Dedicated acceptance target: missing hardware is a failure, not an
    # all-skipped suite that could be misreported as successful validation.
    assert torch.cuda.is_available(), "Dedicated target requires parent-locked CUDA GPU"
    assert torch.cuda.get_device_capability() == (9, 0), "Dedicated target requires SM90"


def _independent_scales(packed, groups):
    shifts = torch.arange(4, dtype=torch.int64, device=packed.device) * 8
    exponents = ((packed.to(torch.int64).unsqueeze(-1) >> shifts) & 255).reshape(packed.shape[0], -1)[:, :groups]
    scales = torch.exp2(exponents.float() - 127)
    return torch.where(exponents == 255, float("nan"), scales)


def _oracle(q, packed, weight, block_scales):
    """Independent extraction and 32x32 expansion, FP32 GEMM, one BF16 cast."""
    rows, k = q.numel() // q.shape[-1], q.shape[-1]
    scales = _independent_scales(packed, k // 32)
    a = (q.reshape(rows, k // 32, 32).float() * scales.unsqueeze(-1)).reshape(rows, k)
    bs = block_scales.float().repeat_interleave(32, 0).repeat_interleave(32, 1)
    b = weight.float() * bs[:weight.shape[0], :k]
    return (a @ b.T).bfloat16()


def _metrics(actual, expected):
    a, b = actual.reshape_as(expected).float(), expected.float()
    diff = (a - b).abs()
    finite = torch.isfinite(a) & torch.isfinite(b)
    return dict(elements=a.numel(), unequal=int((a != b).sum()),
                bitwise_unequal=int((actual.reshape_as(expected).contiguous().view(torch.int16)
                                     != expected.contiguous().view(torch.int16)).sum()),
                tolerance_violations=int(((~finite) | (diff > .02 + .015 * b.abs())).sum()),
                max_abs=float(diff.max()), rms_abs=float(diff.square().mean().sqrt()),
                finite=bool(finite.all()))


def _hash(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _selected_default(adapter, rows):
    from humming.kernel.humming import HummingKernel

    _, _, intervals = HummingKernel._resolve_configs(
        adapter.config.to_str(), adapter.compute_config, None, adapter.weight.device)
    assert isinstance(intervals, list), "Record actual default interval dispatch, not get_config(M)"
    selected = [entry for entry in intervals if entry[0] < rows <= entry[1]]
    assert len(selected) == 1
    low, high, config = selected[0]
    return (low, high), copy.deepcopy(config)


def _candidates(adapter, rows, baseline):
    """Two bounded candidates over the SAME LayerConfig/packed weight buffers."""
    assert baseline["block_shape"][2] == baseline["warp_shape"][2] == 64
    assert not baseline["use_stream_k"]
    configs = []
    # Pipeline-only control; no arithmetic order or operand layout change.
    stage3 = copy.deepcopy(baseline)
    stage3["num_stages"] = 3
    configs.append(("stage3", stage3))
    # Dense-specific tile, not the already-rejected routed W4A16 K32 sweep.
    tile = copy.deepcopy(baseline)
    bm, wm = (8, 8) if rows <= 8 else (64, 32)
    tile.update(block_shape=(bm, 128, 64), warp_shape=(wm, 32, 64),
                num_stages=3, num_ctas_per_sm=2, use_stream_k=False,
                use_tma=False, use_warp_spec=False, use_mbarrier=False)
    configs.append((f"m{bm}n128_s3", tile))
    selected = os.environ.get("DSV41_DENSE_CANDIDATES")
    selected = set(selected.split(",")) if selected else None
    seen = {json.dumps(baseline, sort_keys=True)}
    result, excluded = [], []
    for name, config in configs:
        encoded = json.dumps(config, sort_keys=True)
        if selected is not None and name not in selected:
            excluded.append(dict(name=name, reason="not requested"))
            continue
        if encoded in seen:
            excluded.append(dict(name=name, reason="duplicate of already evaluated config"))
            continue
        seen.add(encoded)
        assert config["block_shape"][2] == config["warp_shape"][2] == 64
        assert not config["use_stream_k"] and not config.get("use_f16_accum", False)
        result.append((name, config))
    return result, excluded


class _Candidate:
    def __init__(self, linear, tuning=None):
        self.linear = linear
        self.adapter = linear._sm90_linear
        # None means untouched baseline interval dispatch. All candidate JSON
        # serialization occurs outside measurements, never per-call.
        self.tuning = None if tuning is None else json.dumps(tuning)

    def prepared(self, q, unpacked, out):
        a = self.adapter
        kwargs = {} if self.tuning is None else {"tuning_config": self.tuning}
        return a._forward(
            a.config, inputs=q.reshape(-1, a.K).contiguous(), input_scale=unpacked,
            weight=a.weight, weight_scale=a.weight_scale, weight_scale_2=a.weight_scale_2,
            outputs=out.reshape(-1, a.N), locks=a.locks,
            compute_config=a.compute_config, **kwargs)

    def quantized(self, q, packed, out):
        if self.tuning is None:
            # Use production validation/scaleprep/forward for baseline chain.
            return self.adapter.forward_quantized(q, packed, out=out)
        self.prepared(q, unpack_ue8m0_input_scales(packed), out)
        return out

    def full_chain(self, x, out):
        q, packed = self.linear._quantize_input(x)
        return self.quantized(q, packed, out)


def _profile(fn, flush, path, *, prepared):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    if flush is not None:
        flush.zero_()
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profiler:
        fn()
        torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - start) * 1000
    profiler.export_chrome_trace(str(path))
    counts, sums = Counter(), Counter()
    for event in profiler.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            counts[event.name] += 1
            sums[event.name] += event.time_range.elapsed_us()
    kernels = [name for name in counts if "memcpy" not in name.lower() and "memset" not in name.lower()]
    humming_count = sum(counts[name] for name in kernels if re.search(r"(?:^|::|\s)humming(?:[<(]|$)", name))
    result = dict(iterations=1, trace=str(path), profiler_context_wall_ms=wall_ms,
                  gpu_events=[dict(name=name, count=counts[name], us=sums[name]) for name in sorted(counts)],
                  kernel_sum_us_per_iter=sum(sums[name] for name in kernels),
                  pure_kernel_launch_count=sum(counts[name] for name in kernels), humming_launch_count=humming_count)
    assert humming_count == 1, result
    if prepared:
        assert sum(counts[name] for name in kernels) == 1, result
    return result


def _measure(fn, flush, path, *, prepared):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    end.record()
    end.synchronize()
    spans = []
    for _ in range(5):
        if flush is not None:
            flush.zero_()
        torch.cuda.synchronize()
        begin.record()
        fn()
        end.record()
        end.synchronize()
        spans.append(begin.elapsed_time(end))
    return dict(stream_span_ms=spans, median_ms=statistics.median(spans),
                profile=_profile(fn, flush, path, prepared=prepared))


def _run(rows, shape_name, n, k, *, benchmark, leading_shape=None):
    import humming

    assert importlib.metadata.version("humming-kernels") == "0.1.16"
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    assert directory, "Persist all candidate results/traces under Bazel outputs"
    case = Path(directory) / f"dense_{shape_name}_m{rows}_n{n}_k{k}"
    case.mkdir(parents=True, exist_ok=False)
    record = dict(M=rows, N=n, K=k, shape_name=shape_name, tolerance=dict(rtol=.015, atol=.02),
                  decision="NOT_RUN", humming_version="0.1.16", humming_file=humming.__file__,
                  scope="One dense projection/group; not routed MoE or combined 8-group wo_a or model performance",
                  timing_scope="full_chain = BF16 g32 quant + packed-scale preparation + GEMM; prepared_gemm = prequantized payload/preexpanded scales + GEMM",
                  inputs="Synthetic weights/activations, original dense test scale distribution; actual shape only")
    try:
        torch.manual_seed(5141)
        weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.pow(2.0, torch.randint(-3, 3, ((n + 31) // 32, k // 32), device="cuda").float()).to(torch.float8_e8m0fnu)
        # Sm90MXFP8Linear transform owns its packed destination; oracle copy
        # makes a future in-place upstream transform detectable independently.
        raw_weight, raw_scales = weight.clone(), scales.clone()
        linear = V41MXFP8Linear(weight, scales)
        adapter = linear._sm90_linear
        assert not adapter.config.use_fused_e8m0_scale and not adapter.config.use_int_weight_scale
        compute = json.loads(adapter.compute_config)
        assert compute["use_batch_invariant"] and not compute["use_f16_accum"]
        x_shape = leading_shape or (rows,)
        x = torch.randn(*x_shape, k, device="cuda", dtype=torch.bfloat16)
        assert x.numel() // k == rows
        q, packed = linear._quantize_input(x)
        q_copy, packed_copy = q.view(torch.uint8).clone(), packed.clone()
        unpacked = unpack_ue8m0_input_scales(packed)
        torch.testing.assert_close(unpacked[:, :k // 32], _independent_scales(packed, k // 32), rtol=0, atol=0)
        oracle = _oracle(q, packed, raw_weight, raw_scales)
        interval, baseline_config = _selected_default(adapter, rows)
        candidates, excluded = _candidates(adapter, rows, baseline_config)
        record.update(baseline_selected_interval=interval, baseline_tuning=baseline_config,
                      compute_config=compute, layer_config=json.loads(adapter.config.to_str()),
                      excluded_candidates=excluded, leading_shape=list(x_shape),
                      hashes=dict(x=_hash(x), q=_hash(q), packed=_hash(packed), weight=_hash(raw_weight), scale=_hash(raw_scales)))
        buffers_before = [(value.data_ptr(), value._version) for value in (adapter.weight, adapter.weight_scale)]
        methods = [("baseline", _Candidate(linear), baseline_config)] + [
            (name, _Candidate(linear, config), config) for name, config in candidates]
        observations = []
        accepted = []
        baseline_output = None
        out = torch.empty((*x_shape, n), device="cuda", dtype=torch.bfloat16)
        for name, method, config in methods:
            warm_start = time.perf_counter()
            result = method.quantized(q, packed, out).clone()
            torch.cuda.synchronize()
            warmup_ms = (time.perf_counter() - warm_start) * 1000
            if baseline_output is None:
                baseline_output = result
            metrics = _metrics(result, oracle)
            versus_baseline = _metrics(result, baseline_output)
            repeatable = True
            for _ in range(3):
                repeated = method.quantized(q, packed, out)
                repeatable &= torch.equal(repeated.view(torch.int16), result.view(torch.int16))
            full = method.full_chain(x, out)
            full_equal = torch.equal(full.view(torch.int16), result.view(torch.int16))
            method.prepared(q, unpacked, out)
            prepared_equal = torch.equal(out.view(torch.int16), result.view(torch.int16))
            passed = not metrics["tolerance_violations"] and versus_baseline["bitwise_unequal"] == 0 and repeatable and full_equal and prepared_equal
            observation = dict(name=name, config=config, oracle=metrics, baseline=versus_baseline,
                               repeatability=repeatable, full_chain_bitwise=full_equal,
                               prepared_bitwise=prepared_equal, compile_first_call_wall_ms=warmup_ms,
                               output_sha256=_hash(result), decision="BOUNDED_SUPPORT" if passed else "REJECT")
            observations.append(observation)
            record["candidates"] = observations
            (case / "result.json").write_text(json.dumps(record, indent=2))
            print("DENSE_TUNING_GATE " + json.dumps(dict(shape=shape_name, M=rows, **observation)), flush=True)
            if name == "baseline":
                torch.testing.assert_close(result.reshape_as(oracle), oracle, rtol=.015, atol=.02)
                assert passed, "Baseline invariance/fullchain gate failed"
            if passed:
                accepted.append((name, method))
        torch.testing.assert_close(q.view(torch.uint8), q_copy, rtol=0, atol=0)
        torch.testing.assert_close(packed, packed_copy, rtol=0, atol=0)
        assert buffers_before == [(value.data_ptr(), value._version) for value in (adapter.weight, adapter.weight_scale)]
        if benchmark and len(accepted) > 1:
            flush = torch.empty(256 * 1024**2, device="cuda", dtype=torch.uint8)
            results = {name: {} for name, _ in accepted}
            for repeat in range(3):
                ordered = accepted if repeat % 2 == 0 else list(reversed(accepted))
                for name, method in ordered:
                    functions = (("full_chain", lambda method=method: method.full_chain(x, out)),
                                 ("prepared_gemm", lambda method=method: method.prepared(q, unpacked, out)))
                    for scope, fn in functions:
                        scoped = results[name].setdefault(scope, {"hot": [], "cold": []})
                        for mode in ("hot", "cold"):
                            scoped[mode].append(_measure(fn, flush if mode == "cold" else None,
                                case / f"{name}_{scope}_{mode}_r{repeat}.trace.json", prepared=scope == "prepared_gemm"))
            record["measurements"] = results
            record["l2_flush_bytes"] = flush.numel()
            for name, scopes in results.items():
                for scope, modes in scopes.items():
                    for mode, samples in modes.items():
                        print(f"DENSE_TUNING_TIME {shape_name} M{rows} {name} {scope} {mode} "
                              f"span_ms={statistics.median(s['median_ms'] for s in samples):.6f} "
                              f"kernel_us={statistics.median(s['profile']['kernel_sum_us_per_iter'] for s in samples):.3f}", flush=True)
        record["decision"] = "BOUNDED_SUPPORT" if all(r["decision"] != "REJECT" for r in observations) else "REJECT"
        record["production_change"] = "NONE; no end-to-end speedup established"
        (case / "result.json").write_text(json.dumps(record, indent=2))
        assert record["decision"] == "BOUNDED_SUPPORT", "Unchanged independent/bitwise gate rejected candidate; see persisted metrics"
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def test_dense_tuning_fixture():
    _run(17, "fixture", 256, 512, benchmark=False)


def test_dense_tuning_grouped_output_fixture():
    # Leading output dimensions, not a new grouped-expert GEMM arithmetic path.
    _run(15, "leading_3x5", 320, 640, benchmark=False, leading_shape=(3, 5))


@pytest.mark.parametrize("rows", [1, 8, 4096, 16384], ids=["m1", "m8", "m4096", "m16384"])
@pytest.mark.parametrize("shape_name", list(SHAPES))
def test_dense_tuning_actual(rows, shape_name):
    n, k = SHAPES[shape_name]
    _run(rows, shape_name, n, k, benchmark=os.environ.get("DSV41_DENSE_BENCHMARK", "1") == "1")


def _cpu_exact_coordinate(q, packed, weight, scales, row, column):
    """Independent scalar g32/block32 decode + fsum, without GPU GEMM oracle.

    Fixture products are finite dyadic numbers. Python double exactly represents
    individual decoded E4M3 products; fsum avoids FP32/BLAS accumulation order.
    A CPU torch FP64 dot is a separately accumulated cross-check, not a new gate.
    """
    q_values = q[row].float().cpu().tolist()
    words = packed[row].to(torch.int64).cpu().tolist()
    b_values = weight[column].float().cpu().tolist()
    b_exponents = scales[column // 32].view(torch.uint8).cpu().tolist()
    a_exponents = [((int(words[group // 4]) >> ((group % 4) * 8)) & 255)
                   for group in range(weight.shape[1] // 32)]
    assert all(e < 255 for e in a_exponents + b_exponents), "Finite diagnostic fixture required"
    a = [value * math.ldexp(1.0, a_exponents[index // 32] - 127)
         for index, value in enumerate(q_values)]
    b = [value * math.ldexp(1.0, b_exponents[index // 32] - 127)
         for index, value in enumerate(b_values)]
    products = [av * bv for av, bv in zip(a, b)]
    accurate = math.fsum(products)
    double_dot = float(torch.dot(torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)))
    rounded = float(torch.tensor(accurate, dtype=torch.float64).bfloat16())
    a_subnormal = [0 < abs(value) < 2**-6 for value in q_values]
    b_subnormal = [0 < abs(value) < 2**-6 for value in b_values]
    ftz_products = [0.0 if a_subnormal[i] or b_subnormal[i] else value
                    for i, value in enumerate(products)]
    ftz_exact = math.fsum(ftz_products)

    def f32(value):
        return struct.unpack("<f", struct.pack("<f", value))[0]

    group_records = []
    accumulated, ftz_accumulated = 0.0, 0.0
    for group in range(len(a_exponents)):
        begin, end = group * 32, (group + 1) * 32
        partial = math.fsum(q_values[i] * b_values[i] for i in range(begin, end))
        ftz_partial = math.fsum(q_values[i] * b_values[i] for i in range(begin, end)
                                if not a_subnormal[i] and not b_subnormal[i])
        scale = math.ldexp(1.0, a_exponents[group] + b_exponents[group] - 254)
        # Model the source's separate raw K32 partial and FP32 scaled C1.
        # This does NOT assume hardware WGMMA produces an exact K32 partial.
        accumulated = f32(accumulated + f32(f32(partial) * scale))
        ftz_accumulated = f32(ftz_accumulated + f32(f32(ftz_partial) * scale))
        group_records.append(dict(group=group, exact_raw_partial=partial,
                                  fp32_raw_partial=f32(partial), scale=scale,
                                  accumulated_fp32=accumulated,
                                  ftz_raw_partial=ftz_partial, ftz_accumulated_fp32=ftz_accumulated))
    return dict(cpu_fsum_fp64=accurate, cpu_torch_dot_fp64=double_dot,
                fp64_dot_minus_fsum=double_dot - accurate, cpu_fsum_rounded_bf16=rounded,
                a_scale_bytes_minmax=[min(a_exponents), max(a_exponents)],
                b_scale_bytes_minmax=[min(b_exponents), max(b_exponents)],
                sum_abs_products=math.fsum(abs(value) for value in products),
                largest_abs_product=max(abs(value) for value in products),
                product_count=len(products),
                a_subnormal_count=sum(a_subnormal), b_subnormal_count=sum(b_subnormal),
                a_subnormal_contribution=math.fsum(value for i, value in enumerate(products) if a_subnormal[i]),
                b_subnormal_contribution=math.fsum(value for i, value in enumerate(products) if b_subnormal[i]),
                union_subnormal_contribution=accurate - ftz_exact,
                ftz_exact_sum=ftz_exact,
                ftz_exact_bf16=float(torch.tensor(ftz_exact, dtype=torch.float64).bfloat16()),
                exact_partials_fp32_promotion=accumulated,
                exact_partials_fp32_promotion_bf16=float(torch.tensor(accumulated).bfloat16()),
                ftz_partials_fp32_promotion=ftz_accumulated,
                ftz_partials_fp32_promotion_bf16=float(torch.tensor(ftz_accumulated).bfloat16()),
                group32_stages=group_records,
                raw_a_payload=q_values, raw_b_payload=b_values,
                raw_a_scale_bytes=a_exponents, raw_b_scale_bytes=b_exponents)


def _make_mma_control(linear, weight, scales):
    """Existing-library arithmetic control with its OWN legal MMA packing.

    WGMMA-packed buffers cannot be reinterpreted as MMA-packed buffers. This
    isolated diagnostic owns a temporary second layout, not a production cache.
    """
    from humming.forward import humming_forward
    from humming.transform import transform_humming_tensors
    from rtp_llm.models_py.modules.dsv41.sm90_dense import expand_block32_scales

    original = linear._sm90_linear
    config = dataclasses.replace(original.config, mma_type="mma", use_packed_k_layout=False)
    tensors = transform_humming_tensors(config, {
        "weight": weight.contiguous().clone().view(torch.int32),
        "weight_scale": expand_block32_scales(scales, *weight.shape),
    })
    adapter = SimpleNamespace(config=config, N=original.N, K=original.K,
                              weight=tensors["weight"], weight_scale=tensors["weight_scale"],
                              weight_scale_2=tensors.get("weight_scale_2"),
                              locks=torch.zeros_like(original.locks),
                              compute_config=original.compute_config, _forward=humming_forward)
    control = SimpleNamespace(_sm90_linear=adapter, _quantize_input=linear._quantize_input)
    tuning = dict(block_shape=(32, 128, 64), warp_shape=(32, 32, 64),
                  num_stages=3, num_ctas_per_sm=2, use_stream_k=False,
                  use_tma=False, use_warp_spec=False, use_mbarrier=False,
                  num_sms=torch.cuda.get_device_properties(weight.device).multi_processor_count)
    return _Candidate(control, tuning)


def _isolate_groups(linear, mma, q, packed, anchor):
    """One active original K32 group per row, and a32-basis control if needed."""
    row, column = anchor["row"], anchor["column"]
    groups, k = q.shape[1] // 32, q.shape[1]
    payload = torch.zeros((groups, k), device=q.device, dtype=torch.float32)
    for group in range(groups):
        payload[group, group * 32:(group + 1) * 32] = q[row, group * 32:(group + 1) * 32].float()
    payload = payload.to(q.dtype)
    group_scales = packed[row:row + 1].expand(groups, -1).contiguous()
    output = torch.empty((groups, linear.N), device=q.device, dtype=torch.bfloat16)
    wgmma_values = linear.forward_quantized(payload, group_scales)[:, column].float().cpu().tolist()
    mma.quantized(payload, group_scales, output)
    mma_values = output[:, column].float().cpu().tolist()
    expected = [float(torch.tensor(stage["exact_raw_partial"] * stage["scale"], dtype=torch.float64).bfloat16())
                for stage in anchor["group32_stages"]]
    differences = [index for index, (value, reference) in enumerate(zip(wgmma_values, expected)) if value != reference]
    mma_differences = [index for index, (value, reference) in enumerate(zip(mma_values, expected)) if value != reference]
    record = dict(row=row, column=column, groups=groups,
                  original_operand_bytes_and_scales_preserved_in_active_group=True,
                  wgmma_bf16_partials=wgmma_values, mma_bf16_partials=mma_values,
                  cpu_fsum_bf16_partials=expected,
                  wgmma_different_groups=differences, mma_different_groups=mma_differences,
                  limitation="BF16 final store can hide sub-ULP raw partial errors; equality alone does not prove exact raw accumulation")
    if differences:
        group = differences[0]
        # Each row selects one coefficient from the FIRST differing K32. No
        # reduction of multiple products: this separates decoding/layout from
        # intra-instruction accumulation for that exact operand block.
        basis = torch.zeros((32, k), dtype=torch.float32, device=q.device)
        indices = torch.arange(32, device=q.device)
        basis[indices, group * 32 + indices] = q[row, group * 32:(group + 1) * 32].float()
        basis = basis.to(q.dtype)
        basis_scales = packed[row:row + 1].expand(32, -1).contiguous()
        wg_basis = linear.forward_quantized(basis, basis_scales)[:, column].float().cpu().tolist()
        basis_output = torch.empty((32, linear.N), device=q.device, dtype=torch.bfloat16)
        mma.quantized(basis, basis_scales, basis_output)
        mma_basis = basis_output[:, column].float().cpu().tolist()
        scale = anchor["group32_stages"][group]["scale"]
        exact_products = [anchor["raw_a_payload"][group * 32 + index]
                          * anchor["raw_b_payload"][group * 32 + index] * scale for index in range(32)]
        exact_basis = [float(torch.tensor(value, dtype=torch.float64).bfloat16()) for value in exact_products]
        record["first_differing_group_basis"] = dict(
            group=group, wgmma=wg_basis, mma=mma_basis, cpu_exact_bf16=exact_basis,
            wgmma_coefficients_equal=wg_basis == exact_basis, mma_coefficients_equal=mma_basis == exact_basis,
            exact_product_sum=math.fsum(exact_products),
            a_payload=anchor["raw_a_payload"][group * 32:(group + 1) * 32],
            b_payload=anchor["raw_b_payload"][group * 32:(group + 1) * 32],
            a_scale_byte=anchor["raw_a_scale_bytes"][group], b_scale_byte=anchor["raw_b_scale_bytes"][group])
    return record


def _diagnose_actual(shape_name):
    """Preserve the failed original test; this target only produces evidence."""
    rows = 4096
    n, k = SHAPES[shape_name]
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    assert directory
    case = Path(directory) / f"dense_diagnostic_{shape_name}_m{rows}"
    case.mkdir(parents=True, exist_ok=False)
    record = dict(M=rows, N=n, K=k, shape=shape_name, seed=5141,
                  decision="DIAGNOSTIC_ONLY; original failed test is unchanged",
                  tolerance=dict(rtol=.015, atol=.02), benchmark="NOT_RUN")
    try:
        # EXACT RNG order from _run: weight -> scales -> layer preparation -> x.
        torch.manual_seed(5141)
        weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.pow(2.0, torch.randint(-3, 3, ((n + 31) // 32, k // 32), device="cuda").float()).to(torch.float8_e8m0fnu)
        raw_weight, raw_scales = weight.clone(), scales.clone()
        linear = V41MXFP8Linear(weight, scales)
        adapter = linear._sm90_linear
        x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
        q, packed = linear._quantize_input(x)
        hashes = dict(x=_hash(x), q=_hash(q), packed=_hash(packed), weight=_hash(raw_weight), scale=_hash(raw_scales))
        record["hashes"] = hashes
        previous_zip = os.environ.get("DSV41_DENSE_DIAG_BASELINE_ZIP")
        if previous_zip:
            suffix = f"dense_{shape_name}_m{rows}_n{n}_k{k}/result.json"
            with zipfile.ZipFile(previous_zip) as archive:
                matches = [name for name in archive.namelist() if name.endswith(suffix)]
                assert len(matches) == 1, (suffix, matches)
                previous = json.loads(archive.read(matches[0]))
            assert previous["hashes"] == hashes, "Diagnostic inputs differ from the original failed case"
            record["previous_input_hash_check"] = dict(path=previous_zip, result="BITWISE_HASH_MATCH")
        else:
            record["previous_input_hash_check"] = "NOT_CHECKED; compare hashes against original artifact"
        oracle = _oracle(q, packed, raw_weight, raw_scales)
        baseline = linear.forward_quantized(q, packed)
        violations = (~torch.isfinite(baseline.float()) | ~torch.isfinite(oracle.float())
                      | ((baseline.float() - oracle.float()).abs() > .02 + .015 * oracle.float().abs()))
        coords = torch.nonzero(violations, as_tuple=False).cpu().tolist()
        record["baseline_vs_original_oracle"] = _metrics(baseline, oracle)
        record["all_violating_coordinates"] = coords
        interval, config = _selected_default(adapter, rows)
        record["baseline_interval"], record["baseline_tuning"] = interval, config
        (case / "result.json").write_text(json.dumps(record, indent=2))
        # Bound unexpected regressions without silently discarding coordinates.
        assert len(coords) <= 1024, "All coordinates persisted; scalar fsum diagnostic is bounded to 1024"
        anchors = []
        for row, column in coords:
            anchor = _cpu_exact_coordinate(q, packed, raw_weight, raw_scales, row, column)
            old_ref = float(oracle[row, column])
            old_output = float(baseline[row, column])
            exact_ref = anchor["cpu_fsum_rounded_bf16"]
            anchor.update(row=row, column=column, original_fp32_oracle_bf16=old_ref,
                          baseline_bf16=old_output, original_oracle_minus_fsum_bf16=old_ref - exact_ref,
                          baseline_minus_fsum_bf16=old_output - exact_ref,
                          baseline_passes_fsum_original_tolerance=abs(old_output - exact_ref) <= .02 + .015 * abs(exact_ref),
                          original_fp32_oracle_matches_fsum_bf16=old_ref == exact_ref,
                          baseline_equals_ftz_prediction=old_output == anchor["ftz_partials_fp32_promotion_bf16"],
                          baseline_equals_exact_group32_promotion=old_output == anchor["exact_partials_fp32_promotion_bf16"])
            anchors.append(anchor)
            short_anchor = {key: value for key, value in anchor.items()
                            if key not in ("group32_stages", "raw_a_payload", "raw_b_payload",
                                           "raw_a_scale_bytes", "raw_b_scale_bytes")}
            print("DENSE_EXACT_ANCHOR " + json.dumps(dict(shape=shape_name, **short_anchor)), flush=True)
        record["exact_anchors"] = anchors
        candidate_records = []
        candidates, excluded = _candidates(adapter, rows, config)
        record["excluded_candidates"] = excluded
        out = torch.empty_like(baseline)
        for name, tuning in candidates:
            result = _Candidate(linear, tuning).quantized(q, packed, out)
            observation = dict(name=name, tuning=tuning, versus_baseline=_metrics(result, baseline),
                               versus_original_oracle=_metrics(result, oracle),
                               values_at_original_violations=[float(result[row, col]) for row, col in coords])
            candidate_records.append(observation)
            print("DENSE_DIAGNOSTIC_CANDIDATE " + json.dumps(dict(shape=shape_name, **observation)), flush=True)
        record["candidate_diagnostics"] = candidate_records
        if os.environ.get("DSV41_DENSE_MMA_DIAGNOSTICS", "0") == "1":
            mma = _make_mma_control(linear, raw_weight, raw_scales)
            mma_output = torch.empty_like(baseline)
            mma.quantized(q, packed, mma_output)
            mma_saved = mma_output.clone()
            repeatable = True
            for _ in range(3):
                mma.quantized(q, packed, mma_output)
                repeatable &= torch.equal(mma_output.view(torch.int16), mma_saved.view(torch.int16))
            record["mma_arithmetic_control"] = dict(
                layer_config=json.loads(mma.adapter.config.to_str()), tuning=json.loads(mma.tuning),
                original_activation_payload_and_scales=True,
                original_weight_values_repacked=True,
                versus_original_oracle=_metrics(mma_saved, oracle),
                versus_wgmma_baseline=_metrics(mma_saved, baseline), repeatability=repeatable,
                values_at_original_violations=[float(mma_saved[row, col]) for row, col in coords],
                scope="Arithmetic control only; new layout transient/test-only, no benchmark or production promotion")
            print("DENSE_MMA_CONTROL " + json.dumps(dict(shape=shape_name, **record["mma_arithmetic_control"])), flush=True)
            group_records = [_isolate_groups(linear, mma, q, packed, anchor) for anchor in anchors]
            record["single_k32_controls"] = group_records
            for group_record in group_records:
                summary = {key: value for key, value in group_record.items()
                           if key not in ("wgmma_bf16_partials", "mma_bf16_partials", "cpu_fsum_bf16_partials")}
                print("DENSE_K32_CONTROL " + json.dumps(dict(shape=shape_name, **summary)), flush=True)
        if os.environ.get("DSV41_DENSE_OPERAND_DIAGNOSTICS", "0") == "1":
            # B*8 normalizes every nonzero E4M3 subnormal in this bounded
            # fixture. Compensate original block exponents by -3, with exact
            # decoded-coefficient equality BEFORE calling the same WGMMA API.
            # Not generally legal for checkpoint blocks whose maxima are448.
            original_bytes = raw_scales.view(torch.uint8)
            assert bool(((original_bytes >= 4) & (original_bytes < 255)).all())
            assert float(raw_weight.float().abs().max()) <= 56.0
            normal_weight = (raw_weight.float() * 8).to(torch.float8_e4m3fn)
            normal_scales = (original_bytes.to(torch.int16) - 3).to(torch.uint8).view(torch.float8_e8m0fnu)
            original_decoded = raw_weight.float() * raw_scales.float().repeat_interleave(32, 0).repeat_interleave(32, 1)[:n, :k]
            normal_decoded = normal_weight.float() * normal_scales.float().repeat_interleave(32, 0).repeat_interleave(32, 1)[:n, :k]
            torch.testing.assert_close(normal_decoded.view(torch.int32), original_decoded.view(torch.int32), rtol=0, atol=0)
            assert not bool(((normal_weight.float().abs() > 0) & (normal_weight.float().abs() < 2**-6)).any())
            del original_decoded, normal_decoded
            normal_linear = V41MXFP8Linear(normal_weight, normal_scales)
            normal_result = normal_linear.forward_quantized(q, packed)
            record["lossless_b_x8_control"] = dict(
                decoded_coefficients_bitwise=True,
                original_weight_subnormals=int(((raw_weight.float().abs() > 0) & (raw_weight.float().abs() < 2**-6)).sum()),
                activation_payload_and_scale_unchanged=True,
                versus_original_oracle=_metrics(normal_result, oracle),
                versus_baseline=_metrics(normal_result, baseline),
                values_at_original_violations=[float(normal_result[row, col]) for row, col in coords],
                eligibility="Diagnostic only: requires Bmax<=56 and finite BS exponent>=4; real checkpoint not established")
            print("DENSE_NORMAL_OPERAND_CONTROL " + json.dumps(dict(shape=shape_name, **record["lossless_b_x8_control"])), flush=True)
        (case / "result.json").write_text(json.dumps(record, indent=2))
        assert coords, "Expected baseline failure did not reproduce; do not infer resolution"
        assert all(anchor["fp64_dot_minus_fsum"] == 0 for anchor in anchors), "FP64 anchors disagree; investigate before interpreting"
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


@pytest.mark.parametrize("shape_name", ["shared_w13", "shared_w2"])
def test_dense_actual_precision_diagnostic(shape_name):
    _diagnose_actual(shape_name)


def _profile_chrome_kernels(fn, flush, path, *, prepared):
    """Count ONLY Chrome cat=kernel events, never PyTorch CUDA annotations."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    if flush is not None:
        flush.zero_()
        torch.cuda.synchronize()
    begin = time.perf_counter()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profiler:
        fn()
        torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - begin) * 1000
    profiler.export_chrome_trace(str(path))
    trace = json.loads(path.read_text())
    counts, durations = Counter(), Counter()
    for event in trace["traceEvents"]:
        if event.get("cat") == "kernel" and event.get("ph") == "X":
            counts[event["name"]] += 1
            durations[event["name"]] += event["dur"]
    humming_count = sum(count for name, count in counts.items()
                        if re.search(r"(?:^|::|\s)humming(?:[<(]|$)", name))
    result = dict(source="Chrome JSON cat=kernel ph=X ONLY; no torch annotation events",
                  iterations=1, trace=str(path), profiler_context_wall_ms=wall_ms,
                  kernels=[dict(name=name, count=counts[name], us=durations[name]) for name in sorted(counts)],
                  kernel_sum_us_per_iter=sum(durations.values()),
                  pure_kernel_launch_count=sum(counts.values()), humming_launch_count=humming_count)
    assert humming_count == 1, result
    if prepared:
        assert sum(counts.values()) == 1, result
    return result


def _measure_chrome(fn, flush, path, *, prepared):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    end.record()
    end.synchronize()
    spans = []
    for _ in range(5):
        if flush is not None:
            flush.zero_()
        torch.cuda.synchronize()
        begin.record()
        fn()
        end.record()
        end.synchronize()
        spans.append(begin.elapsed_time(end))
    return dict(stream_span_ms=spans, median_ms=statistics.median(spans),
                profile=_profile_chrome_kernels(fn, flush, path, prepared=prepared))


def _mma_changed_input_graph(method, x, q, packed, out, weight, scales):
    saved_q, saved_packed = q.clone(), packed.clone()
    records = []
    for _ in range(3):
        method.quantized(q, packed, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        method.quantized(q, packed, out)
    try:
        graph_seed = 514100 + x.shape[0]
        torch.manual_seed(graph_seed)
        for multiplier in (16.0, .03125):
            changed_x = torch.randn_like(x) * multiplier
            changed_q, changed_packed = method.linear._quantize_input(changed_x)
            payload_changed = int((changed_q.view(torch.uint8) != q.view(torch.uint8)).sum())
            scale_words_changed = int((changed_packed != packed).sum())
            assert payload_changed and scale_words_changed
            q.copy_(changed_q)
            packed.copy_(changed_packed)
            graph.replay()
            expected = _oracle(changed_q, changed_packed, weight, scales)
            observed = out.clone()
            metrics = _metrics(observed, expected)
            torch.testing.assert_close(observed, expected, rtol=.015, atol=.02)
            method.quantized(q, packed, out)
            torch.testing.assert_close(out.view(torch.int16), observed.view(torch.int16), rtol=0, atol=0)
            for _ in range(3):
                graph.replay()
                torch.testing.assert_close(out.view(torch.int16), observed.view(torch.int16), rtol=0, atol=0)
            records.append(dict(seed=graph_seed, multiplier=multiplier, payload_bytes_changed=payload_changed,
                                scale_words_changed=scale_words_changed, oracle=metrics,
                                graph_vs_eager_bitwise=True, graph_repeatability=True))
    finally:
        q.copy_(saved_q)
        packed.copy_(saved_packed)
    return records


def _run_mma_acceptance(rows, shape_name):
    """New correctness gate; does NOT require equivalence to a known bad WG."""
    assert importlib.metadata.version("humming-kernels") == "0.1.16"
    n, k = SHAPES[shape_name]
    directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    assert directory
    root = Path(directory) / f"dense_mma_acceptance_{shape_name}_m{rows}"
    root.mkdir(parents=True, exist_ok=False)
    record = dict(M=rows, N=n, K=k, shape=shape_name, seed=5141,
                  tolerance=dict(rtol=.015, atol=.02), decision="NOT_RUN",
                  scope="Isolated dense component; original WGMMA failure tests retained; no production promotion",
                  arithmetic="original A E4M3g32/E8M0; original B E4M3 block32x32/E8M0; library MMA FP32 g32 promotion, K64 invariant/noStreamK",
                  memory="MMA weight layout shared by two candidates; additional WGMMA layout is diagnostic/test-only, not a production design",
                  timing="Eager full-chain and prepared GEMM separately; graph validates correctness only; Chrome cat=kernel authoritative")
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        # Preserve original failing fixture and RNG order exactly.
        torch.manual_seed(5141)
        weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.pow(2.0, torch.randint(-3, 3, ((n + 31) // 32, k // 32), device="cuda").float()).to(torch.float8_e8m0fnu)
        raw_weight, raw_scales = weight.clone(), scales.clone()
        linear = V41MXFP8Linear(weight, scales)
        x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
        q, packed = linear._quantize_input(x)
        unpacked = unpack_ue8m0_input_scales(packed)
        oracle = _oracle(q, packed, raw_weight, raw_scales)
        original_q, original_packed = q.view(torch.uint8).clone(), packed.clone()
        record["hashes"] = dict(x=_hash(x), q=_hash(q), packed=_hash(packed), weight=_hash(raw_weight), scale=_hash(raw_scales))
        baseline = _Candidate(linear)
        out = torch.empty((rows, n), dtype=torch.bfloat16, device="cuda")
        baseline_output = baseline.quantized(q, packed, out).clone()
        baseline_metrics = _metrics(baseline_output, oracle)
        record["wgmma_diagnostic"] = dict(oracle=baseline_metrics,
                                          benchmark_eligible=baseline_metrics["tolerance_violations"] == 0,
                                          meaning="Diagnostic comparison, NOT the MMA correctness reference")
        method = _make_mma_control(linear, raw_weight, raw_scales)
        config32 = json.loads(method.tuning)
        other = copy.deepcopy(config32)
        if rows <= 8:
            other.update(block_shape=(16, 128, 64), warp_shape=(16, 32, 64))
            other_name = "mma_m16n128_s3"
        else:
            other.update(block_shape=(64, 128, 64), warp_shape=(32, 32, 64))
            other_name = "mma_m64n128_s3"
        methods = [("mma_m32n128_s3", method), (other_name, _Candidate(method.linear, other))]
        before = [(value.data_ptr(), value._version) for value in (method.adapter.weight, method.adapter.weight_scale)]
        accepted, observations = [], []
        for name, candidate in methods:
            begin = time.perf_counter()
            result = candidate.quantized(q, packed, out).clone()
            torch.cuda.synchronize()
            warmup_ms = (time.perf_counter() - begin) * 1000
            metrics = _metrics(result, oracle)
            observation = dict(name=name, tuning=json.loads(candidate.tuning), oracle=metrics,
                               wgmma_difference_diagnostic=_metrics(result, baseline_output),
                               first_call_compile_wall_ms=warmup_ms, decision="NOT_RUN")
            observations.append(observation)
            record["candidates"] = observations
            (root / "result.json").write_text(json.dumps(record, indent=2))
            # Independent accuracy is mandatory; equality to incorrect WGMMA is
            # intentionally NOT an acceptance condition for this arithmetic fix.
            torch.testing.assert_close(result, oracle, rtol=.015, atol=.02)
            for _ in range(3):
                candidate.quantized(q, packed, out)
                torch.testing.assert_close(out.view(torch.int16), result.view(torch.int16), rtol=0, atol=0)
            candidate.full_chain(x, out)
            torch.testing.assert_close(out.view(torch.int16), result.view(torch.int16), rtol=0, atol=0)
            candidate.prepared(q, unpacked, out)
            torch.testing.assert_close(out.view(torch.int16), result.view(torch.int16), rtol=0, atol=0)
            observation["changed_input_graph"] = _mma_changed_input_graph(candidate, x, q, packed, out, raw_weight, raw_scales)
            observation.update(repeatability=True, full_chain_bitwise=True, prepared_bitwise=True, decision="BOUNDED_SUPPORT")
            accepted.append((name, candidate))
            print("DENSE_MMA_ACCEPTANCE " + json.dumps(dict(shape=shape_name, M=rows, **observation)), flush=True)
            (root / "result.json").write_text(json.dumps(record, indent=2))
        torch.testing.assert_close(q.view(torch.uint8), original_q, rtol=0, atol=0)
        torch.testing.assert_close(packed, original_packed, rtol=0, atol=0)
        assert before == [(value.data_ptr(), value._version) for value in (method.adapter.weight, method.adapter.weight_scale)]
        if baseline_metrics["tolerance_violations"] == 0:
            # Only valid baselines may be timed as speed comparators.
            accepted.insert(0, ("wgmma_valid_baseline", baseline))
        else:
            record["speedup_vs_wgmma"] = "FORBIDDEN: known-invalid baseline; absolute accepted MMA times only"
        if os.environ.get("DSV41_DENSE_BENCHMARK", "1") == "1":
            flush = torch.empty(256 * 1024**2, dtype=torch.uint8, device="cuda")
            measurements = {name: {} for name, _ in accepted}
            for repeat in range(3):
                ordered = accepted if repeat % 2 == 0 else list(reversed(accepted))
                for name, candidate in ordered:
                    for scope, fn in (("full_chain", lambda candidate=candidate: candidate.full_chain(x, out)),
                                      ("prepared_gemm", lambda candidate=candidate: candidate.prepared(q, unpacked, out))):
                        modes = measurements[name].setdefault(scope, {"hot": [], "cold": []})
                        for mode in ("hot", "cold"):
                            modes[mode].append(_measure_chrome(fn, flush if mode == "cold" else None,
                                root / f"{name}_{scope}_{mode}_r{repeat}.trace.json", prepared=scope == "prepared_gemm"))
            record["measurements"] = measurements
            record["l2_flush_bytes"] = flush.numel()
            for name, scopes in measurements.items():
                for scope, modes in scopes.items():
                    for mode, samples in modes.items():
                        print(f"DENSE_MMA_TIME {shape_name} M{rows} {name} {scope} {mode} "
                              f"span_ms={statistics.median(item['median_ms'] for item in samples):.6f} "
                              f"chrome_kernel_us={statistics.median(item['profile']['kernel_sum_us_per_iter'] for item in samples):.3f}", flush=True)
        record["decision"] = "BOUNDED_SUPPORT; component only"
    except Exception as exc:
        record["decision"] = "REJECT"
        record["failure"] = str(exc)
        raise
    finally:
        (root / "result.json").write_text(json.dumps(record, indent=2))
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


@pytest.mark.parametrize("rows", [1, 4096], ids=["m1", "m4096"])
@pytest.mark.parametrize("shape_name", ["shared_w13", "shared_w2"])
def test_dense_mma_acceptance(rows, shape_name):
    _run_mma_acceptance(rows, shape_name)
