"""Acceptance-gated SM90 g32 Dense comparison, not a global-best claim.

Root owns Bazel/GPU execution. Every arm sees the same original quantized
payload/scales and must independently pass before timing. Rejected arms remain
visible and cannot prevent measurement of other valid arms. WGMMA is diagnostic
only; BF16 is explicitly a non-FP8 comparison, not a production recommendation.
"""

import copy
import fnmatch
import hashlib
import json
import os
import re
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rtp_llm.models_py.modules.dsv41.sm90_dense import (
    Sm90MXFP8Linear,
    unpack_ue8m0_input_scales,
)
from rtp_llm.models_py.modules.dsv41.test.test_sm90_dense_deepgemm import (
    _ExactScaleBf16Candidate,
    _changed_input_graph,
    _check_bits,
    _hash,
    _measure,
)
from rtp_llm.models_py.modules.dsv41.test.test_sm90_dense_tuning import (
    SHAPES,
    _Candidate,
    _make_mma_control,
    _metrics,
    _oracle,
)
from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear


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


class _TritonCandidate:
    def __init__(self, weight, scales, config):
        from rtp_llm.models_py.modules.dsv41.sm90_dense_triton import Sm90TritonMXFP8Linear

        self.adapter = Sm90TritonMXFP8Linear(weight, scales, config=config)
        self.K = weight.shape[1]

    def quantize(self, x):
        return V41MXFP8Linear._quantize_input(self, x)

    def quantized(self, q, packed, out):
        return self.adapter.forward_quantized(q, packed, out=out)

    def prepared(self, q, scales, out):
        return self.adapter.prepared(q, scales, out)

    def full_chain(self, x, out):
        return self.quantized(*self.quantize(x), out)


class _SplitK32Candidate:
    """Feasibility control: native FP8 with zero-padded subgroups, not selected."""

    def __init__(self, weight, scales, splits):
        from rtp_llm.models_py.modules.dsv41.sm90_dense_splitk32 import Sm90SplitK32Linear

        self.adapter = Sm90SplitK32Linear(weight, scales, splits=splits)
        self.K = weight.shape[1]

    def quantize(self, x):
        return V41MXFP8Linear._quantize_input(self, x)

    def quantized(self, q, packed, out):
        return self.adapter.forward_quantized(q, packed, out=out)

    def prepared(self, q, scales, out):
        return self.adapter.prepared(q, scales, out)

    def full_chain(self, x, out):
        return self.quantized(*self.quantize(x), out)


class _SelectedAdapterCandidate:
    """Retained A16 comparison adapter; no production factory selects it."""

    def __init__(self, weight, scales):
        from rtp_llm.models_py.modules.dsv41.sm90_dense_selected import Sm90SelectedG32Linear

        self.adapter = Sm90SelectedG32Linear(weight, scales)
        self.K = weight.shape[1]
        self.last_rows = None

    def quantize(self, x):
        return V41MXFP8Linear._quantize_input(self, x)

    def quantized(self, q, packed, out):
        self.last_rows = q.numel() // self.K
        return self.adapter.forward_quantized(q, packed, out=out)

    def prepared(self, q, scales, out):
        self.last_rows = q.numel() // self.K
        return self.adapter.prepared(q, scales, out)

    def full_chain(self, x, out):
        self.last_rows = x.numel() // self.K
        return self.quantized(*self.quantize(x), out)


class _Wgmma16Candidate(_SplitK32Candidate):
    """Same quantized operands, explicitly BF16 WGMMA arithmetic."""

    def __init__(self, weight, scales):
        from rtp_llm.models_py.modules.dsv41.sm90_dense_wgmma16 import Sm90Wgmma16Linear

        self.adapter = Sm90Wgmma16Linear(weight, scales)
        self.K = weight.shape[1]


def _selected(name):
    patterns = os.environ.get("DSV41_DENSE_G32_COMPARE_ARMS", "*").split(",")
    return any(fnmatch.fnmatchcase(name, pattern.strip()) for pattern in patterns)


def _arm_builders(weight, scales, quantize, rows):
    """Lazy constructors: an import/repack failure cannot mask other arms."""
    from rtp_llm.models_py.modules.dsv41.sm90_dense_triton import TRITON_CONFIGS

    yield ("humming_wgmma16", "W8A8_G32",
           dict(storage="Original FP8/g32", arithmetic="BF16 WGMMA, NOT native FP8",
                production_selected=False), 2, lambda: _Wgmma16Candidate(weight, scales))

    for splits in (2, 4):
        if weight.shape[1] * splits <= 16384:
            yield (f"native_splitk32_{splits}", "W8A8_G32",
                   dict(logical_group_k=32, active_k_per_instruction=32 // splits,
                        expanded_weight_factor=splits, production_selected=False),
                   3, lambda splits=splits: _SplitK32Candidate(weight, scales, splits))

    for name, config in TRITON_CONFIGS.items():
        splits = int(config.get("SPLIT_K", config.get("split_k", 1)))
        yield (f"triton_{name}", "W8A8_G32", dict(config=config, split_k=splits),
               2 if splits > 1 else 1,
               lambda name=name: _TritonCandidate(weight, scales, name))

    # Unsupported fixtures never construct or invoke this bounded adapter.
    if tuple(weight.shape) in ((4608, 5120), (5120, 2304)):
        yield ("selected_adapter", "W8A8_G32",
               dict(adapter="Sm90SelectedG32Linear", production_selected=False,
                    setting_scope="Direct comparison adapter; no production dispatch"),
               2 if rows >= 512 else 1, lambda: _SelectedAdapterCandidate(weight, scales))

    # Each candidate owns a legal MMA-specific repack. Never reinterpret the
    # WGMMA weight layout as MMA. The two tuning arms differ only in M tile.
    for tile in ((16, 32) if rows <= 16 else (32, 64)):
        def build_mma(tile=tile):
            baseline = Sm90MXFP8Linear(weight, scales)
            linear = SimpleNamespace(_sm90_linear=baseline, _quantize_input=quantize)
            original = _make_mma_control(linear, weight, scales)
            tuning = copy.deepcopy(json.loads(original.tuning))
            tuning.update(block_shape=(tile, 128, 64), warp_shape=(min(tile, 32), 32, 64))
            method = _Candidate(original.linear, tuning)
            method.quantize = quantize
            return method

        yield f"humming_mma_m{tile}", "W8A8_G32", dict(mma_type="mma", tile_m=tile), 1, build_mma
    yield ("exactscale_bf16", "BF16_DIAGNOSTIC", {}, 1,
           lambda: _ExactScaleBf16Candidate(weight, scales))


def _save_arithmetic_evidence(method, name, case):
    """Operand FP8 does not establish Tensor Core arithmetic dtype."""
    selected_wgmma16 = (isinstance(method, _SelectedAdapterCandidate)
                        and method.adapter.get_dispatch(method.last_rows)["backend"] == "humming_wgmma16")
    if isinstance(method, _Wgmma16Candidate) or selected_wgmma16:
        from humming.kernel.humming import HummingKernel

        adapter = method.adapter._wgmma16 if selected_wgmma16 else method.adapter
        layer, compute, tuning = HummingKernel._resolve_configs(
            adapter.config.to_str(), adapter.compute_config,
            adapter.tuning_config, adapter.weight.device.index)
        configs = [(0, 1 << 30, tuning)] if isinstance(tuning, dict) else tuning
        binaries = []
        for index, (start, end, config) in enumerate(configs):
            merged = layer | compute | config
            merged.pop("num_sms", None)
            kernel = HummingKernel(**merged)
            binary = Path(kernel.kernel_filename).read_bytes()
            destination = case / f"{name}_{index}.cubin"
            destination.write_bytes(binary)
            binaries.append(dict(m_interval=[start, end], cubin=str(destination),
                                 sha256=hashlib.sha256(binary).hexdigest()))
        return dict(arithmetic_dtype=adapter.arithmetic_contract,
                    layer_config=adapter.config.to_str(), kernels=binaries,
                    evidence_status="Current configuration cubins saved; inspect SASS")
    if isinstance(method, _SplitK32Candidate):
        binary = Path(method.adapter.extension_binary).read_bytes()
        destination = case / f"{name}.so"
        destination.write_bytes(binary)
        return dict(arithmetic_dtype="Native FP8 WGMMA; expanded physical K",
                    binary=str(destination), sha256=hashlib.sha256(binary).hexdigest(),
                    expanded_weight_factor=method.adapter.splits,
                    evidence_status="Current DG extension saved; inspect SASS")
    selected_dispatch = None
    selected_humming = False
    if isinstance(method, _SelectedAdapterCandidate):
        selected_dispatch = method.adapter.get_dispatch(method.last_rows)
        selected_humming = getattr(method.adapter, "last_compiled_kernel", None) is None
    if isinstance(method, _Candidate) or selected_humming:
        return dict(arithmetic_dtype="FP8 input / FP16 HMMA tensor-core arithmetic",
                    selected_dispatch=selected_dispatch,
                    evidence_status="Round6 cubin inspected; current cubin verification NOT_RUN",
                    prior_cubin="round5_humming_cache_installed/bd44e1603cd54035/kernel.cubin",
                    prior_cubin_sha256="689c06297981fab9820aa9e0b9dd01c69dce98cc1ce682cdb6f3d68c60a72b04",
                    prior_sass=["F2FP.F16.E4M3.UNPACK_B", "HMMA.16816.F32"],
                    meaning="Even E4M3 MMA PTX can lower to FP16 SASS; do not claim native FP8 from PTX alone")
    if not isinstance(method, (_TritonCandidate, _SelectedAdapterCandidate)):
        return dict(decision="NOT_RUN", arithmetic_dtype="UNVERIFIED",
                    meaning="Do not infer native FP8 instructions from E4M3 storage or backend name")
    evidence = []
    for attribute in ("last_compiled_kernel", "last_reduce_kernel"):
        compiled = getattr(method.adapter, attribute, None)
        if compiled is None:
            continue
        binary = compiled.asm.get("cubin")
        binary_evidence = dict(cubin_status="NOT_AVAILABLE")
        if isinstance(binary, (bytes, bytearray, memoryview)):
            binary = bytes(binary)
            binary_path = case / f"{name}_{attribute}.cubin"
            binary_path.write_bytes(binary)
            binary_evidence = dict(cubin_status="SAVED; SASS inspection pending",
                                   cubin=str(binary_path), cubin_bytes=len(binary),
                                   cubin_sha256=hashlib.sha256(binary).hexdigest())
        ptx = compiled.asm.get("ptx", "")
        if not ptx:
            evidence.append(dict(kernel=attribute, decision="BLOCKED", reason="PTX unavailable",
                                 **binary_evidence))
            continue
        path = case / f"{name}_{attribute}.ptx"
        path.write_text(ptx)
        instructions = [line.strip() for line in ptx.splitlines()
                        if re.search(r"\b(?:wgmma\.mma_async|mma\.sync)\.", line)]
        counts = dict(
            wgmma_e4m3=sum("wgmma.mma_async" in line and ".e4m3" in line for line in instructions),
            mma_sync_e4m3=sum("mma.sync" in line and ".e4m3" in line for line in instructions),
            mma_sync_f16=sum("mma.sync" in line and ".f16" in line for line in instructions),
            mma_sync_bf16=sum("mma.sync" in line and ".bf16" in line for line in instructions),
        )
        evidence.append(dict(kernel=attribute, decision="SOURCE_FACT", ptx=str(path),
                             instruction_counts=counts, instruction_examples=list(dict.fromkeys(instructions))[:8],
                             **binary_evidence))
    totals = {key: sum(item.get("instruction_counts", {}).get(key, 0) for item in evidence)
              for key in ("wgmma_e4m3", "mma_sync_e4m3", "mma_sync_f16", "mma_sync_bf16")}
    arithmetic = "UNVERIFIED"
    if totals["mma_sync_f16"] and not (totals["wgmma_e4m3"] or totals["mma_sync_e4m3"]):
        arithmetic = "FP16_MMA_FROM_FP8_OPERANDS; NOT_NATIVE_FP8"
    elif totals["mma_sync_e4m3"] and not totals["mma_sync_f16"]:
        arithmetic = "E4M3_MMA_SYNC_PTX; MACHINE_ARITHMETIC_UNVERIFIED"
    elif totals["wgmma_e4m3"] and not totals["mma_sync_f16"]:
        arithmetic = "E4M3_WGMMA_PTX; MACHINE_ARITHMETIC_UNVERIFIED"
    return dict(arithmetic_dtype=arithmetic, total_instruction_counts=totals, kernels=evidence,
                selected_dispatch=selected_dispatch,
                meaning="Generated PTX source evidence, not inferred from timing or FP8 storage")


def _gate(method, kind, x, q, packed, unpacked, out, oracle, weight, scales, observation):
    """Persist all stages; graph helper retains a failing attempt's evidence."""
    start = time.perf_counter()
    assert method.quantized(q, packed, out) is out
    torch.cuda.synchronize()
    observation["compile_first_call_wall_ms"] = (time.perf_counter() - start) * 1000
    result = out.clone()
    observation["oracle"] = _metrics(result, oracle)
    torch.testing.assert_close(result.reshape_as(oracle), oracle, rtol=.015, atol=.02)
    for _ in range(3):
        out.fill_(float("nan"))
        assert method.quantized(q, packed, out) is out
        _check_bits(out, result)
    method.full_chain(x, out)
    _check_bits(out, result)
    if kind == "BF16_DIAGNOSTIC":
        decoded = method.decode_activation(q, packed)
        prepared_fn = lambda: method.prepared(decoded, out)
        observation.update(static_decoded_weight_bytes=method.weight.numel() * method.weight.element_size(),
                           prepared_input="Already decoded BF16 A/B, not FP8 inputs",
                           production_gate="NOT_ELIGIBLE: Torch full-chain diagnostic")
    else:
        prepared_fn = lambda: method.prepared(q, unpacked, out)
        observation["prepared_input"] = "Original E4M3 payload and predecoded FP32 group32 scales"
    prepared_fn()
    _check_bits(out, result)
    observation["changed_input_graph"] = []
    _changed_input_graph(method, x, out, weight, scales, records=observation["changed_input_graph"])
    observation.update(repeatability=True, full_chain_bitwise=True, prepared_bitwise=True,
                       output_sha256=_hash(result), decision="PASS")
    return prepared_fn


def _benchmark(accepted, x, out, case, record):
    flush = torch.empty(256 * 1024**2, device="cuda", dtype=torch.uint8)
    record["l2_flush_bytes"] = flush.numel()
    for arm in accepted:
        arm.observation["measurements"] = {}
    for repeat in range(3):
        ordered = accepted if repeat % 2 == 0 else list(reversed(accepted))
        for arm in ordered:
            if arm.observation.get("timing_decision") == "BLOCKED":
                continue
            try:
                for scope, fn in (("full_chain", lambda arm=arm: arm.method.full_chain(x, out)),
                                  ("prepared_gemm", arm.prepared_fn)):
                    modes = arm.observation["measurements"].setdefault(scope, {"hot": [], "cold": []})
                    for mode in ("hot", "cold"):
                        # Reuse the existing event/profiler harness. Split-K is
                        # intentionally two kernels, so its count is checked
                        # explicitly below, not rejected by prepared=True.
                        sample = _measure(fn, flush if mode == "cold" else None,
                            case / f"{arm.name}_{scope}_{mode}_r{repeat}.trace.json", prepared=False)
                        modes[mode].append(sample)
                        if scope == "prepared_gemm":
                            actual = sample["profile"]["pure_kernel_launch_count"]
                            assert actual == arm.expected_launches, dict(
                                arm=arm.name, expected=arm.expected_launches, actual=actual,
                                kernels=sample["profile"]["kernels"])
                arm.observation["timing_decision"] = "MEASURED"
            except Exception as exc:
                # A profile/kernel-count problem is not a numerical failure.
                arm.observation.update(timing_decision="BLOCKED", timing_failure=str(exc))
            (case / "result.json").write_text(json.dumps(record, indent=2))
    summaries = {}
    for arm in accepted:
        if arm.observation.get("timing_decision") != "MEASURED":
            continue
        summary = {}
        for scope, modes in arm.observation["measurements"].items():
            summary[scope] = {}
            for mode, samples in modes.items():
                assert len(samples) == 3
                summary[scope][mode] = dict(
                    pure_gpu_kernel_us=statistics.median(s["profile"]["kernel_sum_us_per_iter"] for s in samples),
                    cuda_event_span_ms=statistics.median(s["median_ms"] for s in samples),
                    repetitions=3)
        arm.observation["timing_summary"] = summary
        summaries[arm.name] = dict(kind=arm.kind, **summary)
    record["timing_summary"] = summaries
    record["selection_metric"] = "Primary: Chrome pure GPU kernel-sum full_chain; CUDA event stream span secondary; warm/cold separate"
    record["bounded_quantized_g32_winners"] = {}
    for mode in ("hot", "cold"):
        ranked = sorted(((name, summary["full_chain"][mode]["pure_gpu_kernel_us"])
                         for name, summary in summaries.items() if summary["kind"] == "W8A8_G32"), key=lambda item: item[1])
        record["bounded_quantized_g32_winners"][mode] = dict(
            fastest_tested=ranked[0][0] if ranked else None,
            ranking_pure_gpu_us=[dict(name=name, us=value) for name, value in ranked],
            meaning="Original W8A8 g32 operands; inspect arithmetic_evidence before calling arithmetic native FP8; exact synthetic shape only")


def _run(rows, shape_name, n, k, *, leading_shape=None):
    directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    assert directory, "Bazel undeclared outputs required for reproducible evidence"
    case = Path(directory) / f"dense_g32_compare_{shape_name}_m{rows}"
    case.mkdir(parents=True, exist_ok=False)
    record = dict(M=rows, N=n, K=k, seed=5141, shape_name=shape_name,
                  tolerance=dict(rtol=.015, atol=.02), decision="NOT_RUN", arms=[],
                  scope="Original synthetic g32 payload, isolated single projection; not routed MoE/model performance",
                  timing_scope="full_chain includes BF16 quantization + packed-scale conversion + GEMM; prepared_gemm excludes preparation",
                  gates_3_4_5="NOT_RUN: native-forward/model64K128K/phaseMFU not established by component benchmark")
    accepted = []
    try:
        torch.manual_seed(5141)
        weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.pow(2.0, torch.randint(-3, 3, ((n + 31) // 32, k // 32),
                                             device="cuda").float()).to(torch.float8_e8m0fnu)
        x = torch.randn(*(leading_shape or (rows,)), k, device="cuda", dtype=torch.bfloat16)
        quantizer = SimpleNamespace(K=k)
        quantize = lambda value: V41MXFP8Linear._quantize_input(quantizer, value)
        q, packed = quantize(x)
        unpacked = unpack_ue8m0_input_scales(packed)
        oracle = _oracle(q, packed, weight, scales)
        record["hashes"] = dict(x=_hash(x), q=_hash(q), packed=_hash(packed), unpacked=_hash(unpacked),
                                weight=_hash(weight), scales=_hash(scales))
        out = torch.empty((*x.shape[:-1], n), device="cuda", dtype=torch.bfloat16)
        # WGMMA is not an acceptance reference or a speed comparator here.
        try:
            wg = Sm90MXFP8Linear(weight, scales)
            wg.forward_quantized(q, packed, out=out)
            record["wgmma_diagnostic"] = dict(oracle=_metrics(out, oracle), benchmark_eligible=False)
            del wg
        except Exception as exc:
            record["wgmma_diagnostic"] = dict(decision="BLOCKED", failure=str(exc), benchmark_eligible=False)
        after_wg = dict(x=_hash(x), q=_hash(q), packed=_hash(packed), unpacked=_hash(unpacked),
                        weight=_hash(weight), scales=_hash(scales))
        record["wgmma_diagnostic"]["shared_input_unchanged"] = after_wg == record["hashes"]
        assert after_wg == record["hashes"], "Diagnostic WGMMA mutated shared inputs; comparison aborted"
        for name, kind, config, launches, build in _arm_builders(weight, scales, quantize, rows):
            observation = dict(name=name, kind=kind, config=config, decision="NOT_RUN",
                               expected_prepared_kernel_count=launches)
            record["arms"].append(observation)
            if not _selected(name):
                observation.update(decision="NOT_RUN", reason="Excluded by explicit arm selector")
                continue
            method = None
            try:
                begin = time.perf_counter()
                method = build()
                torch.cuda.synchronize()
                observation["load_repack_wall_ms"] = (time.perf_counter() - begin) * 1000
                if isinstance(method, _SelectedAdapterCandidate):
                    observation["adapter_dispatch"] = method.adapter.get_dispatch(rows)
                    expected_backend = "humming_wgmma16" if rows >= 512 else "humming_mma"
                    assert observation["adapter_dispatch"]["backend"] == expected_backend
                    if rows < 512:
                        expected_tile = 16 if rows <= 16 else (64 if n == 4608 else 32)
                        assert observation["adapter_dispatch"]["tile_m"] == expected_tile
                    observation["selected_adapter_class"] = type(method.adapter).__name__
                prepared_fn = _gate(method, kind, x, q, packed, unpacked, out, oracle, weight, scales, observation)
                accepted.append(SimpleNamespace(name=name, kind=kind, method=method,
                    observation=observation, prepared_fn=prepared_fn, expected_launches=launches))
            except AssertionError as exc:
                observation.update(decision="REJECT", failure=str(exc), benchmark_eligible=False)
            except Exception as exc:
                observation.update(decision="BLOCKED", failure=str(exc), benchmark_eligible=False)
            else:
                observation["benchmark_eligible"] = True
            finally:
                # Persist binaries even for numerically rejected arms. Failure
                # to export diagnostic artifacts must not masquerade as a
                # failed numerical gate or silently imply verified arithmetic.
                if method is not None:
                    try:
                        observation["arithmetic_evidence"] = _save_arithmetic_evidence(method, name, case)
                    except Exception as exc:
                        observation["arithmetic_evidence"] = dict(decision="BLOCKED", failure=str(exc))
                # Failed arms must not contaminate subsequent arms. This check
                # deliberately sits outside the per-arm exception handlers:
                # mutation aborts the CASE before any timing/ranking occurs.
                after = dict(x=_hash(x), q=_hash(q), packed=_hash(packed), unpacked=_hash(unpacked),
                             weight=_hash(weight), scales=_hash(scales))
                observation["shared_input_unchanged"] = after == record["hashes"]
                if after != record["hashes"]:
                    observation.update(decision="REJECT", benchmark_eligible=False,
                                       input_mutation_hashes=after)
                    raise AssertionError(f"{name} mutated shared inputs; comparison aborted")
            (case / "result.json").write_text(json.dumps(record, indent=2))
            print("DENSE_G32_COMPARE_ARM " + json.dumps(observation), flush=True)
        if os.environ.get("DSV41_DENSE_G32_COMPARE_BENCHMARK", "0") == "1" and accepted:
            _benchmark(accepted, x, out, case, record)
        else:
            record["timing_decision"] = "NOT_RUN"
        fp8_accepted = [arm.name for arm in accepted if arm.kind == "W8A8_G32"]
        record["accepted_quantized_g32_arms"] = fp8_accepted
        record["decision"] = "BOUNDED_SUPPORT; per-arm failures retained" if fp8_accepted else "REJECT"
        assert fp8_accepted, "No FP8 arm passed the unchanged independent gate"
    except Exception as exc:
        record.update(decision="REJECT", failure=str(exc))
        raise
    finally:
        (case / "result.json").write_text(json.dumps(record, indent=2))
        print("DENSE_G32_COMPARE_RESULT " + json.dumps(record), flush=True)


@pytest.mark.parametrize("rows,n,k,leading_shape", [
    (1, 256, 128, None), (7, 256, 512, None),
    (15, 320, 640, (3, 5)), (129, 256, 512, None),
])
def test_dense_g32_compare_fixture(rows, n, k, leading_shape):
    _run(rows, f"fixture_n{n}_k{k}", n, k, leading_shape=leading_shape)


_ACTUAL_ROWS = [int(value) for value in os.environ.get("DSV41_DENSE_G32_COMPARE_ROWS", "1,16,512,4096").split(",")]
_ACTUAL_SHAPES = os.environ.get("DSV41_DENSE_G32_COMPARE_SHAPES", "shared_w13,shared_w2").split(",")


@pytest.mark.parametrize("rows", _ACTUAL_ROWS, ids=[f"m{value}" for value in _ACTUAL_ROWS])
@pytest.mark.parametrize("shape_name", _ACTUAL_SHAPES)
def test_dense_g32_compare_actual(rows, shape_name):
    _run(rows, shape_name, *SHAPES[shape_name])


@pytest.mark.skipif(os.environ.get("DSV41_DENSE_G32_COMPARE_LARGE_M", "0") != "1",
                    reason="Explicit large-M shape probe; not inferred CP8 model configuration")
@pytest.mark.parametrize("shape_name", ["shared_w13", "shared_w2"])
def test_dense_g32_compare_large_m(shape_name):
    _run(16384, shape_name, *SHAPES[shape_name])
