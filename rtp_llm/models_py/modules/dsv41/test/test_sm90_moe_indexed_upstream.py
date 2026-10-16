"""Round5 indexed W4A16, isolated pinned upstream; never production dispatch.

Baseline means RTP's grouped algorithm on the SAME upstream source, not an
in-process old-wheel comparison (two Humming torch-op registrations are unsafe).
M is already-routed rows; top_k=1 avoids conflating it with original requests.
"""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter


_REVISION = "d113632a2bbcee5f25fcc8a3a7d92ccdfbac62a8"
_BASELINE_ONLY = os.environ.get("DSV41_INDEXED_BASELINE_ONLY", "0") == "1"
_USE_INSTALLED = _BASELINE_ONLY or os.environ.get("DSV41_INDEXED_USE_INSTALLED", "0") == "1"
_ALIGNMENT = os.environ.get("DSV41_INDEXED_ALIGNMENT", "torch")
if _ALIGNMENT not in ("torch", "compiled"):
    raise ValueError("DSV41_INDEXED_ALIGNMENT must be torch or compiled")


def _select_source():
    if any(name == "humming" or name.startswith("humming.") for name in sys.modules):
        raise RuntimeError("Humming already imported; use an isolated test process")
    source = os.environ.get("DSV41_UPSTREAM_HUMMING", "")
    if _USE_INSTALLED:
        if source:
            raise RuntimeError("Installed mode requires DSV41_UPSTREAM_HUMMING unset")
        spec = importlib.util.find_spec("humming")
        if spec is None:
            raise RuntimeError("Installed Humming baseline is unavailable")
        installed = Path(importlib.metadata.distribution("humming-kernels").locate_file("humming/__init__.py")).resolve()
        if Path(spec.origin).resolve() != installed:
            raise RuntimeError(f"Installed baseline shadowed: {spec.origin}; expected {installed}")
        return Path(spec.origin).resolve().parent.parent
    if not source:
        raise RuntimeError("Set DSV41_UPSTREAM_HUMMING, or explicitly select an installed mode")
    root = Path(source).resolve(strict=True)
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
    ).strip()
    if revision != _REVISION:
        raise RuntimeError(f"Expected {_REVISION}, got {revision}")
    if subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--", "humming"], text=True):
        raise RuntimeError("Pinned Humming source has tracked modifications")
    sys.path.insert(0, str(root))
    spec = importlib.util.find_spec("humming")
    if spec is None or Path(spec.origin).resolve() != root / "humming" / "__init__.py":
        raise RuntimeError(f"Wrong Humming import origin: {spec}")
    return root


_SOURCE_ROOT = _select_source()  # Before torch helpers can import Humming.
_OUTPUT_ROOT = Path(os.environ.get("DSV41_INDEXED_RESULT_DIR") or
                    os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR") or
                    tempfile.mkdtemp(prefix="dsv41_indexed_"))

import pytest
import torch

from rtp_llm.models_py.modules.dsv41.moe._sm90_humming_linear import (
    SM90HummingGroupedLinear,
    decode_quantized_activation,
    unpack_activation_scales,
)
from rtp_llm.models_py.modules.dsv41.moe._sm90_route_kernels import reduce_routes, routed_swiglu
from rtp_llm.models_py.modules.dsv41.test.test_sm90_moe_gemm_candidates import (
    _independent_quantized_input,
    _independent_weight,
    _metrics,
    _quantize,
)
from rtp_llm.models_py.triton_kernels.moe.fused_moe_kernel import (
    moe_align_block_size_compiled,
    moe_align_block_size_torch,
)


def _align(ids, block_m):
    function = moe_align_block_size_compiled if _ALIGNMENT == "compiled" else moe_align_block_size_torch
    return function(ids, block_m, 48)


def _check_alignment(ids, block_m):
    """Identical-shaped changing routes, including active-expert count changes."""
    live = ids.clone()
    variants = [ids, (ids + 7) % 48, torch.full_like(ids, 47), ids]
    walls = []
    for changed in variants:
        live.copy_(changed)
        expected = moe_align_block_size_torch(live, block_m, 48)
        torch.cuda.synchronize()
        start = time.perf_counter()
        actual = _align(live, block_m)
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - start)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    return dict(mode=_ALIGNMENT, exact_metadata=True, same_shape_route_changes=4,
                initial_compile_or_warmup_wall_s=walls[0], subsequent_control_wall_s=walls[1:],
                timing_excluded=True, note="Compilation/graph breaks may remain; kernel count is measured, not assumed")


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="SM90, parent-owned GPU lock only",
)


@pytest.fixture(autouse=True)
def _strict_fp32():
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old


class _Projection:
    def __init__(self, linear, m, *, indexed, invariant=True, top_k=1):
        from humming.tune import get_heuristics_config

        self.linear, self.indexed = linear, indexed
        self.top_k = top_k
        self.compute = dict(gemm_type="indexed" if indexed else "grouped_contiguous",
                            use_f16_accum=False, use_batch_invariant=invariant)
        self.tuning = get_heuristics_config(
            linear.config, shape_m=m, device=linear.weight.device, **self.compute,
        )
        self.block_m = self.tuning["block_shape"][0]
        self.tuning_json = json.dumps(self.tuning)
        self.output = torch.empty((m, linear.out_features), dtype=torch.bfloat16, device="cuda")

    def prepare(self, x, ids):
        if self.indexed:
            # Existing RTP fixed-capacity route alignment; no new GEMM/kernel.
            metadata = _align(ids.reshape(-1, self.top_k), self.block_m)
            payload, scales = _quantize(x)
            return decode_quantized_activation(payload, scales), metadata, None
        slots = torch.where(ids >= 0)[0]
        experts = ids.index_select(0, slots)
        slots = slots.index_select(0, torch.argsort(experts, stable=True))
        counts = torch.bincount(experts.long(), minlength=48)
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0))).int()
        payload, scales = _quantize(x.index_select(0, slots).contiguous())
        return decode_quantized_activation(payload, scales), offsets, slots

    def gemm(self, prepared):
        from humming.forward import humming_forward

        inputs, metadata, _ = prepared
        args = dict(
            inputs=inputs, outputs=self.output, weight=self.linear.weight,
            weight_scale=self.linear.weight_scale,
            weight_scale_2=getattr(self.linear, "weight_scale_2", None),
            locks=self.linear.locks, compute_config=self.compute,
            tuning_config=self.tuning_json, top_k=self.top_k if self.indexed else 1, valid_shape_m=inputs.shape[0],
        )
        if self.indexed:
            args.update(zip(("sorted_ids", "expert_ids", "num_tokens_padded"), metadata))
        else:
            args["expert_layout"] = metadata
        return humming_forward(self.linear.config, **args)

    def __call__(self, x, ids):
        prepared = self.prepare(x, ids)
        out = self.gemm(prepared)
        if self.indexed:
            return out
        restored = torch.empty_like(out)
        restored.index_copy_(0, prepared[2], out)
        return restored


def _inputs(m, n, k, distribution):
    torch.manual_seed(20261017 + m)
    weight = torch.randint(-128, 128, (48, n, k // 2), device="cuda", dtype=torch.int8)
    exp = torch.randint(117, 128, (48, n, k // 32), device="cuda", dtype=torch.uint8)
    exp[:, :, 0], exp[:, :, 1] = 117, 127
    x = (torch.randn(m, k, device="cuda") * 12).bfloat16()
    if distribution == "hotspot":
        host_ids = [47] * m
    elif distribution == "sparse":
        host_ids = [(1, 17, 31)[i % 3] for i in range(m)]
    else:
        host_ids = [(i * 13) % 48 for i in range(m)]
    ids = torch.tensor(host_ids, device="cuda", dtype=torch.int32)
    return x, ids, host_ids, weight, exp.view(torch.float8_e8m0fnu)


def _oracle(x, host_ids, weight, scales):
    payload, packed = _quantize(x)
    qref, sfref = _independent_quantized_input(x)
    torch.testing.assert_close(payload.view(torch.uint8), qref.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(unpack_activation_scales(packed, x.shape[1] // 128), sfref, rtol=0, atol=0)
    values = (qref.float().reshape(x.shape[0], -1, 128) * sfref.unsqueeze(-1)).reshape_as(x)
    actual_decode = decode_quantized_activation(payload, packed)
    torch.testing.assert_close(actual_decode.float(), values, rtol=0, atol=0)
    ref = torch.empty((x.shape[0], weight.shape[1]), device="cuda", dtype=torch.bfloat16)
    for expert in sorted(set(host_ids)):
        rows = [i for i, e in enumerate(host_ids) if e == expert]
        decoded = _independent_weight(weight[expert], scales[expert])
        ref[rows] = (values[rows] @ decoded.T).bfloat16()
        del decoded
    return ref


def _tensor_hash(tensor):
    # Diagnostics only, never inside timing. Bound temporary CPU memory by row.
    digest = hashlib.sha256()
    for row in tensor.contiguous().view(torch.uint8).reshape(tensor.shape[0], -1):
        digest.update(row.cpu().numpy().tobytes())
    return digest.hexdigest()


def _source_metadata():
    import humming
    import humming.forward
    import humming.transform
    import humming.tune.sm90_h20

    modules = [humming, humming.forward, humming.transform, humming.tune.sm90_h20]
    paths = {module.__name__: Path(module.__file__).resolve() for module in modules}
    if any(not path.is_relative_to(_SOURCE_ROOT) for path in paths.values()):
        raise RuntimeError(f"Mixed Humming source roots: {paths}")
    return dict(mode="installed_baseline_only" if _BASELINE_ONLY else "installed_candidate" if _USE_INSTALLED else "pinned_upstream",
                revision=None if _USE_INSTALLED else _REVISION,
                version=getattr(humming, "__version__", "unknown"),
                files={name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                       for name, path in paths.items()})


def _persist(directory, record):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "result.json").write_text(json.dumps(record, indent=2))


def _print_summary(record):
    print("SM90_INDEXED_UPSTREAM " + json.dumps({
        key: record[key] for key in ("decision", "M_routed", "distribution", "alignment", "output_directory")
        if key in record
    }), flush=True)
    for name, modes in record.get("measurements", {}).items():
        for mode, rounds in modes.items():
            print(f"  {name} {mode}: stream_span_ms="
                  f"{statistics.median(r['median_ms'] for r in rounds):.6f}, "
                  f"kernel_us={statistics.median(r['profile']['kernel_sum_us_per_iter'] for r in rounds):.3f}", flush=True)


def _profile(fn, flush, trace_path, *, prepared, expected_humming=1):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    if flush is not None:
        flush.zero_()
        torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    p.export_chrome_trace(str(trace_path))
    counts, sums = Counter(), Counter()
    for event in p.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            counts[event.name] += 1
            sums[event.name] += event.time_range.elapsed_us()
    kernel_counts = {n: c for n, c in counts.items() if "memcpy" not in n.lower() and "memset" not in n.lower()}
    humming_count = sum(c for n, c in kernel_counts.items() if re.search(r"(?:^|::|\s)humming(?:[<(]|$)", n))
    result = dict(iterations=1, kernels=[dict(name=n, count=counts[n], us=sums[n]) for n in sorted(counts)],
                  kernel_sum_us_per_iter=sum(sums[n] for n in kernel_counts),
                  humming_launch_count=humming_count, pure_kernel_launch_count=sum(kernel_counts.values()),
                  trace=str(trace_path), prepared=prepared)
    assert humming_count == expected_humming, f"Expected {expected_humming} Humming GEMMs: {result}"
    if prepared:
        assert sum(kernel_counts.values()) == 1, f"Prepared GEMM unexpectedly launched extra kernels: {result}"
    return result


def _measure(fn, flush, trace_path, *, prepared, expected_humming=1):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(13):
        if flush is not None:
            flush.zero_()
            torch.cuda.synchronize()
        begin.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    return dict(stream_span_ms=samples, median_ms=statistics.median(samples),
                profile=_profile(fn, flush, trace_path, prepared=prepared, expected_humming=expected_humming))


def _run(m, n, k, distribution, invariant):
    if _BASELINE_ONLY and not invariant:
        pytest.skip("Installed baseline-only: no indexed/native candidate constructed")
    case_dir = _OUTPUT_ROOT / (
        f"{'installed_baseline' if _BASELINE_ONLY else 'installed_candidate' if _USE_INSTALLED else 'upstream'}_m{m}_n{n}_k{k}_{distribution}_"
        f"{'invariant' if invariant else 'native'}_{_ALIGNMENT}"
    )
    case_dir.mkdir(parents=True, exist_ok=False)
    x, ids, host_ids, weight, scales = _inputs(m, n, k, distribution)
    reference = _oracle(x, host_ids, weight, scales)
    linear = SM90HummingGroupedLinear(weight, scales)
    baseline = _Projection(linear, m, indexed=False)
    candidate = None if _BASELINE_ONLY else _Projection(linear, m, indexed=True, invariant=invariant)
    alignment_check = None if candidate is None else _check_alignment(ids[:, None], candidate.block_m)
    old = baseline(x, ids).clone()
    actual = old if candidate is None else candidate(x, ids).clone()
    torch.testing.assert_close(old, reference, rtol=.02, atol=.05)
    error = _metrics(actual, reference)
    versus_baseline = _metrics(actual, old)
    checked = baseline if candidate is None else candidate
    repeatable = all(torch.equal(checked(x, ids).view(torch.int16), actual.view(torch.int16)) for _ in range(5))
    bitwise = bool(torch.equal(actual.view(torch.int16), old.view(torch.int16)))
    accepted = (not error["outside_existing_tolerance"] and not versus_baseline["outside_existing_tolerance"]
                and repeatable and (bitwise or not invariant))
    record = dict(source_metadata=_source_metadata(), source=str(_SOURCE_ROOT), M_routed=m, N=n, K=k, E=48,
                  distribution=distribution, alignment=_ALIGNMENT, alignment_check=alignment_check,
                  counts=[host_ids.count(i) for i in range(48)],
                  invariant=invariant, error=error, versus_baseline=versus_baseline,
                  baseline_bitwise=bitwise, deterministic_5_repeats=repeatable,
                  candidate_tuning=None if candidate is None else candidate.tuning, baseline_tuning=baseline.tuning,
                  decision=("BASELINE_ONLY" if candidate is None else "BOUNDED_SUPPORT") if accepted else "REJECT",
                  baseline_scope="RTP grouped algorithm with explicit version-specific heuristic; not untouched production default dispatch",
                  timing_scope="eager stream-span includes wrapper/temporary allocator/launch gaps; no graph",
                  moe_scope="single-projection chain; summing two chains double-counts routing metadata. Production shares slots/offsets across w13/down; full-MoE speed NOT_RUN",
                  route_scope="valid local ids only; empty experts covered; EP -1 padding NOT_RUN",
                  hashes=dict(input=_tensor_hash(x), ids=_tensor_hash(ids), raw_weight=_tensor_hash(weight),
                              raw_scale=_tensor_hash(scales), oracle=_tensor_hash(reference),
                              baseline_output=_tensor_hash(old), candidate_output=None if candidate is None else _tensor_hash(actual)),
                  output_directory=str(case_dir))
    _persist(case_dir, record)
    if accepted:
        flush = torch.empty(256 * 1024**2, dtype=torch.uint8, device="cuda")
        bprep = baseline.prepare(x, ids)
        functions = [("baseline_chain", lambda: baseline(x, ids)),
                     ("baseline_prepared_gemm", lambda: baseline.gemm(bprep))]
        if candidate is not None:
            cprep = candidate.prepare(x, ids)
            functions = [("baseline_chain", lambda: baseline(x, ids)),
                         ("indexed_chain", lambda: candidate(x, ids)),
                         ("baseline_prepared_gemm", lambda: baseline.gemm(bprep)),
                         ("indexed_prepared_gemm", lambda: candidate.gemm(cprep))]
        results = {name: {"hot": [], "cold": []} for name, _ in functions}
        for repeat in range(3):
            ordered = functions if repeat % 2 == 0 else list(reversed(functions))
            for name, fn in ordered:
                for mode, eviction in (("hot", None), ("cold", flush)):
                    results[name][mode].append(_measure(
                        fn, eviction, case_dir / f"{name}_{mode}_r{repeat}.trace.json",
                        prepared="prepared_gemm" in name,
                    ))
        record["measurements"] = results
        record["l2_flush_bytes"] = flush.numel()
    _persist(case_dir, record)
    _print_summary(record)
    assert accepted, "Original numerical/determinism gate failed; no promotion"


def test_indexed_fixture():
    _run(8, 256, 256, "sparse", True)


@pytest.mark.parametrize("invariant", [True, False], ids=["invariant", "native"])
@pytest.mark.parametrize("m", [8, 16, 24, 40], ids=["m8", "m16", "m24", "m40"])
@pytest.mark.parametrize("distribution", ["sparse", "hotspot", "balanced"])
@pytest.mark.parametrize("n,k", [(4608, 5120), (5120, 2304)], ids=["w13", "down"])
def test_indexed_real(m, n, k, distribution, invariant):
    _run(m, n, k, distribution, invariant)


def test_indexed_weighted_down_boundary():
    """Small full two-projection correctness fixture; no moved route weights."""
    if _BASELINE_ONLY:
        pytest.skip("Installed mode is a grouped projection baseline only")
    x, ids, host_ids, w13, s13 = _inputs(8, 256, 256, "sparse")
    _, _, _, down_w, down_s = _inputs(8, 256, 128, "sparse")
    l1, l2 = SM90HummingGroupedLinear(w13, s13), SM90HummingGroupedLinear(down_w, down_s)
    route_weights = torch.linspace(.03, .97, 8, device="cuda")
    slots = torch.arange(8, device="cuda", dtype=torch.int64)
    reference1 = _oracle(x, host_ids, w13, s13)
    gate, up = reference1.float().chunk(2, -1)
    hidden_ref = (torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
                  * route_weights[:, None]).bfloat16()
    reference2 = _oracle(hidden_ref, host_ids, down_w, down_s)
    results = []
    for indexed in (False, True):
        first, second = _Projection(l1, 8, indexed=indexed), _Projection(l2, 8, indexed=indexed)
        hidden = routed_swiglu(first(x, ids), route_weights, slots, clamp_limit=10, output_dtype=torch.bfloat16)
        output = second(hidden, ids).clone()
        torch.testing.assert_close(hidden, hidden_ref, rtol=.02, atol=.05)
        torch.testing.assert_close(output, reference2, rtol=.02, atol=.05)
        results.append(output)
    torch.testing.assert_close(results[0].view(torch.int16), results[1].view(torch.int16), rtol=0, atol=0)
    _persist(_OUTPUT_ROOT / "weighted_down_boundary", dict(
        source_metadata=_source_metadata(), decision="BOUNDED_SUPPORT",
        route_contract="clamp10 -> SiLU/up product -> routeweight -> BF16 -> original g128 -> down",
        hashes=dict(input=_tensor_hash(x), ids=_tensor_hash(ids), route_weights=_tensor_hash(route_weights),
                    baseline_output=_tensor_hash(results[0]), indexed_output=_tensor_hash(results[1]),
                    oracle_output=_tensor_hash(reference2)),
        baseline_error=_metrics(results[0], reference2), indexed_error=_metrics(results[1], reference2),
        negative_one_padding="NOT_RUN",
    ))


def _decoded(x):
    q, scales = _quantize(x)
    return decode_quantized_activation(q, scales)


class _LocalMoe:
    """Two projections share routing once; valid local routes only, no EP comm."""
    def __init__(self, first, second, m, *, indexed):
        self.indexed = indexed
        self.first = _Projection(first, m, indexed=indexed, top_k=2)
        self.second = _Projection(second, m, indexed=indexed)
        self.route_slots = torch.arange(m, dtype=torch.int64, device="cuda")

    def __call__(self, x, ids, weights):
        if self.indexed:
            metadata = _align(ids, self.first.block_m)
            metadata2 = metadata if self.second.block_m == self.first.block_m else _align(ids, self.second.block_m)
            slots = self.route_slots
            first_input = _decoded(x)
        else:
            flat = ids.flatten()
            slots = torch.where(flat >= 0)[0]
            experts = flat.index_select(0, slots)
            slots = slots.index_select(0, torch.argsort(experts, stable=True))
            counts = torch.bincount(experts.long(), minlength=48)
            metadata = torch.cat((counts.new_zeros(1), counts.cumsum(0))).int()
            metadata2 = metadata  # Production's one shared grouped route layout.
            first_input = _decoded(x.index_select(0, slots // 2).contiguous())
        gate_up = self.first.gemm((first_input, metadata, slots))
        hidden = routed_swiglu(gate_up, weights, slots, clamp_limit=10, output_dtype=torch.bfloat16)
        down = self.second.gemm((_decoded(hidden), metadata2, slots))
        return reduce_routes(down, slots, x.shape[0], 2)


def _run_local_moe(m, hidden, intermediate, distribution):
    case_dir = _OUTPUT_ROOT / f"local_moe_{'installed_baseline' if _BASELINE_ONLY else 'installed_candidate' if _USE_INSTALLED else 'upstream'}_m{m}_h{hidden}_i{intermediate}_{distribution}_{_ALIGNMENT}"
    case_dir.mkdir(parents=True, exist_ok=False)
    raw_x, ids, host_ids, w13, s13 = _inputs(m, 2 * intermediate, hidden, distribution)
    _, _, _, w2, s2 = _inputs(m, hidden, intermediate, distribution)
    # Top-k2 is a bounded local-MoE fixture, NOT the model's global top-k6.
    # Each request's two selections are distinct, including the hotspot case.
    if distribution == "hotspot":
        host_ids = [46 + i % 2 for i in range(m)]
        ids = torch.tensor(host_ids, device="cuda", dtype=torch.int32)
    ids = ids.reshape(m // 2, 2)
    x = raw_x[:m // 2].contiguous()
    weights = torch.linspace(.03, .97, m, device="cuda").reshape(m // 2, 2)
    routed_x = x.repeat_interleave(2, dim=0)
    ref_first = _oracle(routed_x, host_ids, w13, s13)
    gate, up = ref_first.float().chunk(2, -1)
    ref_hidden = (torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
                  * weights.flatten()[:, None]).bfloat16()
    ref_down = _oracle(ref_hidden, host_ids, w2, s2)
    reference = ref_down.float().reshape(m // 2, 2, hidden).sum(1)
    l1, l2 = SM90HummingGroupedLinear(w13, s13), SM90HummingGroupedLinear(w2, s2)
    baseline = _LocalMoe(l1, l2, m, indexed=False)
    candidate = None if _BASELINE_ONLY else _LocalMoe(l1, l2, m, indexed=True)
    alignment_checks = []
    if candidate is not None:
        for block_m in sorted({candidate.first.block_m, candidate.second.block_m}):
            alignment_checks.append(_check_alignment(ids, block_m))
    old = baseline(x, ids, weights).clone()
    torch.testing.assert_close(old, reference, rtol=.02, atol=.05)
    actual = old if candidate is None else candidate(x, ids, weights).clone()
    error = _metrics(actual, reference)
    bitwise = bool(torch.equal(actual.view(torch.int32), old.view(torch.int32)))
    checked = baseline if candidate is None else candidate
    repeatable = all(torch.equal(checked(x, ids, weights).view(torch.int32), actual.view(torch.int32)) for _ in range(5))
    accepted = error["outside_existing_tolerance"] == 0 and bitwise and repeatable
    record = dict(
        decision=("BASELINE_ONLY" if candidate is None else "BOUNDED_SUPPORT") if accepted else "REJECT",
        source_metadata=_source_metadata(), M_routed=m, requests=m // 2, local_topk=2,
        hidden=hidden, intermediate=intermediate, distribution=distribution, alignment=_ALIGNMENT,
        alignment_checks=alignment_checks, baseline_bitwise=bitwise, deterministic_5_repeats=repeatable,
        error=error, baseline_error=_metrics(old, reference), counts=[host_ids.count(i) for i in range(48)],
        metadata_builds_baseline=1,
        metadata_builds_indexed=None if candidate is None else 1 + int(candidate.first.block_m != candidate.second.block_m),
        scope="full local routed chain: align/gather + g128/decode + w13 + clamp10/SiLU/routeweight/BF16 + g128/decode + down + reduce; EP/shared/model NOT_RUN",
        route_scope="synthetic local topk2 with valid unique IDs; empty experts covered; -1 NOT_RUN",
        timing_scope="eager stream-span includes temporary allocator/wrapper overhead; pure kernel profile separate",
        output_directory=str(case_dir),
        hashes=dict(input=_tensor_hash(x), ids=_tensor_hash(ids), weights=_tensor_hash(weights),
                    w13=_tensor_hash(w13), s13=_tensor_hash(s13), w2=_tensor_hash(w2), s2=_tensor_hash(s2),
                    oracle=_tensor_hash(reference), baseline_output=_tensor_hash(old),
                    indexed_output=None if candidate is None else _tensor_hash(actual)),
    )
    _persist(case_dir, record)
    assert accepted, "Local MoE original oracle/bitwise/determinism gate failed"
    flush = torch.empty(256 * 1024**2, device="cuda", dtype=torch.uint8)
    functions = [("baseline_local_moe", lambda: baseline(x, ids, weights))]
    if candidate is not None:
        functions.append(("indexed_local_moe", lambda: candidate(x, ids, weights)))
    results = {name: {"hot": [], "cold": []} for name, _ in functions}
    for repeat in range(3):
        ordered = functions if repeat % 2 == 0 else list(reversed(functions))
        for name, fn in ordered:
            for mode, eviction in (("hot", None), ("cold", flush)):
                results[name][mode].append(_measure(
                    fn, eviction, case_dir / f"{name}_{mode}_r{repeat}.trace.json",
                    prepared=False, expected_humming=2,
                ))
    record["measurements"] = results
    _persist(case_dir, record)
    _print_summary(record)


def test_indexed_local_moe_fixture():
    _run_local_moe(8, 256, 128, "sparse")


@pytest.mark.parametrize("m", [8, 16, 24, 40], ids=["m8", "m16", "m24", "m40"])
@pytest.mark.parametrize("distribution", ["sparse", "hotspot"])
def test_indexed_local_moe_real(m, distribution):
    _run_local_moe(m, 5120, 2304, distribution)
