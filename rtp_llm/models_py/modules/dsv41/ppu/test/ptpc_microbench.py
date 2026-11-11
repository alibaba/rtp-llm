"""Component-only PTPC correctness, stream latency and GPU kernel evidence.

This entry loads no model, checkpoint or server. CUDA events measure a whole
call on its stream, including host launch gaps. Separate profiler traces supply
actual GPU kernel durations and counts; those metrics are never inferred from
the CUDA event measurements.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import statistics

import torch

from rtp_llm.models_py.modules.dsv41.ppu import linear as linear_module
from rtp_llm.models_py.modules.dsv41.ppu.linear import PPUPTPCLinear
from rtp_llm.models_py.modules.dsv41.ppu.ptpc_norm import ptpc_rms_norm
from rtp_llm.models_py.modules.dsv41.ppu.ptpc_quant import (
    quantize_ptpc_activation,
    quantize_ptpc_grouped_activation,
    ptpc_fusion_enabled,
)


def old_quant(value):
    """Frozen pre-fusion expression, independent of the production helper."""
    values = value.float()
    amax = values.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    scale = amax * 0.0022321429569274187
    payload = (values * (448.0 / amax)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return payload.contiguous(), scale.contiguous()


def old_norm(value, gamma, eps):
    values = value.float()
    inverse = torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    return (values * inverse * gamma.float()).to(value.dtype)


def old_linear(op, value):
    payload, scale = old_quant(value)
    out = torch.empty(
        (value.shape[0], op.weight.shape[0]), device=value.device,
        dtype=torch.bfloat16,
    )
    op._gemm((payload, scale), (op.weight, op.scale), out)
    return out


def assert_exact(actual, expected, label):
    if isinstance(expected, tuple):
        if not isinstance(actual, tuple) or len(actual) != len(expected):
            raise AssertionError(f"{label}: tuple contract changed")
        return [assert_exact(a, b, f"{label}/{i}")
                for i, (a, b) in enumerate(zip(actual, expected))]
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise AssertionError(
            f"{label}: dtype/shape changed: {actual.dtype} {actual.shape} "
            f"vs {expected.dtype} {expected.shape}"
        )
    if actual.dtype == torch.float8_e4m3fn:
        bits = torch.uint8
    elif actual.dtype == torch.float32:
        bits = torch.int32
    elif actual.dtype == torch.bfloat16:
        bits = torch.int16
    else:
        raise AssertionError(f"{label}: unexpected comparison dtype {actual.dtype}")
    left, right = actual.contiguous().view(bits), expected.contiguous().view(bits)
    if not torch.equal(left, right):
        left = actual.contiguous().view(torch.uint8).reshape(-1)
        right = expected.contiguous().view(torch.uint8).reshape(-1)
        offset = int((left != right).nonzero()[0].item())
        raise AssertionError(
            f"{label}: byte {offset}: actual=0x{int(left[offset].item()):02x}, "
            f"expected=0x{int(right[offset].item()):02x}"
        )
    return dict(exact=True, dtype=str(actual.dtype), shape=list(actual.shape),
                compared_bytes=actual.numel() * actual.element_size())


def percentile(values, fraction):
    ordered = sorted(values)
    coordinate = (len(ordered) - 1) * fraction
    low = int(coordinate)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (coordinate - low)


def stream_latency(call, warmup, measure):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(measure):
        start.record()
        output = call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
        del output
    return dict(
        metric="stream_call_latency_us_including_host_launch_gaps",
        warmup=warmup, measure=measure, compile_warmup_excluded=True,
        median_us=statistics.median(samples), p90_us=percentile(samples, 0.9),
        min_us=min(samples), max_us=max(samples), samples_us=samples,
    )


def kernel_profile(call, path, iterations):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite profiler evidence: {path}")
    try:
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False, with_stack=False,
        ) as profiler:
            for _ in range(iterations):
                output = call()
            torch.cuda.synchronize()
            del output
        profiler.export_chrome_trace(str(path))
    except (RuntimeError, AttributeError, NotImplementedError) as error:
        return dict(status="NOT_RUN", reason=f"GPU profiler unavailable: {error}",
                    trace=path.name, iterations=iterations)
    events = json.loads(path.read_text())["traceEvents"]
    kernels = [event for event in events
               if event.get("cat") == "kernel" and event.get("ph") == "X"]
    if not kernels:
        return dict(status="NOT_RUN", reason="trace contains no actual GPU kernel events",
                    trace=path.name, iterations=iterations)
    counts = Counter(event["name"] for event in kernels)
    if len(kernels) % iterations or any(count % iterations for count in counts.values()):
        raise AssertionError(f"unstable per-invocation kernel counts in {path.name}: {counts}")
    durations = Counter()
    for event in kernels:
        durations[event["name"]] += event["dur"]
    copies = [event for event in events if event.get("ph") == "X"
              and event.get("cat") in ("gpu_memcpy", "gpu_memset")]
    return dict(
        status="PASS", trace=path.name, iterations=iterations,
        metric="sum_of_actual_GPU_kernel_durations_us",
        kernel_count_total=len(kernels), kernel_count_per_call=len(kernels) // iterations,
        kernel_sum_us_per_call=sum(event["dur"] for event in kernels) / iterations,
        gpu_copy_or_memset_event_count=len(copies),
        gpu_copy_or_memset_sum_us=sum(event["dur"] for event in copies),
        kernels=[dict(name=name, count_total=counts[name], count_per_call=counts[name] // iterations,
                      duration_us_per_call=durations[name] / iterations)
                 for name in sorted(counts)],
    )


def run_case(name, shape, baseline, candidate, args, extra_checks=None):
    expected, actual = baseline(), candidate()
    exact = assert_exact(actual, expected, name)
    del expected, actual
    torch.cuda.synchronize()
    record = dict(name=name, shape=shape, exact=exact, extra_exact=extra_checks)
    for variant, call in (("baseline", baseline), ("candidate", candidate)):
        record[variant] = dict(
            event=stream_latency(call, args.warmup, args.measure),
            profile=kernel_profile(call, args.result_dir / f"{name}-{variant}.json", args.profile),
        )
    record["stream_call_median_speedup"] = (
        record["baseline"]["event"]["median_us"] / record["candidate"]["event"]["median_us"]
    )
    profiles = [record[variant]["profile"] for variant in ("baseline", "candidate")]
    record["status"] = "PASS" if all(p["status"] == "PASS" for p in profiles) else "TIMING_ONLY"
    if record["status"] == "PASS":
        record["kernel_sum_speedup"] = profiles[0]["kernel_sum_us_per_call"] / profiles[1]["kernel_sum_us_per_call"]
    (args.result_dir / f"{name}-result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(name=name, status=record["status"], exact=True,
                          stream_call_median_speedup=record["stream_call_median_speedup"],
                          kernel_counts=[p.get("kernel_count_per_call") for p in profiles])), flush=True)
    return record


def random_input(shape):
    return torch.randn(shape, device="cuda", dtype=torch.bfloat16)


def make_linear(n, k):
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    scale = torch.rand((n, 1), device="cuda", dtype=torch.float32) * 0.005 + 0.001
    return PPUPTPCLinear(weight, scale)


def grouped_producer_check(value):
    parts = [old_quant(value[:, group].contiguous()) for group in range(value.shape[1])]
    payload = torch.stack([part[0].view(torch.uint8) for part in parts], 0).view(torch.float8_e4m3fn)
    scale = torch.stack([part[1] for part in parts], 0)
    return assert_exact(quantize_ptpc_grouped_activation(value), (payload, scale), "grouped producer")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"))
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--measure", type=int, default=100)
    parser.add_argument("--profile", type=int, default=5)
    parser.add_argument("--cases", default="quant,norm,dense,grouped")
    args = parser.parse_args()
    if args.result_dir is None or min(args.warmup, args.measure, args.profile) <= 0:
        parser.error("an output directory and positive iteration counts are required")
    selected = args.cases.split(",")
    if len(set(selected)) != len(selected) or not set(selected) <= {"quant", "norm", "dense", "grouped"}:
        parser.error("--cases must select unique quant,norm,dense,grouped entries")
    if not torch.cuda.is_available():
        raise RuntimeError("this component benchmark requires an explicitly allocated PPU")
    if not ptpc_fusion_enabled():
        raise RuntimeError("component candidate requires RTP_V41_PPU_PTPC_FUSION=1")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.result_dir / "ptpc-microbench-summary.json"
    if summary_path.exists():
        raise FileExistsError("refusing to overwrite a previous benchmark summary")
    torch.manual_seed(20261111)
    snapshot_path = Path(linear_module.__file__).resolve().parents[5] / ".v41_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else {}
    summary = dict(
        scope="component correctness and operator microbenchmarks; no model/service run",
        device=torch.cuda.get_device_name(0), torch=torch.__version__,
        dtype="BF16 input and output; E4M3 payload; FP32 row/channel scales",
        native_gemm="existing deep_gemm.fp8_gemm_nt / fp8_einsum", torch_compile=False,
        graph=False, input_allocation_outside_timing=True,
        ptpc_fusion_enabled=True,
        output_allocation_included=True,
        snapshot={key: snapshot.get(key) for key in ("snapshot_id", "source_sha256", "public_commit", "internal_commit")},
        warmup=args.warmup, measure=args.measure, profile_iterations=args.profile,
        cases=[], status="RUNNING",
    )

    def save():
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    save()
    if "quant" in selected:
        quant_shapes = [(144, 4096), (4096, 5120), (4096, 8192), (4096, 15360)]
        quant_shapes.extend((m, k) for m in (8, 24, 32) for k in (1280, 4096, 5120, 8192, 15360))
        for m, k in quant_shapes:
            x = random_input((m, k))
            x[0].zero_()
            summary["cases"].append(run_case(
                f"quant-M{m}-K{k}", dict(m=m, k=k, workload="decode" if m in (8, 24, 32) else "prefill"),
                lambda: old_quant(x), lambda: quantize_ptpc_activation(x), args,
            ))
            save()
    if "norm" in selected:
        for m in (144, 4096, 8, 24, 32):
            for k in ((512, 1280, 5120) if m in (8, 24, 32) else (1280, 5120)):
                x, gamma = random_input((m, k)), random_input((k,))
                summary["cases"].append(run_case(
                    f"norm-M{m}-K{k}", dict(m=m, k=k, eps=1e-6, workload="decode" if m in (8, 24, 32) else "prefill"),
                    lambda: old_norm(x, gamma, 1e-6), lambda: ptpc_rms_norm(x, gamma, 1e-6), args,
                ))
                save()
    if "dense" in selected:
        for m in (144, 4096, 8, 24, 32):
            x, op = random_input((m, 5120)), make_linear(1280, 5120)
            summary["cases"].append(run_case(
                f"dense-M{m}-N1280-K5120", dict(m=m, n=1280, k=5120, workload="decode" if m in (8, 24, 32) else "prefill"),
                lambda: old_linear(op, x), lambda: op(x), args,
            ))
            save()
    if "grouped" in selected:
        for m in (144, 4096, 8, 24, 32):
            x, op = random_input((m, 8, 4096)), make_linear(8192, 4096)
            op.prepare_grouped(8, 1024)
            producer = grouped_producer_check(x)
            summary["cases"].append(run_case(
                f"grouped-M{m}-G8-N1024-K4096",
                dict(m=m, groups=8, n=1024, k=4096, workload="decode" if m in (8, 24, 32) else "prefill",
                     provider="fp8_einsum" if op._group_gemm is not None else "eight_dense_and_pack"),
                lambda: torch.stack([old_linear(part, x[:, i].contiguous())
                                     for i, part in enumerate(op._groups)], 1),
                lambda: op.forward_grouped(x), args, extra_checks=producer,
            ))
            save()
    summary["status"] = "PASS" if all(case["status"] == "PASS" for case in summary["cases"]) else "NOT_RUN_KERNEL_PROFILE"
    save()
    print(json.dumps(dict(status=summary["status"], cases=len(summary["cases"]), summary=str(summary_path))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
