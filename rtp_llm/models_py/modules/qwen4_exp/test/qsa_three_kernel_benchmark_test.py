"""Warm old-pair versus full prepared projection evidence on SM90.

No latency threshold is asserted. CUDA event spans include gaps between
submissions, while profiler kernel-duration sums exclude those gaps. The host
measurement covers only the callable's enqueue work, not event record/wait.
Library workspace memsets remain part of measured cost and trace evidence.
This is a projection component benchmark, not an end-to-end model benchmark.
"""

import collections
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from rtp_llm.models_py.modules.base.common import fp8_linear, qsa_fp8_pair
from rtp_llm.models_py.modules.base.common.fp8_linear import (
    Fp8Linear,
    prepare_fp8_pair,
    project_fp8_pair,
)

ROWS = (1, 127, 128, 1371, 1576, 2048, 4096, 8192, 8193, 16384, 32768, 32769)
BASELINE = "baseline_old_pair"
CANDIDATE = "candidate_prepared_pair"


def _gpu_snapshot():
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    raw_uuid = str(getattr(properties, "uuid", "") or "")
    uuid = raw_uuid
    if re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", uuid):
        uuid = "GPU-" + uuid
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    tokens = [value.strip() for value in visible.split(",") if value.strip()]
    logical = torch.cuda.current_device()
    selectors = [uuid] if uuid.startswith(("GPU-", "MIG-")) else []
    if logical < len(tokens) and tokens[logical] not in selectors:
        selectors.append(tokens[logical])
    fields = (
        "index,uuid,name,driver_version,pstate,utilization.gpu,memory.used,"
        "clocks.current.sm,clocks.current.memory,power.draw,power.limit,temperature.gpu"
    )
    record = {
        "time_ns": time.time_ns(),
        "monotonic_ns": time.perf_counter_ns(),
        "raw_uuid": raw_uuid,
        "visible_devices": visible,
        "logical_device": logical,
        "fields": fields.split(","),
        "attempts": [],
    }
    for selector in selectors:
        try:
            done = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    selector,
                    "--query-gpu=" + fields,
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            attempt = {"selector": selector, "returncode": done.returncode}
            record["attempts"].append(attempt)
            if done.returncode == 0:
                record.update(selector=selector, returncode=0, csv=done.stdout.strip())
                return record
            attempt["stderr"] = done.stderr.strip()[:1000]
        except (OSError, subprocess.TimeoutExpired) as error:
            record["attempts"].append(
                {"selector": selector, "error": str(error)[:1000]}
            )
    record["unavailable"] = "No successful physical-GPU query; no identity guessed"
    return record


def _distribution(values):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * 0.9
    lo = int(pos)
    return {
        "n": len(values),
        "median": statistics.median(values),
        "p90": ordered[lo]
        + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (pos - lo),
        "min": min(values),
        "max": max(values),
    }


def _interleaved_timings(functions, warm_pairs=8, measured_pairs=40, telemetry=None):
    """Alternate A/B then B/A; synchronize the measured stream per sample."""
    names = tuple(functions)
    stream = torch.cuda.current_stream()
    if telemetry is not None:
        # Keep a potentially slow query before warmup, not between warming and
        # the first sample. Queries never alter GPU clocks or power settings.
        telemetry["pre_warm_gpu"] = _gpu_snapshot()
    for iteration in range(warm_pairs):
        for name in names if iteration % 2 == 0 else names[::-1]:
            functions[name]()
    stream.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    # Materialize event handles outside the recorded host intervals.
    start.record(stream)
    stop.record(stream)
    stop.synchronize()
    samples = {name: [] for name in names}
    for iteration in range(measured_pairs):
        order = names if iteration % 2 == 0 else names[::-1]
        for position, name in enumerate(order):
            stream.synchronize()
            start.record(stream)
            before = time.perf_counter_ns()
            result = functions[name]()
            enqueue_us = (time.perf_counter_ns() - before) / 1000.0
            stop.record(stream)
            stop.synchronize()
            if result is None:
                raise AssertionError(f"benchmark unexpectedly declined {name}")
            samples[name].append(
                {
                    "iteration": iteration,
                    "order_position": position,
                    "monotonic_start_ns": before,
                    "host_enqueue_us": enqueue_us,
                    "cuda_event_span_us": start.elapsed_time(stop) * 1000.0,
                }
            )
            del result
    if telemetry is not None:
        telemetry["post_measurement_gpu"] = _gpu_snapshot()
    return {
        name: {
            "host_enqueue_us": _distribution([x["host_enqueue_us"] for x in rows]),
            "cuda_event_span_us": _distribution(
                [x["cuda_event_span_us"] for x in rows]
            ),
            "samples": rows,
            "first_ten": {
                metric: _distribution([row[metric] for row in rows[:10]])
                for metric in ("host_enqueue_us", "cuda_event_span_us")
            },
            "last_ten": {
                metric: _distribution([row[metric] for row in rows[-10:]])
                for metric in ("host_enqueue_us", "cuda_event_span_us")
            },
        }
        for name, rows in samples.items()
    }


def _profile_call(function, path):
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.current_stream().synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        result = function()
        torch.cuda.current_stream().synchronize()
    if result is None:
        raise AssertionError("profiled projection unexpectedly declined")
    prof.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    apis = {
        event.get("args", {}).get("correlation"): event
        for event in events
        if event.get("cat") in ("cuda_runtime", "cuda_driver")
        and event.get("ph") == "X"
    }
    device_ops = []
    for index, event in enumerate(events):
        if event.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        correlation = event.get("args", {}).get("correlation")
        api = apis.get(correlation, {})
        external = api.get("args", {}).get("External id")
        owners = [
            cpu
            for cpu in events
            if cpu.get("cat") == "cpu_op"
            and cpu.get("args", {}).get("External id") == external
            and cpu.get("pid") == api.get("pid")
            and cpu.get("tid") == api.get("tid")
        ]
        device_ops.append(
            {
                "event_id": index,
                "category": event["cat"],
                "name": event["name"],
                "duration_us": event.get("dur", 0),
                "bytes": event.get("args", {}).get("bytes"),
                "correlation": correlation,
                "api_name": api.get("name"),
                "api_duration_us": api.get("dur"),
                "external_id": external,
                "cpu_owners": sorted({cpu["name"] for cpu in owners}),
            }
        )
    counts = collections.Counter(op["category"] for op in device_ops)
    return {
        "trace": path.name,
        "kernel_count": counts["kernel"],
        "memcpy_count": counts["gpu_memcpy"],
        "memset_count": counts["gpu_memset"],
        "kernel_duration_sum_us": sum(
            op["duration_us"] for op in device_ops if op["category"] == "kernel"
        ),
        "memset_duration_sum_us": sum(
            op["duration_us"] for op in device_ops if op["category"] == "gpu_memset"
        ),
        "memcpy_duration_sum_us": sum(
            op["duration_us"] for op in device_ops if op["category"] == "gpu_memcpy"
        ),
        "memset_bytes": sum(
            op["bytes"] or 0 for op in device_ops if op["category"] == "gpu_memset"
        ),
        "memcpy_bytes": sum(
            op["bytes"] or 0 for op in device_ops if op["category"] == "gpu_memcpy"
        ),
        "device_operations": device_ops,
    }


def _paired_deltas(timings, baseline=BASELINE, candidate=CANDIDATE):
    old = timings[baseline]["samples"]
    new = timings[candidate]["samples"]
    if [row["iteration"] for row in old] != [row["iteration"] for row in new]:
        raise AssertionError("A/B sample iterations do not align")
    return {
        metric: _distribution([b[metric] - a[metric] for a, b in zip(old, new)])
        for metric in ("host_enqueue_us", "cuda_event_span_us")
    }


def _numeric_metrics(actual, reference):
    actual_f, reference_f = actual.float(), reference.float()
    error = actual_f - reference_f
    relative_rms = (
        error.square().mean().sqrt()
        / reference_f.square().mean().sqrt().clamp_min(1.0e-12)
    )

    def number(value):
        value = float(value)
        return value if math.isfinite(value) else str(value)

    return {
        "max_absolute_error": number(error.abs().max().item()),
        "relative_rms_error": number(relative_rms.item()),
        "different_elements": int((actual != reference).count_nonzero().item()),
        "nonfinite_elements": int((~torch.isfinite(actual_f)).count_nonzero().item()),
        "elements": actual.numel(),
    }


def _numeric_pair(x, old, new, projections, failures, label):
    record = {}
    for name, baseline, candidate, projection in zip(
        ("gate", "qk"), old, new, projections
    ):
        reference = F.linear(
            x, (projection.weight.float() * projection.scale).to(torch.bfloat16)
        )
        item = {
            "candidate_vs_old": _numeric_metrics(candidate, baseline),
            "old_vs_bf16": _numeric_metrics(baseline, reference),
            "candidate_vs_bf16": _numeric_metrics(candidate, reference),
        }
        try:
            torch.testing.assert_close(candidate, baseline, rtol=1.0e-2, atol=1.0e-3)
            item["assert_close_pass"] = True
        except AssertionError as error:
            item.update(assert_close_pass=False, assert_close_error=str(error))
            failures.append(
                {"case": label, "projection": name, "kind": "assert_close_old"}
            )
        old_rms = item["old_vs_bf16"]["relative_rms_error"]
        new_rms = item["candidate_vs_bf16"]["relative_rms_error"]
        baseline_pass = isinstance(old_rms, float) and old_rms < 0.04
        candidate_pass = (
            isinstance(old_rms, float)
            and isinstance(new_rms, float)
            and new_rms < 0.04
            and new_rms <= old_rms * 1.10 + 1.0e-4
        )
        item.update(
            baseline_reference_gate_pass=baseline_pass,
            candidate_reference_gate_pass=candidate_pass,
        )
        if not baseline_pass:
            failures.append(
                {
                    "case": label,
                    "projection": name,
                    "kind": "baseline_reference_rms",
                    "value": old_rms,
                }
            )
        if not candidate_pass:
            failures.append(
                {
                    "case": label,
                    "projection": name,
                    "kind": "candidate_reference_rms",
                    "value": new_rms,
                    "baseline_reference_gate_pass": baseline_pass,
                }
            )
        record[name] = item
        del reference
    return record


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class QsaThreeKernelBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (9, 0):
            raise unittest.SkipTest("SM90 benchmark target")

    def test_full_prepared_pair_timing_numeric_and_device_evidence(self):
        directory = Path(
            os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
            or tempfile.mkdtemp(prefix="qsa_three_kernel_benchmark_")
        )
        directory.mkdir(parents=True, exist_ok=True)
        artifact = directory / "qsa-three-kernel-benchmark.json"
        failures = []
        seed = 110113
        document = {
            "revision": os.environ.get(
                "QSA_CANDIDATE_REVISION", "working tree under test"
            ),
            "source_sha256": {
                module.__name__: hashlib.sha256(
                    Path(module.__file__).read_bytes()
                ).hexdigest()
                for module in (fp8_linear, qsa_fp8_pair)
            },
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "torch_version": torch.__version__,
            "torch_git_version": torch.version.git_version,
            "cuda_build_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "cpu_torch_threads": torch.get_num_threads(),
            "workspace_environment": {
                name: os.environ.get(name)
                for name in ("CUBLASLT_WORKSPACE_SIZE", "CUBLAS_WORKSPACE_CONFIG")
            },
            "precision": "BF16 input/output, E4M3 weights, independent FP32 weight scales",
            "seed": seed,
            "rows": ROWS,
            "warm_pairs": 8,
            "measured_pairs": 40,
            "timing_units": "microseconds per full projection pair",
            "assert_close": {"rtol": 1.0e-2, "atol": 1.0e-3},
            "reference_rms_gates": {
                "absolute_exclusive": 0.04,
                "relative_to_old": "old_rms * 1.10 + 1e-4",
            },
            "notes": [
                "Actual old project_fp8_pair versus full PreparedFp8Pair.project, including owner checks and output views.",
                "One unified prepared route across tested row counts; no benchmark-only GEMM/padding alternatives.",
                "inference_mode; same per-case input, weights and independent scale domains across A/B.",
                "Eight warm pairs and forty A/B then B/A pairs; profiler-off host/event evidence has no time threshold.",
                "GPU status queries precede warmup and follow measured samples; clocks/power remain unchanged.",
                "CUDA event span retains host submission gaps; kernel duration sums and all copy/memset operations are separate.",
                "Candidate requires exactly three compute kernels and no GPU copy; library workspace memsets are measured and retained.",
                "Preparation, input construction, reference computation and profiler execution are outside timing.",
                "Fixed numeric failures are collected after measurement; baseline reference failures are labeled separately.",
                "This component benchmark does not establish full indexer or end-to-end model latency.",
            ],
            "start_gpu": _gpu_snapshot(),
            "shapes": [],
            "failures": failures,
        }

        def persist():
            artifact.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")

        persist()
        with torch.inference_mode():
            for scenario, scales in (
                ("normal", (0.119713, 0.03125)),
                ("mixed_dynamic_range", (0.0078125, 0.75)),
            ):
                scenario_seed = seed + (0 if scenario == "normal" else 100000)
                torch.manual_seed(scenario_seed)
                projections = [
                    Fp8Linear(
                        (torch.randn(width, 2560, device="cuda") * 0.1).to(
                            torch.float8_e4m3fn
                        ),
                        torch.tensor([scale], dtype=torch.float32, device="cuda"),
                        role="PREFILL",
                    )
                    for width, scale in zip((3072, 640), scales)
                ]
                first, second = projections
                pair = prepare_fp8_pair(first, second)
                self.assertIsNotNone(pair)
                for rows in ROWS:
                    torch.manual_seed(scenario_seed + rows)
                    x = torch.randn(rows, 2560, dtype=torch.bfloat16, device="cuda")
                    if scenario == "mixed_dynamic_range":
                        columns = torch.logspace(-2, 2, 2560, device="cuda")
                        row_gains = torch.tensor(
                            [0.03125, 0.25, 1.0, 8.0, 32.0], device="cuda"
                        )
                        gains = row_gains[
                            torch.arange(rows, device="cuda") % row_gains.numel()
                        ]
                        x = (x.float() * columns * gains[:, None]).to(torch.bfloat16)
                    label = f"{scenario}-t{rows}"
                    functions = {
                        BASELINE: lambda: project_fp8_pair(x, first, second),
                        CANDIDATE: lambda: pair.project(x, first=first, second=second),
                    }
                    telemetry = {}
                    record = {
                        "label": label,
                        "real_rows": rows,
                        "scenario": scenario,
                        "weight_seed": scenario_seed,
                        "input_seed": scenario_seed + rows,
                        "weight_scales": scales,
                        "baseline_padded_m": (rows + 15) // 16 * 16,
                        "candidate_padded_m": (rows + 15) // 16 * 16,
                        "candidate_packed_n": int(pair.weight.shape[0]),
                        "input_shape": list(x.shape),
                        "input_stride": list(x.stride()),
                        "packed_weight_shape": list(pair.weight.shape),
                        "packed_weight_t_stride": list(pair.weight_t.stride()),
                        "input_generation": (
                            "BF16 randn"
                            if scenario == "normal"
                            else "BF16 randn * column logspace(1e-2,1e2) * cyclic row gains [.03125,.25,1,8,32], rounded BF16"
                        ),
                    }
                    document["shapes"].append(record)
                    persist()
                    record["pair_timings"] = _interleaved_timings(
                        functions, telemetry=telemetry
                    )
                    record["timing_telemetry"] = telemetry
                    record["paired_candidate_minus_baseline_us"] = _paired_deltas(
                        record["pair_timings"]
                    )
                    record["pair_profiles"] = {
                        name: _profile_call(
                            fn, directory / f"{label}-{name}.trace.json"
                        )
                        for name, fn in functions.items()
                    }
                    profile = record["pair_profiles"][CANDIDATE]
                    if profile["kernel_count"] != 3 or profile["memcpy_count"]:
                        failures.append(
                            {
                                "case": label,
                                "kind": "candidate_device_count",
                                "kernel_count": profile["kernel_count"],
                                "memcpy_count": profile["memcpy_count"],
                            }
                        )
                    persist()
                    old, new = functions[BASELINE](), functions[CANDIDATE]()
                    record["numeric"] = _numeric_pair(
                        x, old, new, projections, failures, label
                    )
                    del old, new
                    persist()
                    print(
                        json.dumps(
                            {
                                "case": label,
                                "timings": {
                                    name: {
                                        key: value
                                        for key, value in result.items()
                                        if key != "samples"
                                    }
                                    for name, result in record["pair_timings"].items()
                                },
                                "paired_delta_us": record[
                                    "paired_candidate_minus_baseline_us"
                                ],
                                "device_counts": {
                                    name: {
                                        key: value[key]
                                        for key in (
                                            "kernel_count",
                                            "memcpy_count",
                                            "memset_count",
                                            "memset_bytes",
                                        )
                                    }
                                    for name, value in record["pair_profiles"].items()
                                },
                                "failures_so_far": len(failures),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        document["end_gpu"] = _gpu_snapshot()
        persist()
        print(f"projection benchmark artifacts: {artifact}", flush=True)
        self.assertEqual(
            failures,
            [],
            "Fixed numeric/device gates failed; full measurements and raw traces retained",
        )


if __name__ == "__main__":
    unittest.main()
