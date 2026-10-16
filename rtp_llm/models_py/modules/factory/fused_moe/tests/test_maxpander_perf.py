"""Rank-local routing evidence, not OFF-model throughput or prefill speedup."""

import json
import os
import statistics
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import torch
import triton

from rtp_llm.eplb.maxpander_plan import build_circulant_plan
from rtp_llm.models_py.modules.factory.fused_moe.tests.maxpander_reference import (
    LegacyFullWF,
)
from rtp_llm.models_py.modules.maxpander import MaxPanderRouter

WARMUP_ITERATIONS = 30
MEASURE_ITERATIONS = 100
PROFILE_ITERATIONS = 5
WF_KERNEL_NAMES = (
    "_wf_count_kernel",
    "_wf_hot_order_kernel",
    "_wf_quota_kernel",
    "_wf_scatter_kernel",
)
WF_WORKSPACE_NAMES = ("counts", "hot_order", "quota", "alloc", "rank_counter")


def _workspace_bytes(router):
    return sum(
        workspace.numel() * workspace.element_size()
        for name in (*WF_WORKSPACE_NAMES, "refine_order", "work_counts")
        if (workspace := getattr(router, name, None)) is not None
    )


def measure_stream_latency(function):
    for _ in range(WARMUP_ITERATIONS):
        function()
    torch.cuda.synchronize()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(MEASURE_ITERATIONS)
    ]
    for start, end in events:
        start.record()
        end.record()
    torch.cuda.synchronize()
    allocations_before = torch.cuda.memory_stats()["allocation.all.allocated"]
    for start, end in events:
        start.record()
        function()
        end.record()
    torch.cuda.synchronize()
    allocations_after = torch.cuda.memory_stats()["allocation.all.allocated"]
    samples = sorted(start.elapsed_time(end) * 1000 for start, end in events)
    allocator_calls = allocations_after - allocations_before
    return {
        "timing": {
            "metric": "stream_latency_us",
            "source": "CUDA events; includes ID copy and host launch/stream idle gaps",
            "median_us": statistics.median(samples),
            "p90_us": samples[89],
            "min_us": samples[0],
            "max_us": samples[-1],
            "warmup_iterations": WARMUP_ITERATIONS,
            "measured_iterations": MEASURE_ITERATIONS,
        },
        "allocations": {
            "source": "torch.cuda.memory_stats: allocation.all.allocated",
            "window": "unprofiled measured iterations only; workspace/setup excluded",
            "torch_allocator_calls": allocator_calls,
            "torch_allocator_calls_per_iter": allocator_calls / MEASURE_ITERATIONS,
        },
    }


def _activity_summary(events):
    return {
        "count_per_iter": len(events) / PROFILE_ITERATIONS,
        "sum_us_per_iter": sum(event["dur"] for event in events) / PROFILE_ITERATIONS,
        "by_name": [
            {
                "name": name,
                "count_per_iter": sum(event["name"] == name for event in events)
                / PROFILE_ITERATIONS,
                "sum_us_per_iter": sum(
                    event["dur"] for event in events if event["name"] == name
                )
                / PROFILE_ITERATIONS,
            }
            for name in sorted({event["name"] for event in events})
        ],
    }


def _cpu_profile_scopes(events, labels):
    return [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") == "user_annotation"
        and event.get("name") in labels
    ]


class MaxPanderPerfTest(unittest.TestCase):
    def test_profile_scopes_exclude_duplicate_gpu_annotations(self):
        label = "maxpander/baseline/0"
        cpu_scope = {"ph": "X", "cat": "user_annotation", "name": label}
        events = [
            cpu_scope,
            {"ph": "X", "cat": "gpu_user_annotation", "name": label},
            {"ph": "X", "cat": "user_annotation", "name": "other"},
            {"ph": "M", "cat": "user_annotation", "name": label},
        ]
        self.assertEqual(_cpu_profile_scopes(events, {label: "baseline"}), [cpu_scope])

    def _summarize_profile(self, iterations, path):
        expected = {
            "full_wf": WF_KERNEL_NAMES,
            "fused_wf": (
                "_wf_count_order_kernel",
                "_wf_quota_kernel",
                "_wf_scatter_kernel",
            ),
            "direct": ("_direct_remap_kernel",),
            "copy_only": (),
        }[path]
        kernels = []
        copies = []
        memsets = []
        launch_counts = []
        kernel_sums = []
        self.assertEqual(len(iterations), PROFILE_ITERATIONS)
        for activities in iterations:
            iteration_kernels = [
                event for event in activities if event["cat"] == "kernel"
            ]
            iteration_copies = [
                event for event in activities if event["cat"] == "gpu_memcpy"
            ]
            iteration_memsets = [
                event for event in activities if event["cat"] == "gpu_memset"
            ]
            self.assertEqual(len(iteration_copies), 1, activities)
            self.assertIn("DtoD", iteration_copies[0]["name"])
            matched = Counter()
            reset_kernels = []
            for event in iteration_kernels:
                self.assertGreater(event["dur"], 0, event)
                matches = [name for name in expected if name in event["name"]]
                if path == "full_wf" and "FillFunctor" in event["name"]:
                    reset_kernels.append(event)
                else:
                    self.assertEqual(len(matches), 1, event)
                    matched.update(matches)
            self.assertEqual(matched, Counter(expected))
            self.assertEqual(
                len(reset_kernels) + len(iteration_memsets), int(path == "full_wf")
            )
            kernel_sum = sum(event["dur"] for event in iteration_kernels)
            if expected:
                self.assertGreater(kernel_sum, 0, "Missing CUDA kernel durations")
            kernels.extend(iteration_kernels)
            copies.extend(iteration_copies)
            memsets.extend(iteration_memsets)
            launch_counts.append(len(iteration_kernels))
            kernel_sums.append(kernel_sum)
        self.assertEqual(len(set(launch_counts)), 1, launch_counts)
        summary = _activity_summary(kernels)
        return {
            "source": (
                "Torch profiler/CUPTI CUDA device activities, not CPU operator time"
            ),
            "profiled_iterations": PROFILE_ITERATIONS,
            "iteration_boundary": (
                "record_function enclosing copy/remap and CUDA synchronize"
            ),
            "kernel_sum_us_per_iter": summary["sum_us_per_iter"],
            "kernel_launch_count_per_iter": summary["count_per_iter"],
            "per_iteration_kernel_launch_counts": launch_counts,
            "per_iteration_kernel_sums_us": kernel_sums,
            "kernels_by_name": summary["by_name"],
            "expected_routing_kernel_names": list(expected),
            "expected_reset_operations_per_iter": int(path == "full_wf"),
            "reset_contract": (
                "counts.zero_: one FillFunctor kernel or one CUDA memset"
                if path == "full_wf"
                else "no separate counts reset"
            ),
            "names_and_counts_match": True,
            "id_copy_excluded_from_kernel_sum": _activity_summary(copies),
            "memset_excluded_from_kernel_sum": _activity_summary(memsets),
        }

    def _profile_variants(self, variants, paths, trace_path):
        torch.cuda.synchronize()
        labels = {}
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            for iteration in range(PROFILE_ITERATIONS):
                for name, function in variants.items():
                    label = f"maxpander/{name}/{iteration}"
                    labels[label] = name
                    with torch.profiler.record_function(label):
                        function()
                        torch.cuda.synchronize()
        profiler.export_chrome_trace(str(trace_path))
        with trace_path.open(encoding="utf-8") as trace_file:
            events = json.load(trace_file)["traceEvents"]
        scopes = _cpu_profile_scopes(events, labels)
        self.assertEqual(
            Counter(event["name"] for event in scopes), Counter(labels.keys())
        )
        device_events = [
            event
            for event in events
            if event.get("ph") == "X"
            and event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
        ]
        self.assertTrue(device_events, "CUDA profiler unavailable or trace is empty")
        grouped = {name: [] for name in variants}
        assigned = 0
        for scope in scopes:
            activities = [
                event
                for event in device_events
                if scope["ts"] <= event["ts"]
                and event["ts"] + event["dur"] <= scope["ts"] + scope["dur"]
            ]
            assigned += len(activities)
            grouped[labels[scope["name"]]].append(activities)
        self.assertEqual(assigned, len(device_events), "Unattributed CUDA activity")
        return {
            name: self._summarize_profile(iterations, paths[name])
            for name, iterations in grouped.items()
        }

    def _benchmark_shape(self, router, legacy, physical_to_logical, shape, trace_dir):
        logical = (
            torch.arange(shape["tokens"] * shape["topk"], device="cuda")
            .view(shape["tokens"], shape["topk"])
            .remainder(
                16 if shape["distribution"] == "sparse" else shape["logical_experts"]
            )
        )
        baseline_ids = torch.empty_like(logical)
        candidate_ids = torch.empty_like(logical)

        def copy_only():
            candidate_ids.copy_(logical)

        def baseline():
            baseline_ids.copy_(logical)
            return legacy.remap_inplace(baseline_ids)

        def candidate():
            candidate_ids.copy_(logical)
            return router.remap_inplace(candidate_ids)

        self.assertIs(baseline(), baseline_ids)
        self.assertIs(candidate(), candidate_ids)
        for physical_ids in (baseline_ids, candidate_ids):
            self.assertTrue(torch.all(physical_ids >= 0).item())
            self.assertTrue(torch.all(physical_ids < shape["physical_experts"]).item())
            self.assertTrue(torch.equal(physical_to_logical[physical_ids], logical))
        if shape["redundant_experts"] == 0:
            self.assertTrue(torch.equal(baseline_ids, candidate_ids))
            self.assertTrue(
                torch.equal(candidate_ids, legacy.cand_phys[logical, 0].long())
            )
            self.assertFalse(torch.equal(candidate_ids, logical))
            self.assertEqual(_workspace_bytes(router), 0)
            equivalence = (
                "exact physical IDs vs legacy full-WF and placement lookup; "
                "nonidentity remap"
            )
        else:
            for name in WF_WORKSPACE_NAMES:
                expected = getattr(legacy, name)
                if name == "rank_counter":
                    split = (router.quota[:, 0] > 0) & (router.quota[:, 1] > 0)
                    expected = torch.where(split, expected, torch.zeros_like(expected))
                self.assertTrue(torch.equal(getattr(router, name), expected))
            self.assertTrue(
                torch.equal(
                    torch.bincount(
                        baseline_ids.flatten(), minlength=shape["physical_experts"]
                    ),
                    torch.bincount(
                        candidate_ids.flatten(), minlength=shape["physical_experts"]
                    ),
                )
            )
            equivalence = (
                "exact WF workspace and physical histograms; "
                "per-ID logical equivalence; atomic replica order may differ"
            )
        copy_only()
        self.assertTrue(torch.equal(candidate_ids, logical))
        variants = {
            "baseline": baseline,
            "candidate": candidate,
            "copy_only_reference": copy_only,
        }
        candidate_path = "full_wf"
        if shape["redundant_experts"] == 0:
            candidate_path = "direct"
        elif logical.numel() <= 1024 and shape["logical_experts"] <= 512:
            candidate_path = "fused_wf"
        paths = {
            "baseline": "full_wf",
            "candidate": candidate_path,
            "copy_only_reference": "copy_only",
        }
        measurements = {}
        for name, function in variants.items():
            measurements[name] = measure_stream_latency(function)
            self.assertEqual(
                measurements[name]["allocations"]["torch_allocator_calls"], 0, name
            )
            measurements[name]["path"] = paths[name]
        case_name = (
            f'{shape["model"].replace(".", "_")}_e{shape["logical_experts"]}'
            f'_r{shape["redundant_experts"]}_ep{shape["ep_size"]}'
            f'_{shape["distribution"]}_tokens{shape["tokens"]}'
        )
        trace_path = trace_dir / f"{case_name}.trace.json"
        profiles = self._profile_variants(variants, paths, trace_path)
        for name, measurement in measurements.items():
            measurement["cuda_profile"] = profiles[name]
        measurements["baseline"]["wf_workspace_bytes"] = _workspace_bytes(legacy)
        measurements["candidate"]["wf_workspace_bytes"] = _workspace_bytes(router)
        return {
            "shape": shape,
            "correctness": {"checked_before_timing": True, "equivalence": equivalence},
            "trace_path": str(trace_path),
            "comparison": (
                "same_shape_legacy_full_wf_vs_direct"
                if shape["redundant_experts"] == 0
                else "frozen_full_wf_vs_active_expert_wf"
            ),
            **measurements,
            "ratios": {
                "routing_kernel_sum_baseline_over_candidate": profiles["baseline"][
                    "kernel_sum_us_per_iter"
                ]
                / profiles["candidate"]["kernel_sum_us_per_iter"],
                "copy_plus_routing_stream_median_baseline_over_candidate": measurements[
                    "baseline"
                ]["timing"]["median_us"]
                / measurements["candidate"]["timing"]["median_us"],
            },
        }

    def test_rank_local_prefill_routing_cost(self):
        self.assertTrue(torch.cuda.is_available(), "A CUDA GPU is required, not a skip")
        self.assertTrue(
            torch.profiler.kineto_available(), "Kineto profiling is required"
        )
        self.assertIn(
            torch.profiler.ProfilerActivity.CUDA,
            torch.profiler.supported_activities(),
            "CUPTI CUDA activities are required for pure kernel-time evidence",
        )
        self.assertEqual(
            torch.cuda.memory.get_allocator_backend(),
            "native",
            "Allocator-call evidence requires native caching allocator counters",
        )
        output_root = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        self.assertTrue(
            output_root, "TEST_UNDECLARED_OUTPUTS_DIR is required for traces"
        )
        trace_dir = Path(tempfile.mkdtemp(prefix="maxpander_routing_", dir=output_root))
        records = []
        for model, num_logical, redundant, ep_size in (
            ("V4", 256, 0, 2),
            ("V4", 256, 64, 2),
            ("V4.1", 384, 0, 2),
            ("V4.1", 384, 96, 2),
            ("V4", 256, 64, 4),
            ("V4.1", 384, 96, 4),
        ):
            num_physical = num_logical + redundant
            placement, candidates, replicas = build_circulant_plan(
                num_logical, num_physical, ep_size
            )
            log2phy = torch.tensor(candidates, device="cuda", dtype=torch.int32)
            replica_counts = torch.tensor(replicas, device="cuda", dtype=torch.int32)
            arguments = dict(
                num_logical=num_logical,
                num_physical=num_physical,
                ep_size=ep_size,
                wf_passes=8,
            )
            router = MaxPanderRouter(log2phy, replica_counts, **arguments)
            legacy = LegacyFullWF(log2phy, replica_counts, **arguments)
            physical_to_logical = torch.tensor(placement, device="cuda")
            distributions = ("dense", "sparse") if redundant else ("dense",)
            for distribution in distributions:
                for tokens in (1, 17, 32, 257, 512, 1024, 2048, 8192):
                    shape = dict(
                        model=model,
                        logical_experts=num_logical,
                        redundant_experts=redundant,
                        physical_experts=num_physical,
                        ep_size=ep_size,
                        topk=6,
                        tokens=tokens,
                        distribution=distribution,
                        id_dtype="int64",
                        row_stride=6,
                        wf_passes=8,
                    )
                    with self.subTest(**shape):
                        records.append(
                            self._benchmark_shape(
                                router, legacy, physical_to_logical, shape, trace_dir
                            )
                        )
        self.assertEqual(len(records), 80, "Incomplete evidence matrix")
        report_path = trace_dir / "report.json"
        report = dict(
            benchmark="maxpander_active_expert_routing_evidence",
            evidence="RTP_MEASURED",
            device=torch.cuda.get_device_name(),
            compute_capability=list(torch.cuda.get_device_capability()),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            triton_version=triton.__version__,
            allocator_backend=torch.cuda.memory.get_allocator_backend(),
            torch_compile=False,
            compilation_and_workspace_allocation="outside all measured regions",
            workload=(
                "synthetic dense and 16-active-expert cyclic IDs; EP2/EP4; "
                "single rank, no EP collective"
            ),
            placement="build_circulant_plan, default seed 0, including nonidentity R0",
            baseline=(
                "test-private original count/hot-order/quota/scatter full-WF, "
                "not OFF-model throughput"
            ),
            candidate=(
                "production remap_inplace: R0 direct mapping; R>0 stable active-expert WF"
            ),
            copy_only_reference=(
                "ID reset copy only, independent reference, "
                "not model OFF or a routing baseline"
            ),
            timing=(
                "CUDA-event stream latency includes host launch gaps; profiler "
                "kernel sums exclude ID copy and separately report memsets"
            ),
            scope=(
                "routing-local evidence only; "
                "no whole-model prefill or serving throughput claim"
            ),
            report_path=str(report_path),
            records=records,
        )
        with report_path.open("x", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2)
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    unittest.main()
