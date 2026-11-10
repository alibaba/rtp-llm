"""GPU evidence for the net cost of once-per-forward side-table validation.

Runs the real geometry path with twelve distinct layer pools. The two modes
differ only in the explicit validation scope: one dual verdict + async assert
per layer, or one pair per forward. Timing has no pass/fail threshold. Kernel
counts and pool/plan equivalence are correctness assertions.
"""

import collections
import hashlib
import json
import os
import statistics
import tempfile
import time
import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import indexer_prefix_reuse_triton_test as prefix_fixture
import torch

from rtp_llm.models_py.model_desc.qwen4_exp_runtime import (
    qsa_cache_views,
    qsa_geometry,
    qsa_step_plan,
)

LAYERS = 12
WARM_PAIRS = 8
MEASURED_PAIRS = 40


def _distribution(values):
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.9
    lower = int(position)
    return {
        "n": len(values),
        "median": statistics.median(values),
        "p90": ordered[lower]
        + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower])
        * (position - lower),
        "min": min(values),
        "max": max(values),
    }


def _timings(functions):
    names = tuple(functions)
    stream = torch.cuda.current_stream()
    for iteration in range(WARM_PAIRS):
        for name in names if iteration % 2 == 0 else names[::-1]:
            functions[name]()
    stream.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    # Materialize event handles before recording host measurements.
    start.record(stream)
    stop.record(stream)
    stop.synchronize()
    samples = {name: [] for name in names}
    for iteration in range(MEASURED_PAIRS):
        order = names if iteration % 2 == 0 else names[::-1]
        for order_position, name in enumerate(order):
            stream.synchronize()
            start.record(stream)
            begin = time.perf_counter_ns()
            result = functions[name]()
            host_us = (time.perf_counter_ns() - begin) / 1000.0
            stop.record(stream)
            stop.synchronize()
            samples[name].append(
                {
                    "iteration": iteration,
                    "order_position": order_position,
                    "host_enqueue_us": host_us,
                    "cuda_event_span_us": start.elapsed_time(stop) * 1000.0,
                }
            )
            del result
    return {
        name: {
            "host_enqueue_us": _distribution(
                [sample["host_enqueue_us"] for sample in rows]
            ),
            "cuda_event_span_us": _distribution(
                [sample["cuda_event_span_us"] for sample in rows]
            ),
            "samples": rows,
        }
        for name, rows in samples.items()
    }


def _profile(function, path):
    from torch.profiler import ProfilerActivity, profile, record_function

    stream = torch.cuda.current_stream()
    stream.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("qsa.validation_once_benchmark.twelve_layers"):
            result = function()
        stream.synchronize()
    prof.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    kernels = [event for event in events if event.get("cat") == "kernel"]
    device_categories = collections.Counter(
        event.get("cat")
        for event in events
        if event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    )
    return {
        "raw_trace": path.name,
        "returned_layers": len(result),
        "compute_kernel_count": len(kernels),
        "dual_verdict_count": sum(
            "_dual_table_required_blocks_kernel" in event["name"] for event in kernels
        ),
        "async_assert_count": sum(
            "_assert_async_cuda_kernel" in event["name"] for event in kernels
        ),
        "kernel_gpu_duration_sum_us": sum(event.get("dur", 0) for event in kernels),
        "device_operation_counts": dict(device_categories),
        "kernel_names": dict(collections.Counter(event["name"] for event in kernels)),
    }


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class QSAValidationOnceBenchmarkTest(unittest.TestCase):
    def test_twelve_real_layer_geometry_cost_and_kernel_counts(self):
        directory = Path(
            os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
            or tempfile.mkdtemp(prefix="qsa_validation_once_benchmark_")
        )
        directory.mkdir(parents=True, exist_ok=True)
        document = {
            "candidate_revision": os.environ.get(
                "QSA_CANDIDATE_REVISION", "working tree under test"
            ),
            "source_sha256": {
                module.__name__: hashlib.sha256(
                    Path(module.__file__).read_bytes()
                ).hexdigest()
                for module in (qsa_geometry, qsa_step_plan, qsa_cache_views)
            },
            "torch_version": torch.__version__,
            "torch_git_version": torch.version.git_version,
            "cuda_build_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "layers": LAYERS,
            "warm_pairs": WARM_PAIRS,
            "measured_pairs": MEASURED_PAIRS,
            "timing_units": "microseconds per twelve-layer forward",
            "notes": [
                "Real _draft_incremental_prefill_geometry; no mocked verdict or launch.",
                "Twelve different KV/state pool tensor owners with equal capacities.",
                "Shared block-table owners and step required-column vectors.",
                "Both modes use the candidate code; per_layer has no validation scope.",
                "This isolates validation-scope cost, not a full pre-change checkout comparison.",
                "Plan build, tensor allocation/clone setup and JIT excluded by warmup.",
                "Per-layer geometry pool-view/allocation work remains included equally.",
                "Inference-mode metadata remains frozen; every candidate sample starts a new scope.",
                "A/B then B/A alternation; same measured stream synchronized before each sample.",
                "Host interval includes scope entry/exit and all twelve geometry calls, not event record/wait.",
                "CUDA event span includes host submission gaps and is not pure device compute.",
                "Timing samples have no profiler; separate profiles use equal settings and explicit final waits.",
                "No latency threshold; report median/p90 and raw samples even when candidate regresses.",
            ],
            "cases": [],
        }
        artifact = directory / "qsa_validation_once_benchmark.json"
        with torch.inference_mode(), patch.dict(
            os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": "0"}
        ):
            for label, prefixes, lengths in (
                ("b1_t1578", [5120], [1578]),
                ("b2_t1578", [5120, 5120], [1025, 553]),
            ):
                qsa_step_plan.clear_step_plan_cache()
                harness = prefix_fixture.IndexerPrefixReuseTritonTest()
                original, indexer, rope, args = harness.runtime_inputs(
                    prefixes, lengths
                )
                contexts = []
                for _ in range(LAYERS):
                    # Clone before timing/profile: genuinely distinct layer
                    # allocations, same shape/capacity and immutable contents.
                    contexts.append(
                        replace(
                            original,
                            indexer_kv_cache=SimpleNamespace(
                                **{
                                    **vars(original.indexer_kv_cache),
                                    "kv_cache_base": original.indexer_kv_cache.kv_cache_base.clone(),
                                }
                            ),
                            indexer_state_cache=SimpleNamespace(
                                **{
                                    **vars(original.indexer_state_cache),
                                    "kv_cache_base": original.indexer_state_cache.kv_cache_base.clone(),
                                }
                            ),
                        )
                    )
                self.assertEqual(
                    len(
                        {
                            context.indexer_kv_cache.kv_cache_base.data_ptr()
                            for context in contexts
                        }
                    ),
                    LAYERS,
                )
                self.assertEqual(
                    len(
                        {
                            context.indexer_state_cache.kv_cache_base.data_ptr()
                            for context in contexts
                        }
                    ),
                    LAYERS,
                )

                def run(scoped):
                    with (
                        qsa_step_plan.qsa_validation_step() if scoped else nullcontext()
                    ):
                        return tuple(
                            context._draft_incremental_prefill_geometry(
                                indexer=indexer,
                                token_count=sum(lengths),
                                device=args["cu"].device,
                                rope_config=rope,
                                draft=False,
                                draft_prefix_reuse=False,
                            )
                            for context in contexts
                        )

                functions = {
                    "per_layer": lambda: run(False),
                    "once_per_forward": lambda: run(True),
                }
                # Validate the actual fixture wiring and warm the plan before
                # counting kernels; no plan-build kernels belong in this test.
                geometry = run(False)
                for context, value in zip(contexts, geometry):
                    self.assertIs(value.step, geometry[0].step)
                    self.assertIs(value.kv_table, geometry[0].kv_table)
                    self.assertIs(value.state_table, geometry[0].state_table)
                    self.assertEqual(
                        value.kv_pool.data_ptr(),
                        context.indexer_kv_cache.kv_cache_base.data_ptr(),
                    )
                    self.assertEqual(
                        value.state_pool.data_ptr(),
                        context.indexer_state_cache.kv_cache_base.data_ptr(),
                    )
                record = {
                    "name": label,
                    "prefixes": prefixes,
                    "lengths": lengths,
                    "kv_table_shape": list(geometry[0].kv_table.shape),
                    "state_table_shape": list(geometry[0].state_table.shape),
                    "kv_pool_blocks": int(geometry[0].kv_pool.shape[0]),
                    "state_pool_blocks": int(geometry[0].state_pool.shape[0]),
                    "distinct_kv_pool_allocations": LAYERS,
                    "distinct_state_pool_allocations": LAYERS,
                    "timings": _timings(functions),
                    "profiles": {
                        name: _profile(
                            function, directory / f"{label}-{name}.trace.json"
                        )
                        for name, function in functions.items()
                    },
                }
                document["cases"].append(record)
                # Persist timing and full raw profiles before assertions, so
                # unexpected counts remain reviewable in failed-test outputs.
                artifact.write_text(json.dumps(document, indent=2) + "\n")
                for name, pairs in (("per_layer", LAYERS), ("once_per_forward", 1)):
                    profile_result = record["profiles"][name]
                    self.assertEqual(
                        profile_result["compute_kernel_count"],
                        2 * pairs,
                        profile_result,
                    )
                    self.assertEqual(
                        profile_result["dual_verdict_count"], pairs, profile_result
                    )
                    self.assertEqual(
                        profile_result["async_assert_count"], pairs, profile_result
                    )
                # Geometry/validation must not mutate any layer's payload.
                for context in contexts:
                    self.assertTrue(
                        torch.equal(
                            context.indexer_kv_cache.kv_cache_base,
                            original.indexer_kv_cache.kv_cache_base,
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            context.indexer_state_cache.kv_cache_base,
                            original.indexer_state_cache.kv_cache_base,
                        )
                    )
                print(
                    json.dumps(
                        {
                            "case": label,
                            "timings": {
                                name: {
                                    key: value
                                    for key, value in timings.items()
                                    if key != "samples"
                                }
                                for name, timings in record["timings"].items()
                            },
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        print(f"validation benchmark artifacts: {artifact}", flush=True)


if __name__ == "__main__":
    unittest.main()
