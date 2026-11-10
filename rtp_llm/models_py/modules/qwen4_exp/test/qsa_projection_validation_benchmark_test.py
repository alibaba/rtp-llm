"""Manual combined projection + validation component benchmark on SM90.

Every layer executes its real projection and then real incremental geometry.
This measures their combined scheduling/cost, not a sum of standalone medians.
It is not a full indexer or an end-to-end model benchmark: scoring, cache
writers, attention and the transformation of hidden states between layers are
absent. Single-layer controls accompany the twelve-layer chain.
"""

import collections
import hashlib
import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import indexer_prefix_reuse_triton_test as prefix_fixture
import qsa_three_kernel_benchmark_test as benchmark
import torch

from rtp_llm.models_py.model_desc.qwen4_exp_runtime import (
    qsa_cache_views,
    qsa_geometry,
    qsa_step_plan,
)
from rtp_llm.models_py.modules.base.common import fp8_linear, qsa_fp8_pair

LAYERS = 12
WARM_PAIRS = 8
MEASURED_PAIRS = 40
BASELINE = "baseline_five_kernel"
CANDIDATE = "candidate_three_kernel"


def _profile_summary(function, path, *, candidate, layers):
    record = benchmark._profile_call(function, path)
    kernels = [op for op in record["device_operations"] if op["category"] == "kernel"]
    dual = sum("_dual_table_required_blocks_kernel" in op["name"] for op in kernels)
    asserts = sum("_assert_async_cuda_kernel" in op["name"] for op in kernels)
    expected_validation_pairs = 1 if candidate else layers
    expected_projection = (3 if candidate else 5) * layers
    expected_total = expected_projection + 2 * expected_validation_pairs
    record.update(
        {
            "dual_verdict_count": dual,
            "async_assert_count": asserts,
            "other_compute_kernel_count": len(kernels) - dual - asserts,
            "expected_projection_kernels": expected_projection,
            "expected_validation_pairs": expected_validation_pairs,
            "expected_total_compute_kernels": expected_total,
            "kernel_count_matches": len(kernels) == expected_total,
            "kernel_names": dict(collections.Counter(op["name"] for op in kernels)),
            "device_operation_notes": [
                "Memcpy and memset remain in raw trace, operation inventory and timing.",
                "A library workspace memset is not counted as a compute kernel.",
                "The candidate contract expects no GPU memcpy; library workspace memsets remain measured and are recorded separately.",
                "If counts differ, full names/correlation/API owners are retained; no extra kernel is filtered.",
            ],
        }
    )
    return record


def _paired_deltas(timings):
    return benchmark._paired_deltas(timings, baseline=BASELINE, candidate=CANDIDATE)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class QSAProjectionValidationBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (9, 0):
            raise unittest.SkipTest("SM90 prepared projection benchmark")

    def test_combined_projection_then_geometry(self):
        directory = Path(
            os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
            or tempfile.mkdtemp(prefix="qsa_projection_validation_benchmark_")
        )
        directory.mkdir(parents=True, exist_ok=True)
        artifact = directory / "qsa_projection_validation_benchmark.json"
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        document = {
            "candidate_revision": os.environ.get(
                "QSA_CANDIDATE_REVISION", "working tree under test"
            ),
            "source_sha256": {
                module.__name__: hashlib.sha256(
                    Path(module.__file__).read_bytes()
                ).hexdigest()
                for module in (
                    fp8_linear,
                    qsa_fp8_pair,
                    qsa_geometry,
                    qsa_step_plan,
                    qsa_cache_views,
                    benchmark,
                    prefix_fixture,
                )
            },
            "benchmark_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "torch_version": torch.__version__,
            "torch_git_version": torch.version.git_version,
            "cuda_build_version": torch.version.cuda,
            "gpu": properties.name,
            "capability": torch.cuda.get_device_capability(),
            "sm_count": properties.multi_processor_count,
            "device_total_memory_bytes": properties.total_memory,
            "cpu_torch_threads": torch.get_num_threads(),
            "seed": 1101371,
            "warm_pairs": WARM_PAIRS,
            "measured_pairs": MEASURED_PAIRS,
            "timing_units": "microseconds per complete component chain",
            "dtype_contract": "BF16 input/output, E4M3 weights, separate FP32 gate/QK weight scales",
            "candidate_backend": "full PreparedFp8Pair projection with native quantization and rowwise library GEMM",
            "workspace_environment": {
                name: os.environ.get(name)
                for name in ("CUBLASLT_WORKSPACE_SIZE", "CUBLAS_WORKSPACE_CONFIG")
            },
            "native_projection_available": qsa_fp8_pair._native_projection_available(),
            "strict_validation": False,
            "notes": [
                "Manual component evidence: projection then real geometry per layer, all on one stream.",
                "Baseline uses project_fp8_pair and no validation scope; candidate uses prepared native pair and one scope for the whole chain.",
                "Baseline geometry is the candidate's no-scope fallback, not a separate historical checkout.",
                "Twelve distinct gate/QK weights, packed weights, normalization weights and KV/state layer pool allocations.",
                "Same step/table/required vectors/capacities across layers; same inputs and owners across A/B.",
                "One BF16 input is reused across layers to isolate these components; real inter-layer hidden-state transformation is not modeled.",
                "Projection output remains live through its geometry call; only the final layer result is returned.",
                "Inference mode; frozen metadata; every candidate invocation starts a fresh validation scope.",
                "Preparation, weight packing, input/pool setup, plan construction and JIT warmup are outside timing/profile.",
                "Host timing includes scope entry/exit, projection, geometry and loop dispatch; excludes event record/wait.",
                "CUDA event span includes host submission gaps; profiler kernel-duration sum is reported separately.",
                "Forty paired samples alternate A/B and B/A; no latency pass/fail threshold.",
                "Delta statistics use each measured combined chain, never addition of standalone p50 values.",
                "Single-layer controls expose isolated cost; twelve-layer scheduling is not extrapolated to full indexer or end-to-end latency.",
                "Timing is profiler-off; separate profiler records preserve every compute/copy/memset operation.",
                "Library workspace memsets remain included in cost; no zero-memset performance contract.",
                "Read-only GPU snapshots precede warmup and follow samples; first/last ten pairs expose drift.",
                "This test does not replace the projection numerical and cache/undo correctness suites.",
            ],
            "cases": [],
        }
        mismatches = []
        torch.manual_seed(document["seed"])
        with torch.inference_mode(), patch.dict(
            os.environ, {"RTP_LLM_QSA_STRICT_VALIDATION": "0"}
        ):
            projections = []
            for layer in range(LAYERS):
                owners = []
                for width, scale in (
                    (3072, 0.119713 * (1 + layer * 0.01)),
                    (640, 0.03125 * (1 + layer * 0.02)),
                ):
                    weight = (torch.randn(width, 2560, device="cuda") * 0.1).to(
                        torch.float8_e4m3fn
                    )
                    owners.append(
                        fp8_linear.Fp8Linear(
                            weight,
                            torch.tensor([scale], dtype=torch.float32, device="cuda"),
                            role="PREFILL",
                        )
                    )
                pair = fp8_linear.prepare_fp8_pair(*owners)
                self.assertIsNotNone(pair)
                projections.append((owners[0], owners[1], pair))
            self.assertEqual(
                len({first.weight.data_ptr() for first, _, _ in projections}), LAYERS
            )
            self.assertEqual(
                len({second.weight.data_ptr() for _, second, _ in projections}), LAYERS
            )
            self.assertEqual(
                len({pair.weight.data_ptr() for _, _, pair in projections}), LAYERS
            )

            for tokens in (1371, 1576, 8193, 32768):
                qsa_step_plan.clear_step_plan_cache()
                harness = prefix_fixture.IndexerPrefixReuseTritonTest()
                original, original_indexer, rope, args = harness.runtime_inputs(
                    [5120], [tokens]
                )
                x = torch.randn(tokens, 2560, dtype=torch.bfloat16, device="cuda")
                # Query the actual native input layout outside either timing
                # surface instead of encoding an obsolete physical-M policy.
                quantized, row_scales = qsa_fp8_pair.quantize_fp8_prefill_input(x)
                candidate_input_shape = list(quantized.shape)
                candidate_row_scale_shape = list(row_scales.shape)
                del quantized, row_scales
                layers = []
                for first, second, pair in projections:
                    context = replace(
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
                    indexer = SimpleNamespace(
                        **{
                            **vars(original_indexer),
                            "q_norm_gamma": original_indexer.q_norm_gamma.clone(),
                            "k_norm_gamma": original_indexer.k_norm_gamma.clone(),
                        }
                    )
                    layers.append((first, second, pair, context, indexer))
                self.assertEqual(
                    len(
                        {
                            layer[3].indexer_kv_cache.kv_cache_base.data_ptr()
                            for layer in layers
                        }
                    ),
                    LAYERS,
                )
                self.assertEqual(
                    len(
                        {
                            layer[3].indexer_state_cache.kv_cache_base.data_ptr()
                            for layer in layers
                        }
                    ),
                    LAYERS,
                )

                def geometry(context, indexer, token_count, device):
                    return context._draft_incremental_prefill_geometry(
                        indexer=indexer,
                        token_count=token_count,
                        device=device,
                        rope_config=rope,
                        draft=False,
                        draft_prefix_reuse=False,
                    )

                # Build and verify the shared step outside both measurement
                # surfaces; distinct pools must still bind to their own layer.
                prepared = [
                    geometry(layer[3], layer[4], tokens, x.device) for layer in layers
                ]
                for layer, value in zip(layers, prepared):
                    self.assertIs(value.step, prepared[0].step)
                    self.assertIs(value.kv_table, prepared[0].kv_table)
                    self.assertIs(value.state_table, prepared[0].state_table)
                    self.assertEqual(
                        value.kv_pool.data_ptr(),
                        layer[3].indexer_kv_cache.kv_cache_base.data_ptr(),
                    )
                    self.assertEqual(
                        value.state_pool.data_ptr(),
                        layer[3].indexer_state_cache.kv_cache_base.data_ptr(),
                    )

                for layer_count in (1, LAYERS):
                    active = layers[:layer_count]

                    def run(candidate):
                        result = None
                        with (
                            qsa_step_plan.qsa_validation_step()
                            if candidate
                            else nullcontext()
                        ):
                            for first, second, pair, context, indexer in active:
                                projected = (
                                    pair.project(x, first=first, second=second)
                                    if candidate
                                    else fp8_linear.project_fp8_pair(x, first, second)
                                )
                                if projected is None:
                                    raise AssertionError(
                                        "combined component projection declined"
                                    )
                                # Derive the real geometry's shape/device from
                                # the just-produced QK output, in model order.
                                plan = geometry(
                                    context,
                                    indexer,
                                    int(projected[1].shape[0]),
                                    projected[1].device,
                                )
                                result = (projected, plan)
                        return result

                    functions = {
                        BASELINE: lambda: run(False),
                        CANDIDATE: lambda: run(True),
                    }
                    timing_telemetry = {}
                    timings = benchmark._interleaved_timings(
                        functions,
                        warm_pairs=WARM_PAIRS,
                        measured_pairs=MEASURED_PAIRS,
                        telemetry=timing_telemetry,
                    )
                    record = {
                        "tokens": tokens,
                        "batch": 1,
                        "prefix": 5120,
                        "layers": layer_count,
                        "baseline_padded_m": (tokens + 15) // 16 * 16,
                        "candidate_padded_m": candidate_input_shape[0],
                        "candidate_quantized_input_shape": candidate_input_shape,
                        "candidate_row_scale_shape": candidate_row_scale_shape,
                        "candidate_packed_n": int(active[0][2].weight.shape[0]),
                        "visible_gate_qk_widths": [3072, 640],
                        "input_shape": list(x.shape),
                        "input_stride": list(x.stride()),
                        "weight_shapes": [
                            list(active[0][0].weight.shape),
                            list(active[0][1].weight.shape),
                        ],
                        "packed_weight_shape": list(active[0][2].weight.shape),
                        "packed_weight_stride": list(active[0][2].weight.stride()),
                        "kv_pool_shape": list(prepared[0].kv_pool.shape),
                        "state_pool_shape": list(prepared[0].state_pool.shape),
                        "heads": original_indexer.n_heads,
                        "head_dim": original_indexer.head_dim,
                        "compress_ratio": original_indexer.compress_ratio,
                        "kv_table_shape": list(prepared[0].kv_table.shape),
                        "state_table_shape": list(prepared[0].state_table.shape),
                        "kv_pool_blocks": int(prepared[0].kv_pool.shape[0]),
                        "state_pool_blocks": int(prepared[0].state_pool.shape[0]),
                        "distinct_projection_and_pool_owners": layer_count,
                        "timings": timings,
                        "timing_telemetry": timing_telemetry,
                        "paired_candidate_minus_baseline_us": _paired_deltas(timings),
                        "profiles": {
                            name: _profile_summary(
                                function,
                                directory
                                / f"t{tokens}-layers{layer_count}-{name}.trace.json",
                                candidate=name == CANDIDATE,
                                layers=layer_count,
                            )
                            for name, function in functions.items()
                        },
                    }
                    document["cases"].append(record)
                    artifact.write_text(json.dumps(document, indent=2) + "\n")
                    for name, profile_result in record["profiles"].items():
                        expected_pairs = 1 if name == CANDIDATE else layer_count
                        if (
                            not profile_result["kernel_count_matches"]
                            or profile_result["dual_verdict_count"] != expected_pairs
                            or profile_result["async_assert_count"] != expected_pairs
                            or (name == CANDIDATE and profile_result["memcpy_count"])
                        ):
                            mismatches.append(
                                {
                                    "tokens": tokens,
                                    "layers": layer_count,
                                    "mode": name,
                                    "profile": profile_result,
                                }
                            )
                    print(
                        json.dumps(
                            {
                                "tokens": tokens,
                                "layers": layer_count,
                                "timings": {
                                    name: {
                                        key: value
                                        for key, value in timing.items()
                                        if key != "samples"
                                    }
                                    for name, timing in timings.items()
                                },
                                "paired_candidate_minus_baseline_us": record[
                                    "paired_candidate_minus_baseline_us"
                                ],
                                "compute_kernel_counts": {
                                    name: profile_result["kernel_count"]
                                    for name, profile_result in record[
                                        "profiles"
                                    ].items()
                                },
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        document["device_operation_count_mismatches"] = mismatches
        artifact.write_text(json.dumps(document, indent=2) + "\n")
        print(f"combined component benchmark artifacts: {artifact}", flush=True)
        self.assertEqual(
            mismatches,
            [],
            "Unexpected device operation counts; full inventories and raw traces retained in artifacts",
        )


if __name__ == "__main__":
    unittest.main()
