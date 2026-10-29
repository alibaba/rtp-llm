"""Three-arm PLE page64/192 component benchmark; correctness gates precede timing."""

import contextlib
import statistics
import time
import unittest

import torch
from ple_native_quant_adapter import native_projection_adapter, provenance
from ple_sm90_pages_test import capture_scaled_mm, fixture, make_layer, reference_pages
from ple_sm90_test_utils import capture_kernel, exact, metadata, save, sha

from rtp_llm.models_py.modules.qwen4_exp.ple import ple as owner
from rtp_llm.models_py.modules.qwen4_exp.ple import ple_conv_triton as conv
from rtp_llm.models_py.modules.qwen4_exp.ple import ple_gate_triton as gate
from rtp_llm.models_py.modules.qwen4_exp.ple import ple_page_projection as projection


@contextlib.contextmanager
def route(arm):
    old_owner, old_gate = owner._sm_supported, gate._sm_supported
    old_project = projection.project_pages_pair
    try:
        if arm == "A":
            owner._sm_supported = lambda device: False
            gate._sm_supported = lambda device: False
        elif arm == "B":
            projection.project_pages_pair = lambda *args, **kwargs: None
        elif arm != "C":
            raise ValueError(arm)
        yield
    finally:
        owner._sm_supported, gate._sm_supported = old_owner, old_gate
        projection.project_pages_pair = old_project


def sample(call):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    initial = torch.cuda.memory_allocated()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    clock = time.perf_counter()
    start.record()
    value = call()
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter() - clock) * 1000
    result = {
        "cuda_ms": start.elapsed_time(end),
        "wall_ms": wall_ms,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "incremental_peak_bytes": torch.cuda.max_memory_allocated() - initial,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    del value
    return result


def profile(call):
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as p:
        value = call()
        torch.cuda.synchronize()
    kernels = [
        {"name": e.name, "duration_us": e.device_time_total}
        for e in p.events()
        if str(e.device_type) == "DeviceType.CUDA"
    ]
    del value
    if not kernels:
        raise RuntimeError("No CUDA events captured; profile cannot be marked PASS")
    return {
        "kernels": kernels,
        "kernel_count": len(kernels),
        "kernel_sum_us": sum(x["duration_us"] for x in kernels),
    }


def summarize(samples):
    result = {}
    for metric in ("cuda_ms", "wall_ms"):
        values = sorted(x[metric] for x in samples)
        result[metric] = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p90": values[int(0.9 * (len(values) - 1))],
            "samples": len(values),
        }
    result["incremental_peak_bytes"] = max(x["incremental_peak_bytes"] for x in samples)
    return result


class PleSM90Benchmark(unittest.TestCase):
    def test_three_arm(self):
        record = {
            "metadata": metadata(),
            "native": provenance(),
            "status": "RUNNING",
            "cases": [],
            "arms": {
                "A": "production owner forced to old per-page route, gate fallback",
                "B": "production owner with per-page projections, whole gate/conv",
                "C": "production owner with two batched GEMMs and whole gate/conv",
            },
            "order": "3 rounds of A B C C B A; each step 5 samples; 30 per arm",
            "scope": "synthetic PLE component, output plus all snapshots; not engine/end-to-end",
            "source": {
                "owner": sha(owner.__file__),
                "gate": sha(gate.__file__),
                "conv": sha(conv.__file__),
                "projection": sha(projection.__file__),
            },
        }
        try:
            with native_projection_adapter():
                layer = make_layer(81000, quantized_weights=True)
                for rows, page in [(s, 64) for s in (5120, 10240, 15360, 20480)] + [
                    (5120, 192)
                ]:
                    hidden, emb, history, initial = fixture(
                        rows, page, 82000 + rows + page
                    )

                    def call():
                        return layer.prefill_page_snapshots(
                            hidden,
                            history,
                            page,
                            initial,
                            embeddings=emb,
                            add_residual=True,
                        )

                    reference, states = reference_pages(
                        layer, hidden, emb, page, initial, True
                    )
                    item = {
                        "rows": rows,
                        "page": page,
                        "seed": 82000 + rows + page,
                        "arms": {},
                    }
                    record["cases"].append(item)
                    for arm in "ABC":
                        with route(arm), capture_scaled_mm() as mm, capture_kernel(
                            gate, "_ple_gate"
                        ) as gc, capture_kernel(conv, "_ple_conv") as cc:
                            output, snapshots = call()
                        checks = {
                            "output": exact(output, reference),
                            "snapshots": [
                                exact(a, b) for a, b in zip(snapshots, states)
                            ],
                        }
                        item["arms"][arm] = {
                            "gate": checks,
                            "gemm_calls": mm,
                            "gate_calls": gc,
                            "conv_calls": cc,
                            "samples": [],
                        }
                        self.assertEqual(len(snapshots), len(states))
                        self.assertTrue(
                            checks["output"]["pass"]
                            and all(x["pass"] for x in checks["snapshots"]),
                            checks,
                        )
                        self.assertEqual(
                            len(mm),
                            2 if arm == "C" else 2 * ((rows + page - 1) // page),
                        )
                        self.assertEqual(len(gc), 0 if arm == "A" else 1)
                        self.assertEqual(len(cc), 0 if arm == "A" else 1)
                        del output, snapshots
                    del reference, states
                    for arm in "ABC":
                        with route(arm):
                            for _ in range(3):
                                result = call()
                                torch.cuda.synchronize()
                                del result
                    for round_id in range(3):
                        for step, arm in enumerate("ABCCBA"):
                            with route(arm):
                                for i in range(5):
                                    value = sample(call)
                                    value.update(round=round_id, step=step, sample=i)
                                    item["arms"][arm]["samples"].append(value)
                        save("ple_three_arm_benchmark", record)
                    for arm in "ABC":
                        item["arms"][arm]["summary"] = summarize(
                            item["arms"][arm]["samples"]
                        )
                        with route(arm):
                            item["arms"][arm]["profile"] = profile(call)
                    item["status"] = "PASS"
                    save("ple_three_arm_benchmark", record)
            record["status"] = "PASS"
        except Exception as exc:
            record.update(status="FAIL", error=repr(exc))
            raise
        finally:
            save("ple_three_arm_benchmark", record)


if __name__ == "__main__":
    unittest.main()
