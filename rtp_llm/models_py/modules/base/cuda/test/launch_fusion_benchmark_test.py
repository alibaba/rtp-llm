import json
import os
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import torch
from flash_mla import flash_mla_sparse_fwd

from rtp_llm.models_py.modules.base.cuda.indexer_launch_fusion import try_write_topk
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.cp_topk_offsets_triton import (
    offset_cp_topk,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.rope_emb_new import (
    NewMlaRotaryEmbeddingOp,
)

Operation = Callable[[], tuple[torch.Tensor, ...]]


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median": statistics.median(values),
        "p10": ordered[9],
        "p90": ordered[89],
        "min": ordered[0],
        "max": ordered[-1],
    }


def measure_pair(
    baseline: Operation,
    candidate: Operation,
    setup: Callable[[], None] | None = None,
) -> tuple[dict, dict]:
    operations = (baseline, candidate)
    for _ in range(30):
        for operation in operations:
            if setup is not None:
                setup()
            operation()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    gpu_us, enqueue_us, wall_us = ([], []), ([], []), ([], [])
    for sample in range(100):
        for side in (0, 1) if sample % 2 == 0 else (1, 0):
            if setup is not None:
                setup()
                torch.cuda.synchronize()
            start.record()
            t0 = time.perf_counter_ns()
            result = operations[side]()
            t1 = time.perf_counter_ns()
            stop.record()
            stop.synchronize()
            t2 = time.perf_counter_ns()
            gpu_us[side].append(start.elapsed_time(stop) * 1000)
            enqueue_us[side].append((t1 - t0) / 1000)
            wall_us[side].append((t2 - t0) / 1000)
            del result
    results = []
    for side, operation in enumerate(operations):
        if setup is not None:
            setup()
            torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            operation()
            torch.cuda.synchronize()
        events = [
            e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
        ]
        kernels = [
            e for e in events if not e.name.lower().startswith(("memcpy", "memset"))
        ]
        results.append(
            {
                "cuda_event_stream_latency_us": distribution(gpu_us[side]),
                "host_enqueue_us": distribution(enqueue_us[side]),
                "synchronized_wall_us": distribution(wall_us[side]),
                "kernel_sum_us_per_iter": sum(
                    e.time_range.elapsed_us() for e in kernels
                ),
                "kernel_launches": len(kernels),
                "kernel_names": [e.name for e in kernels],
                "copy_or_memset_events": [e.name for e in events if e not in kernels],
            }
        )
    return results[0], results[1]


class LaunchFusionBenchmarkTest(TestCase):
    def assert_bits(
        self, got: tuple[torch.Tensor, ...], want: tuple[torch.Tensor, ...]
    ) -> None:
        self.assertEqual(len(got), len(want))
        for actual, expected in zip(got, want):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertTrue(
                torch.equal(
                    actual.contiguous().view(torch.uint8),
                    expected.contiguous().view(torch.uint8),
                )
            )

    def fixtures(self) -> list[tuple[str, str, Operation, Operation]]:
        indices = torch.randint(
            -1, 100000, (1247, 2048), device="cuda", dtype=torch.int32
        )
        offsets = torch.arange(1247, device="cuda", dtype=torch.int32) * 100001
        baseline_out = torch.empty((1249, 2052), device="cuda", dtype=torch.int32)[
            1:-1, 2:-2
        ]
        candidate_out = torch.empty((1249, 2052), device="cuda", dtype=torch.int32)[
            1:-1, 2:-2
        ]

        def sort_before() -> tuple[torch.Tensor, ...]:
            sentinel = torch.iinfo(indices.dtype).max
            sortable = torch.where(
                indices >= 0, indices, torch.full_like(indices, sentinel)
            )
            ordered = torch.sort(sortable, dim=-1).values
            ordered = torch.where(
                ordered == sentinel, torch.full_like(ordered, -1), ordered
            )
            baseline_out.copy_(
                torch.where(ordered >= 0, ordered + offsets[:, None], ordered)
            )
            return (baseline_out,)

        def sort_after() -> tuple[torch.Tensor, ...]:
            self.assertTrue(try_write_topk(indices, offsets, candidate_out, True))
            return (candidate_out,)

        query = torch.randn((1252, 64, 256), device="cuda", dtype=torch.bfloat16)
        angles = torch.randn((1252, 32), device="cuda")
        rope = NewMlaRotaryEmbeddingOp(
            torch.cat((angles.cos(), angles.sin()), dim=-1), False
        )
        q_before, q_after = query.clone(), query.clone()
        k_before = torch.randn((1252, 576), device="cuda", dtype=torch.bfloat16)
        k_after = k_before.clone()
        k_initial = k_before.clone()

        def reset_rope() -> None:
            q_before.copy_(query)
            q_after.copy_(query)
            k_before.copy_(k_initial)
            k_after.copy_(k_initial)

        self.reset_rope = reset_rope
        positions = SimpleNamespace(
            positions_d=torch.arange(1252, device="cuda", dtype=torch.int32)
        )

        def rope_before() -> tuple[torch.Tensor, ...]:
            rope.forward(q_before[..., 192:], k_before[:, 512:], positions)
            return q_before, k_before[:, 512:].contiguous()

        def rope_after() -> tuple[torch.Tensor, ...]:
            contiguous_k = torch.empty((1252, 64), device="cuda", dtype=query.dtype)
            rope.forward(
                q_after[..., 192:], k_after[:, 512:], positions, key_output=contiguous_k
            )
            return q_after, contiguous_k

        raw = torch.randn((99840, 1, 576), device="cuda", dtype=torch.bfloat16) * 0.1
        mapping = torch.arange(99611, device="cuda", dtype=torch.int64) * 99840 // 99611
        history_q = (
            torch.randn((1247, 64, 576), device="cuda", dtype=torch.bfloat16) * 0.1
        )
        history_topk = torch.randint(
            0, 99611, (1247, 2048), device="cuda", dtype=torch.int32
        )
        history_topk[:, -7:] = -1
        history_offsets = torch.zeros(1247, device="cuda", dtype=torch.int32)

        def history_before() -> tuple[torch.Tensor, ...]:
            restored = torch.index_select(raw, 0, mapping)
            logical = offset_cp_topk(history_topk, history_offsets)
            return flash_mla_sparse_fwd(
                history_q, restored, logical[:, None], 576**-0.5
            )

        def history_after() -> tuple[torch.Tensor, ...]:
            physical = offset_cp_topk(
                history_topk, history_offsets, mapping, raw.shape[0]
            )
            return flash_mla_sparse_fwd(history_q, raw, physical[:, None], 576**-0.5)

        return [
            (
                "topk_sort",
                "indices[1247,2048] int32; offset int32; baseline output row stride2052",
                sort_before,
                sort_after,
            ),
            (
                "rope_output",
                "1252 tokens H64 Q stride256, K stride576 ->contiguous64; non-NeoX",
                rope_before,
                rope_after,
            ),
            (
                "history_remap_flashmla",
                "rawKV[99840,1,576] logical99611 BF16; timeline Q[1247,64,576],TopK2048",
                history_before,
                history_after,
            ),
        ]

    def test_launch_fusion_execution_speed(self) -> None:
        self.assertTrue(torch.cuda.is_available())
        torch.manual_seed(412)
        enabled = {
            name: "1"
            for name in (
                "RTP_INDEXER_FUSED_TOPK",
                "RTP_CP_FUSED_TOPK_OFFSETS",
            )
        }
        selected = set(
            filter(None, os.environ.get("RTP_FUSION_BENCH_CASES", "").split(","))
        )
        report = {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "warmup": 30,
            "samples": 100,
            "timing_note": "CUDA events measure stream latency including host launch gaps; profiler kernel sum is separate. Input allocation and compile warmup excluded; operator output allocation included. No whole-model claim.",
            "cases": [],
        }
        with patch.dict(os.environ, enabled):
            for name, shape, baseline, candidate in self.fixtures():
                if selected and name not in selected:
                    continue
                with self.subTest(case=name):
                    setup = self.reset_rope if name == "rope_output" else None
                    if setup is not None:
                        setup()
                    expected = tuple(t.clone() for t in baseline())
                    if setup is not None:
                        setup()
                    actual = candidate()
                    self.assert_bits(actual, expected)
                    before, after = measure_pair(baseline, candidate, setup)
                    result = {
                        "case": name,
                        "shape": shape,
                        "exact_bits": True,
                        "baseline": before,
                        "candidate": after,
                        "stream_speedup": before["cuda_event_stream_latency_us"][
                            "median"
                        ]
                        / after["cuda_event_stream_latency_us"]["median"],
                        "kernel_sum_speedup": before["kernel_sum_us_per_iter"]
                        / after["kernel_sum_us_per_iter"],
                    }
                    report["cases"].append(result)
                    print(
                        "LAUNCH_FUSION_BENCHMARK_CASE", json.dumps(result), flush=True
                    )
        self.assertTrue(report["cases"])
        output = json.dumps(report, indent=2)
        print("LAUNCH_FUSION_BENCHMARK_JSON", output, flush=True)
        artifact_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if artifact_dir:
            Path(artifact_dir, "launch_fusion_benchmark.json").write_text(output)


if __name__ == "__main__":
    main()
