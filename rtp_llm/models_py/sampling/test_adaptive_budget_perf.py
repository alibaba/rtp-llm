import json
import os
import statistics
import time
import unittest
from pathlib import Path

import numpy as np
import torch

from rtp_llm.models_py.sampling.adaptive_verification import (
    BudgetCostBufferPool,
    select_draft_token_budget,
    select_draft_token_budget_device,
)


class AdaptiveBudgetPerfTest(unittest.TestCase):
    def test_selector_and_cost_pool(self) -> None:
        generator = np.random.default_rng(20261117)
        reports = []
        for batch, steps in ((31, 3), (127, 5), (1031, 5)):
            probabilities = generator.uniform(0.4, 0.99, (batch, steps)).astype(
                np.float32
            )
            capacities = generator.integers(1, steps + 1, batch, dtype=np.int32)
            confidence = torch.tensor(probabilities, device="cuda")
            slots = torch.arange(batch, dtype=torch.int64, device="cuda")
            caps = torch.tensor(capacities, device="cuda")
            draft_costs = np.linspace(0, 1, batch + 1)
            verify_costs = 1 + np.arange(batch * (steps + 1) + 1) * 0.002
            pool = BudgetCostBufferPool()

            def cpu_selector() -> int:
                return select_draft_token_budget(
                    confidence.index_select(0, slots).cpu().numpy(),
                    capacities,
                    draft_cost_ms=draft_costs,
                    verify_cost_ms=verify_costs,
                    num_non_draft_tokens=batch,
                    num_sampling_requests=batch,
                )

            def device_selector() -> int:
                return select_draft_token_budget_device(
                    confidence,
                    slots,
                    caps,
                    capacities,
                    draft_cost_ms=draft_costs,
                    verify_cost_ms=verify_costs,
                    num_non_draft_tokens=batch,
                    num_sampling_requests=batch,
                )

            def pooled_selector() -> int:
                return select_draft_token_budget_device(
                    confidence,
                    slots,
                    caps,
                    capacities,
                    cost_buffer_pool=pool,
                    draft_cost_ms=draft_costs,
                    verify_cost_ms=verify_costs,
                    num_non_draft_tokens=batch,
                    num_sampling_requests=batch,
                )

            operations = (cpu_selector, device_selector, pooled_selector)
            names = ("cpu_reference", "cuda_allocating", "cuda_pooled")
            expected = cpu_selector()
            for _ in range(10):
                for operation in operations:
                    self.assertEqual(operation(), expected)
            wall = [[], [], []]
            stream = [[], [], []]
            for repeat in range(30):
                for index in ((repeat + offset) % 3 for offset in range(3)):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    torch.cuda.synchronize()
                    start.record()
                    begin = time.perf_counter_ns()
                    actual = operations[index]()
                    end.record()
                    end.synchronize()
                    wall[index].append((time.perf_counter_ns() - begin) / 1000)
                    stream[index].append(start.elapsed_time(end) * 1000)
                    self.assertEqual(actual, expected)
            for index, operation in enumerate(operations):
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profile:
                    for _ in range(10):
                        operation()
                    torch.cuda.synchronize()
                kernels = [
                    event
                    for event in profile.events()
                    if event.device_type == torch.autograd.DeviceType.CUDA
                    and not event.name.lower().startswith(("memcpy", "memset"))
                ]
                reports.append(
                    {
                        "batch": batch,
                        "steps": steps,
                        "variant": names[index],
                        "selected_budget": expected,
                        "median_synchronized_wall_us": statistics.median(wall[index]),
                        "median_cuda_event_stream_latency_us": statistics.median(
                            stream[index]
                        ),
                        "wall_samples_us": wall[index],
                        "kernel_sum_us_per_iter": sum(
                            event.time_range.elapsed_us() for event in kernels
                        )
                        / 10,
                        "kernel_launches_per_iter": len(kernels) / 10,
                        "kernel_names": sorted({event.name for event in kernels}),
                    }
                )
        destination = (
            Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
            / "adaptive_budget_perf.json"
        )
        destination.write_text(json.dumps(reports, indent=2))
        print(json.dumps(reports), flush=True)


if __name__ == "__main__":
    unittest.main()
