import json
import statistics
import unittest

import torch

from rtp_llm.eplb.maxpander_plan import build_circulant_plan
from rtp_llm.models_py.modules.maxpander import MaxPanderRouter


def measure_stream_latency(function):
    for _ in range(30):
        function()
    torch.cuda.synchronize()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(100)
    ]
    allocations_before = torch.cuda.memory_stats()["allocation.all.allocated"]
    for start, end in events:
        start.record()
        function()
        end.record()
    torch.cuda.synchronize()
    allocations_after = torch.cuda.memory_stats()["allocation.all.allocated"]
    samples = sorted(start.elapsed_time(end) * 1000 for start, end in events)
    return dict(
        median_us=statistics.median(samples),
        p90_us=samples[89],
        min_us=samples[0],
        max_us=samples[-1],
        torch_allocator_calls_per_iter=(allocations_after - allocations_before) / 100,
    )


class MaxPanderPerfTest(unittest.TestCase):
    def test_rank_local_prefill_routing_cost(self):
        self.assertTrue(torch.cuda.is_available())
        placement, candidates, replicas = build_circulant_plan(256, 320, 2)
        router = MaxPanderRouter(
            torch.tensor(candidates, device="cuda", dtype=torch.int32),
            torch.tensor(replicas, device="cuda", dtype=torch.int32),
            num_logical=256,
            num_physical=320,
            ep_size=2,
            wf_passes=8,
        )
        physical_to_logical = torch.tensor(placement, device="cuda")
        records = []
        for tokens in (1, 17, 257, 1024, 4096, 8192):
            logical = torch.arange(tokens * 6, device="cuda").view(tokens, 6) % 256
            packed = torch.empty_like(logical)

            def disabled():
                packed.copy_(logical)

            def enabled():
                packed.copy_(logical)
                router.remap_inplace(packed)

            enabled()
            self.assertTrue(torch.equal(physical_to_logical[packed], logical))
            records.append(
                dict(
                    tokens=tokens,
                    disabled_id_copy=measure_stream_latency(disabled),
                    enabled_id_copy_and_wf=measure_stream_latency(enabled),
                )
            )
            self.assertEqual(
                records[-1]["enabled_id_copy_and_wf"]["torch_allocator_calls_per_iter"],
                0.0,
            )
        print(
            json.dumps(
                dict(
                    benchmark="maxpander_rank_local_routing_stream_latency",
                    device=torch.cuda.get_device_name(),
                    logical_experts=256,
                    physical_experts=320,
                    ep_size=2,
                    topk=6,
                    wf_passes=8,
                    torch_compile=False,
                    timing="CUDA events; includes host launch gaps, excludes compilation and tensor setup; not pure kernel time or serving latency",
                    warmup=30,
                    measured=100,
                    records=records,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
