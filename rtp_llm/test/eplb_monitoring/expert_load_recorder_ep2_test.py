# type: ignore
"""EP2 cross-source test for the production EPLB expert-load recorder.

Contract acceptance case 2: physical experts 0,1 -> rank0; 2,3 -> rank1.

* source0 routes [[2,3],[2,3]] (every token destined to rank1's experts)
* source1 routes [[0,1]]      (every token destined to rank0's experts)

Source contributions cover ALL destinations (global physical indexing), so
after one SUM over sources the histogram is [1,1,2,2], per-rank totals are
[2,4], and the grand total is 6.  Distinct EP/DP shards are independent
sources: both count (unlike TP replication, which must dedup).

Each rank runs the REAL production path: a real PureTpRouterNoQuant (tp_size
1, so every rank is its own canonical owner) bound by the real
bind_expert_load_monitor, recording into its own step_counts.

The cross-source SUM transport here is test scaffolding
(torch.distributed.all_reduce, NCCL). This test verifies source-count
recording and the summed histogram, not the engine's window lifecycle or
DeepEP dispatch. Those require the C++ tests and real-model EP smoke.

Requires 2 GPUs (NCCL).  No fake monitor anywhere: production imports hard-fail.
"""

import sys
import unittest
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Loose-script runs (python -I <path>) also need the workspace root so that
# `import rtp_llm...` resolves; under Bazel runfiles it is already present.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from window_ratio_oracle import WindowTracker  # noqa: E402

# --------------------------------------------------------------------------
# Production imports -- hard failure, never a substitute.
# --------------------------------------------------------------------------
from rtp_llm.models_py.utils.expert_load_monitor import (  # noqa: E402
    bind_expert_load_monitor,
)

NUM_LAYERS = 1
NUM_EXPERTS = 4
EXPERT_TO_RANK = [0, 0, 1, 1]
WORLD = 2

# Routing per rank: rank0's tokens all go to experts 2/3 (owned by rank1),
# rank1's token goes to experts 0/1 (owned by rank0).
ROUTING = {
    0: [[2, 3], [2, 3]],
    1: [[0, 1]],
}
EXPECTED_HIST = [1, 1, 2, 2]
EXPECTED_RANK_TOTALS = [2, 4]
EXPECTED_TOTAL = 6


def _reduce_sources(step_counts: torch.Tensor) -> torch.Tensor:
    """Test transport for exactly one SUM of source contributions."""
    out = step_counts.clone()
    torch.distributed.all_reduce(out, op=torch.distributed.ReduceOp.SUM)
    return out


def _rank_main(rank: int, init_method: str, result_queue) -> None:
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=init_method,
        world_size=WORLD,
        rank=rank,
    )
    try:
        # Deferred import: the parent process must not initialise CUDA.
        from expert_load_recorder_test import (  # noqa: PLC0415
            _make_router,
            _routing_tensors,
            _SyntheticModel,
        )

        step_counts = torch.zeros(
            NUM_LAYERS, NUM_EXPERTS, dtype=torch.int64, device="cuda"
        )
        routers = [
            _make_router(tp_size=1, tp_rank=0, ep_size=WORLD, ep_rank=rank)
            for _ in range(NUM_LAYERS)
        ]
        model = _SyntheticModel(routers).cuda()
        bind_expert_load_monitor(model, step_counts)

        rows = ROUTING[rank]
        model([_routing_tensors(rows)] * NUM_LAYERS)
        torch.cuda.synchronize()

        summed = _reduce_sources(step_counts)
        torch.cuda.synchronize()
        result_queue.put((rank, summed.to("cpu").tolist()))
    finally:
        torch.distributed.destroy_process_group()


class ExpertLoadRecorderEp2Test(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD:
            self.fail(f"This distributed GPU target requires {WORLD} CUDA devices")

    def test_ep2_cross_source_sum_and_ratios(self) -> None:
        import socket

        import torch.multiprocessing as mp

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            init_method = f"tcp://127.0.0.1:{sock.getsockname()[1]}"

        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        procs = [
            ctx.Process(target=_rank_main, args=(r, init_method, queue))
            for r in range(WORLD)
        ]
        results = {}
        try:
            for p in procs:
                p.start()
            for _ in range(WORLD):
                rank, hist = queue.get(timeout=300)
                results[rank] = hist
            for p in procs:
                p.join(timeout=300)
                self.assertEqual(p.exitcode, 0, f"rank process failed: {p.pid}")
        finally:
            for p in procs:
                if p.pid is not None:
                    if p.is_alive():
                        p.terminate()
                    p.join(timeout=10)
            queue.close()

        # Every rank must observe the same SUM result.
        for rank in range(WORLD):
            self.assertEqual(
                results[rank][0],
                EXPECTED_HIST,
                f"rank{rank} summed histogram mismatch",
            )

        # Feed the observed histogram through the independent oracle: rank
        # totals [2,4], total 6, ratios computed only after the cross-source
        # SUM.
        # Every rank holds the same global SUM. Feed one copy to the oracle;
        # summing the returned replicas would count every activation twice.
        tracker = WindowTracker(
            num_layers=NUM_LAYERS,
            num_physical_experts=NUM_EXPERTS,
            expert_to_rank=EXPERT_TO_RANK,
            num_ranks=WORLD,
            window_size=1,
        )
        rec = tracker.begin_step()
        for e, count in enumerate(results[0][0]):
            rec.record(0, [e] * count)
        tracker.finish_step()
        report = tracker.close_window()
        ratios = report.layer_ratios(0)
        self.assertEqual(ratios["rank_totals"], EXPECTED_RANK_TOTALS)
        self.assertEqual(ratios["sample_count"], EXPECTED_TOTAL)
        self.assertTrue(ratios["valid"])
        # max[2,4]/mean[2,4] = 4/3 for both ratio definitions at W=1.
        self.assertEqual(float(ratios["moe_lb"]), 4.0 / 3.0)
        self.assertEqual(float(ratios["moe_lb_window"]), 4.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
