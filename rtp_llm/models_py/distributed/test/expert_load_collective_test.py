"""Exercise C++ target/draft monitors on the existing two-rank NCCL group."""

import json
import multiprocessing as mp
import os
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from rtp_llm.cpp.models.eplb.test import libth_eplb_py_wrapper_test as monitor_test_ops
from rtp_llm.models_py.distributed import collective_torch as collective


def _layer_counts(rank, model_id, step):
    return [
        10 * (rank + step + 1) + 100 * model_id + layer + 1
        for layer in range(2 - model_id)
    ]


def _worker(rank, directory):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{directory}/store",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    collective._group_map = {collective.Group.DP_AND_TP: dist.group.WORLD}
    collective._parallelism_config = SimpleNamespace(
        tp_size=1, dp_size=2, world_size=2, local_world_size=2, tp_rank=0
    )

    def register_comm_ops(broadcast, allreduce, allgather):
        monitor_test_ops._allgather_callback = allgather

    # Inject the production callback into the test extension without loading the
    # service DSO and duplicating the extension's static dependencies.
    with patch.dict(
        sys.modules,
        {"librtp_compute_ops": SimpleNamespace(register_comm_ops=register_comm_ops)},
    ):
        collective._register_process_groups_to_cpp()
    try:
        # Different layer counts and report frequencies share the model communicator.
        # A nondefault producing stream also exercises NCCL -> blocking D2H ordering.
        with (
            torch.cuda.stream(torch.cuda.Stream()),
            patch.object(dist, "new_group") as new_group,
            patch.dict(os.environ, {"LOG_PATH": directory}),
        ):
            monitors = [
                monitor_test_ops.ExpertLoadMonitor(
                    2 - model_id, 4, 2, rank, 2 + model_id, model_id, 10
                )
                for model_id in range(2)
            ]
            steps = [0, 0]
            pointers = [monitor.counts.data_ptr() for monitor in monitors]

            def step(model_id):
                monitor = monitors[model_id]
                values = _layer_counts(rank, model_id, steps[model_id])
                counts = torch.tensor(
                    [[value - 1, 1] for value in values],
                    device="cuda",
                    dtype=torch.int32,
                )
                monitor.counts.add_(counts)
                monitor.record_step()
                assert torch.count_nonzero(monitor.counts).item() == 0
                assert monitor.counts.data_ptr() == pointers[model_id]
                steps[model_id] += 1
                # Ordinary model communication remains usable between monitor windows.
                value = torch.tensor([rank + 1], device="cuda")
                dist.all_reduce(value)
                assert value.item() == 3

            for model_id in (0, 1, 1, 0, 1, 0, 1, 1, 0, 1):
                step(model_id)
            monitors[0] = None
            for _ in range(3):
                step(1)
            monitors.clear()
            new_group.assert_not_called()
    finally:
        del monitor_test_ops._allgather_callback
        collective._group_map.clear()
        dist.destroy_process_group()


class ExpertLoadCollectiveTest(unittest.TestCase):
    def test_target_and_draft_share_nccl_with_independent_windows(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        with tempfile.TemporaryDirectory() as directory:
            context = mp.get_context("spawn")
            workers = [
                context.Process(target=_worker, args=(rank, directory))
                for rank in range(2)
            ]
            for worker in workers:
                worker.start()
            try:
                for worker in workers:
                    worker.join(90)
                    self.assertEqual(worker.exitcode, 0)
            finally:
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                    worker.join()

            for model_id, window_count in ((0, 2), (1, 3)):
                snapshots = sorted(
                    (
                        json.loads(path.read_text())
                        for path in (Path(directory) / "eplb_snapshot").glob(
                            f"model_{model_id}_*.json"
                        )
                    ),
                    key=lambda window: window["step_begin"],
                )
                self.assertEqual(len(snapshots), window_count)
                capacity = 2 + model_id
                for index, window in enumerate(snapshots):
                    self.assertEqual(window["world_rank"], 0)
                    self.assertEqual(
                        window["model_type"], "target" if model_id == 0 else "draft"
                    )
                    self.assertEqual(window["step_begin"], index * capacity + 1)
                    self.assertEqual(
                        window["rank_samples"],
                        [
                            [
                                _layer_counts(rank, model_id, index * capacity + step)
                                for step in range(capacity)
                            ]
                            for rank in range(2)
                        ],
                    )


if __name__ == "__main__":
    unittest.main()
