# CPU-only: validates process groups, stage capacity all-gather, and PP lane snapshot exchange via gloo.

import json
import multiprocessing as mp
import os
import socket
import unittest
from types import SimpleNamespace
from unittest.mock import patch


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, world_size, pp_size, dp_size, tp_size, master_port, queue):
    import torch
    import torch.distributed

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        from rtp_llm.models_py.distributed import collective_torch as ct

        cfg = ct.ParallelismConfig()
        cfg.world_rank = rank
        cfg.world_size = world_size
        cfg.tp_size = tp_size
        cfg.dp_size = dp_size
        cfg.pp_size = pp_size
        cfg.local_world_size = world_size
        ct._normalize_parallelism_ranks(cfg)
        ct._parallelism_config = cfg
        # gloo is already up: mark initialized so _get_group resolves keys directly.
        ct._initialized = True
        ct._group_map.clear()
        ct._group_map[ct.Group.WORLD] = torch.distributed.group.WORLD
        ct._create_process_groups(cfg, "gloo", None)

        callbacks = {}
        cpp_ops = SimpleNamespace(
            register_comm_ops=lambda broadcast, allreduce, allgather: callbacks.update(
                allgather=allgather
            ),
            register_pp_ops=lambda isend, irecv, exchange: callbacks.update(
                pp_snapshot_exchange=exchange
            ),
        )
        with patch.dict("sys.modules", {"librtp_compute_ops": cpp_ops}):
            ct._register_process_groups_to_cpp()

        membership = {
            str(key): torch.distributed.get_process_group_ranks(pg)
            for key, pg in ct._group_map.items()
        }
        # Runtime key derivation must land on a group this rank belongs to.
        looked_up = {
            "STAGE": torch.distributed.get_process_group_ranks(
                ct._get_group(ct.Group.STAGE)
            ),
            "WORLD": torch.distributed.get_process_group_ranks(
                ct._get_group(ct.Group.WORLD)
            ),
        }
        if tp_size > 1 and world_size != tp_size:
            looked_up["TP"] = torch.distributed.get_process_group_ranks(
                ct._get_group(ct.Group.TP)
            )
        if dp_size > 1 and world_size != dp_size:
            looked_up["DP"] = torch.distributed.get_process_group_ranks(
                ct._get_group(ct.Group.DP)
            )
        if max(pp_size, 1) > 1:
            looked_up["PP"] = torch.distributed.get_process_group_ranks(
                ct._get_group(ct.Group.PP)
            )
        # Each entry is a scalar-capacity case; minima fall on different TP/DP/PP ranks.
        local_blocks = [
            80 + (rank - 5) % world_size,
            96 + (rank - 2) % world_size,
            112 + rank,
            128 + (rank - 7) % world_size,
        ]
        stage_size = tp_size * dp_size
        local_rank = cfg.dp_rank * tp_size + cfg.tp_rank
        stage_blocks = []
        gathered_blocks = []
        tensor_to = torch.Tensor.to

        def cpu_staging(tensor, *args, **kwargs):
            if args and isinstance(args[0], torch.device) and args[0].type == "cuda":
                return tensor.clone()
            return tensor_to(tensor, *args, **kwargs)

        # Exercise the actual C++ all-gather callback and mode mapping with real gloo collectives.
        # Only CUDA staging is replaced so this topology test stays CPU-only.
        with patch.object(torch.cuda, "current_device", return_value=0), patch.object(
            torch.Tensor, "to", cpu_staging
        ):
            for local_block_num in local_blocks:
                block_nums = torch.full((stage_size,), -1, dtype=torch.int32)
                block_nums[local_rank] = local_block_num
                if stage_size > 1:
                    callbacks["allgather"](
                        [block_nums], ct._CPP_PARALLEL_MODE_STAGE, [], True
                    )
                gathered_blocks.append(block_nums.tolist())
                stage_blocks.append(block_nums.min().item())

        stage_snapshots = [stage_blocks]
        if pp_size > 1:
            payloads = callbacks["pp_snapshot_exchange"](
                json.dumps(stage_blocks).encode()
            )
            stage_snapshots = [json.loads(payload) for payload in payloads]
        capacities = (local_blocks, stage_blocks, stage_snapshots, gathered_blocks)
        queue.put(
            (
                rank,
                cfg.pp_rank,
                cfg.dp_rank,
                cfg.tp_rank,
                membership,
                looked_up,
                capacities,
                None,
            )
        )
    except Exception as exc:  # noqa: BLE001 - propagate to parent
        queue.put((rank, None, None, None, {}, {}, None, repr(exc)))
    finally:
        torch.distributed.destroy_process_group()


def _run_topology(test, pp_size, dp_size, tp_size):
    world_size = pp_size * dp_size * tp_size
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    # All workers must share one master port; pick it once before spawning.
    master_port = _free_port()
    procs = [
        ctx.Process(
            target=_worker,
            args=(r, world_size, pp_size, dp_size, tp_size, master_port, queue),
        )
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    results = {}
    try:
        for _ in range(world_size):
            rank, pp_rank, dp_rank, tp_rank, membership, looked_up, capacities, err = (
                queue.get(timeout=180)
            )
            test.assertIsNone(err, f"rank {rank} failed: {err}")
            stage_size = dp_size * tp_size
            test.assertEqual(
                looked_up["STAGE"],
                list(range(pp_rank * stage_size, (pp_rank + 1) * stage_size)),
            )
            test.assertEqual(looked_up["WORLD"], list(range(world_size)))
            results[rank] = (
                pp_rank,
                dp_rank,
                tp_rank,
                membership,
                looked_up,
                capacities,
            )
    finally:
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
                test.fail("worker deadlocked (group creation/barrier mismatch)")
    return results


class PPGroupTopologyTest(unittest.TestCase):
    @staticmethod
    def _merge(results):
        # Full topology is the union over all ranks' snapshots (each rank stores only its own groups).
        merged = {}
        for _, (_, _, _, membership, _, _) in results.items():
            for key, ranks in membership.items():
                if key in merged:
                    test_ranks = merged[key]
                    assert test_ranks == ranks, f"conflicting views of {key}"
                merged[key] = ranks
        return merged

    def test_pp1_dp2_tp2_matches_historical_groups(self):
        results = _run_topology(self, pp_size=1, dp_size=2, tp_size=2)
        membership = self._merge(results)
        self.assertEqual(membership["TP0"], [0, 1])
        self.assertEqual(membership["TP1"], [2, 3])
        self.assertEqual(membership["DP0"], [0, 2])
        self.assertEqual(membership["DP1"], [1, 3])
        self.assertFalse(any(k.startswith("PP") for k in membership))
        for result in results.values():
            _, stage_blocks, stage_snapshots, gathered_blocks = result[5]
            self.assertEqual(stage_blocks, [80, 96, 112, 128])
            self.assertEqual(stage_snapshots, [stage_blocks])
            self.assertEqual(
                gathered_blocks,
                [list(column) for column in zip(*(results[r][5][0] for r in range(4)))],
            )

    def test_pp2_dp1_tp2_stage_local_tp_and_lane_pp(self):
        results = _run_topology(self, pp_size=2, dp_size=1, tp_size=2)
        membership = self._merge(results)
        # TP groups must not cross the stage boundary at rank 2.
        self.assertEqual(membership["TP0"], [0, 1])
        self.assertEqual(membership["TP1"], [2, 3])
        # PP groups span the stages of each tp lane.
        self.assertEqual(membership["PP0"], [0, 2])
        self.assertEqual(membership["PP1"], [1, 3])
        self.assertFalse(any(k.startswith("DP") for k in membership))
        self.assertEqual(results[0][0], 0)
        self.assertEqual(results[3][0], 1)
        self.assertEqual(results[0][4]["TP"], [0, 1])
        self.assertEqual(results[0][4]["PP"], [0, 2])
        self.assertEqual(results[3][4]["TP"], [2, 3])
        self.assertEqual(results[3][4]["PP"], [1, 3])

    def test_pp2_dp2_tp2_full_combination(self):
        results = _run_topology(self, pp_size=2, dp_size=2, tp_size=2)
        membership = self._merge(results)
        self.assertEqual(membership["STAGE0"], [0, 1, 2, 3])
        self.assertEqual(membership["STAGE1"], [4, 5, 6, 7])
        # 4 stage-local TP groups.
        self.assertEqual(membership["TP0"], [0, 1])
        self.assertEqual(membership["TP1"], [2, 3])
        self.assertEqual(membership["TP2"], [4, 5])
        self.assertEqual(membership["TP3"], [6, 7])
        # 4 stage-local DP groups (same pp and tp, across dp).
        self.assertEqual(membership["DP0"], [0, 2])
        self.assertEqual(membership["DP1"], [1, 3])
        self.assertEqual(membership["DP2"], [4, 6])
        self.assertEqual(membership["DP3"], [5, 7])
        # 4 lane PP groups (same dp and tp, across pp).
        self.assertEqual(membership["PP0"], [0, 4])
        self.assertEqual(membership["PP1"], [1, 5])
        self.assertEqual(membership["PP2"], [2, 6])
        self.assertEqual(membership["PP3"], [3, 7])
        self.assertEqual(results[0][4]["TP"], [0, 1])
        self.assertEqual(results[0][4]["DP"], [0, 2])
        self.assertEqual(results[0][4]["PP"], [0, 4])

        # Verify the intermediate stage capacities, not only the final global minimum.
        expected_by_stage = []
        for stage in range(2):
            local_capacities = [
                results[rank][5][0] for rank in range(stage * 4, (stage + 1) * 4)
            ]
            expected_by_stage.append([min(column) for column in zip(*local_capacities)])
        self.assertNotEqual(expected_by_stage[0], expected_by_stage[1])
        for rank, result in results.items():
            _, stage_blocks, stage_snapshots, gathered_blocks = result[5]
            stage_begin = (rank // 4) * 4
            local_capacities = [
                results[r][5][0] for r in range(stage_begin, stage_begin + 4)
            ]
            self.assertEqual(
                gathered_blocks,
                [list(column) for column in zip(*local_capacities)],
                f"rank {rank}",
            )
            self.assertEqual(stage_blocks, expected_by_stage[rank // 4], f"rank {rank}")
            self.assertEqual(stage_snapshots, expected_by_stage, f"rank {rank}")
            self.assertEqual(
                [min(column) for column in zip(*stage_snapshots)], [80, 96, 112, 128]
            )

    def test_pp2_dp2_tp1_stage_group(self):
        results = _run_topology(self, pp_size=2, dp_size=2, tp_size=1)
        membership = self._merge(results)
        self.assertEqual(membership["STAGE0"], [0, 1])
        self.assertEqual(membership["STAGE1"], [2, 3])
        for result in results.values():
            stage_snapshots = result[5][2]
            self.assertEqual(
                [min(column) for column in zip(*stage_snapshots)], [80, 96, 112, 128]
            )

    def test_pp2_dp1_tp1_pp_group_only(self):
        results = _run_topology(self, pp_size=2, dp_size=1, tp_size=1)
        membership = self._merge(results)
        self.assertEqual(membership["PP0"], [0, 1])
        self.assertFalse(any(k.startswith(("TP", "DP")) for k in membership))
        self.assertEqual(results[0][4]["PP"], [0, 1])
        self.assertEqual(membership["STAGE0"], [0])
        self.assertEqual(membership["STAGE1"], [1])
        for result in results.values():
            local_blocks, stage_blocks, stage_snapshots, _ = result[5]
            self.assertEqual(stage_blocks, local_blocks)
            self.assertEqual(
                [min(column) for column in zip(*stage_snapshots)], [80, 96, 112, 128]
            )


if __name__ == "__main__":
    unittest.main()
