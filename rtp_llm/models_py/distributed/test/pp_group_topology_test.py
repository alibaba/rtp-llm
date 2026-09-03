# CPU-only: validates _create_process_groups topology under gloo via multiprocessing spawn.

import multiprocessing as mp
import os
import socket
import unittest


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
        ct._create_process_groups(cfg, "gloo", None)

        membership = {
            str(key): torch.distributed.get_process_group_ranks(pg)
            for key, pg in ct._group_map.items()
        }
        # Runtime key derivation must land on a group this rank belongs to.
        looked_up = {}
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
        queue.put(
            (rank, cfg.pp_rank, cfg.dp_rank, cfg.tp_rank, membership, looked_up, None)
        )
    except Exception as exc:  # noqa: BLE001 - propagate to parent
        queue.put((rank, None, None, None, {}, {}, repr(exc)))
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
            rank, pp_rank, dp_rank, tp_rank, membership, looked_up, err = queue.get(
                timeout=180
            )
            test.assertIsNone(err, f"rank {rank} failed: {err}")
            results[rank] = (pp_rank, dp_rank, tp_rank, membership, looked_up)
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
        for _, (_, _, _, membership, _) in results.items():
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

    def test_pp2_dp1_tp1_pp_group_only(self):
        results = _run_topology(self, pp_size=2, dp_size=1, tp_size=1)
        membership = self._merge(results)
        self.assertEqual(membership["PP0"], [0, 1])
        self.assertFalse(any(k.startswith(("TP", "DP")) for k in membership))
        self.assertEqual(results[0][4]["PP"], [0, 1])


if __name__ == "__main__":
    unittest.main()
