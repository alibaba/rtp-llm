# CPU-only: validates group routing and stage/PP tensor all-gather with Gloo replacing NCCL.

import json
import multiprocessing as mp
import os
import socket
import threading
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
        ct._create_process_groups(cfg, "gloo", None, disable_custom_all_reduce=True)
        if pp_size == 1:
            assert ct._get_group(ct.Group.STAGE) is torch.distributed.group.WORLD
            for cpu_backend in (False, True):
                try:
                    ct._get_group(ct.Group.PP, cpu_backend=cpu_backend)
                except ValueError as exc:
                    assert "PP process groups require pp_size > 1" in str(exc)
                else:
                    raise AssertionError("PP group lookup must fail when pp_size=1")

        callbacks = {}
        cpp_ops = SimpleNamespace(
            register_comm_ops=lambda broadcast, allreduce, allgather, isend, irecv: callbacks.update(
                allgather=allgather, isend=isend, irecv=irecv
            ),
        )
        with patch.dict("sys.modules", {"librtp_compute_ops": cpp_ops}):
            ct._register_process_groups_to_cpp()

        membership = {
            str(key): torch.distributed.get_process_group_ranks(pg)
            for key, pg in ct._group_map.items()
        }
        if pp_size > 1:
            assert (f"PP_RESULT{cfg.dp_rank}" in membership) == (
                cfg.tp_rank == 0 and cfg.pp_rank in (0, pp_size - 1)
            )
        # Runtime key derivation must land on a group this rank belongs to.
        looked_up = {
            "WORLD": torch.distributed.get_process_group_ranks(
                ct._get_group(ct.Group.WORLD)
            ),
        }
        for group, size in (
            (ct.Group.TP, tp_size),
            (ct.Group.DP, dp_size),
            (ct.Group.STAGE, tp_size * dp_size),
        ):
            pg = ct._get_group(group)
            looked_up[group.name] = torch.distributed.get_process_group_ranks(pg)
            assert pg.size() == size
            if size == world_size:
                assert pg is torch.distributed.group.WORLD
            if size == 1:
                assert looked_up[group.name] == [rank]
                tensor = torch.tensor([[rank, rank + 10]], dtype=torch.int32)
                gathered = ct.all_gather(tensor, group)
                assert gathered.shape == tensor.shape
                assert torch.equal(gathered, tensor)
                assert gathered.data_ptr() != tensor.data_ptr()
        if pp_size > 1 and dp_size == 1:
            assert ct._get_group(ct.Group.STAGE) is ct._get_group(ct.Group.TP)
        elif pp_size > 1 and tp_size == 1:
            assert ct._get_group(ct.Group.STAGE) is ct._get_group(ct.Group.DP)
        if max(pp_size, 1) > 1:
            pp_group = ct._get_group(ct.Group.PP)
            pp_cpu_group = ct._get_group(ct.Group.PP, cpu_backend=True)
            # Both transports use Gloo in this CPU test, but remain distinct groups.
            assert pp_group is not torch.distributed.group.WORLD
            assert pp_cpu_group is not torch.distributed.group.WORLD
            assert pp_group is not pp_cpu_group
            looked_up["PP"] = torch.distributed.get_process_group_ranks(pp_group)
            assert (
                torch.distributed.get_process_group_ranks(pp_cpu_group)
                == looked_up["PP"]
            )

            # Non-contiguous lanes distinguish global and group-local ranks.
            pp_ranks = looked_up["PP"]
            pp_index = pp_ranks.index(rank)
            next_rank = pp_ranks[(pp_index + 1) % len(pp_ranks)]
            prev_rank = pp_ranks[(pp_index - 1) % len(pp_ranks)]
            result_values = []
            result_thread = None
            result_works = []
            result_buffers = []
            lengths = (1, 17, 5, 257)
            if cfg.tp_rank == 0 and cfg.pp_rank == 0:
                result_group = ct._group_map[f"PP_RESULT{cfg.dp_rank}"]
                assert result_group is not pp_cpu_group
                assert torch.distributed.get_process_group_ranks(result_group) == [
                    rank,
                    pp_ranks[-1],
                ]

                def receive_results():
                    for length in lengths:
                        size = torch.empty(1, dtype=torch.int64)
                        callbacks["irecv"](
                            size,
                            prev_rank,
                            ct._PP_RESULT_GROUP_NAME,
                        ).wait()
                        assert size.item() == length
                        payload = torch.empty(size.item(), dtype=torch.int64)
                        callbacks["irecv"](
                            payload,
                            prev_rank,
                            ct._PP_RESULT_GROUP_NAME,
                        ).wait()
                        result_values.append(payload.tolist())

                result_thread = threading.Thread(target=receive_results)
                result_thread.start()
            elif cfg.tp_rank == 0 and cfg.pp_rank == pp_size - 1:
                for length in lengths:
                    size = torch.tensor([length], dtype=torch.int64)
                    payload = torch.full((length,), rank, dtype=torch.int64)
                    result_buffers.extend((size, payload))
                    for tensor in (size, payload):
                        result_works.append(
                            callbacks["isend"](
                                tensor,
                                next_rank,
                                ct._PP_RESULT_GROUP_NAME,
                            )
                        )

            sent = torch.tensor([rank], dtype=torch.int64)
            received = torch.full((1,), -1, dtype=torch.int64)
            recv_work = callbacks["irecv"](received, prev_rank, ct._PP_CPU_GROUP_NAME)
            send_work = callbacks["isend"](
                sent,
                next_rank,
                ct._PP_CPU_GROUP_NAME,
            )
            send_work.wait()
            recv_work.wait()
            assert received.item() == prev_rank, (rank, pp_ranks, received)
            looked_up["PP_P2P_SOURCE"] = [received.item()]
            for work in result_works:
                work.wait()
            if result_thread is not None:
                result_thread.join(timeout=30)
                assert not result_thread.is_alive()
                assert result_values == [[prev_rank] * length for length in lengths]
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

        # Test Python callbacks with real Gloo collectives, not the C++ bridge.
        # Replace CUDA staging only to keep tensors on CPU.
        with patch.object(torch.cuda, "current_device", return_value=0), patch.object(
            torch.Tensor, "to", cpu_staging
        ):
            stage_begin = rank // stage_size * stage_size
            tp_begin = rank // tp_size * tp_size
            for mode, ranks in (
                (ct._CPP_PARALLEL_MODE_TP, range(tp_begin, tp_begin + tp_size)),
                (
                    ct._CPP_PARALLEL_MODE_DP,
                    range(
                        stage_begin + rank % tp_size,
                        stage_begin + stage_size,
                        tp_size,
                    ),
                ),
                (
                    ct._CPP_PARALLEL_MODE_STAGE,
                    range(stage_begin, stage_begin + stage_size),
                ),
            ):
                send = torch.tensor([rank, rank + 10], dtype=torch.int32)
                recv = torch.full((len(ranks), 2), -1, dtype=torch.int32)
                callbacks["allgather"]([recv], mode, [send], False)
                expected = torch.tensor([[r, r + 10] for r in ranks], dtype=torch.int32)
                assert torch.equal(recv, expected), (rank, mode, recv, expected)

            for local_block_num in local_blocks:
                block_nums = torch.full((stage_size,), -1, dtype=torch.int32)
                block_nums[local_rank] = local_block_num
                callbacks["allgather"](
                    [block_nums], ct._CPP_PARALLEL_MODE_STAGE, [], True
                )
                gathered_blocks.append(block_nums.tolist())
                stage_blocks.append(block_nums.min().item())

            stage_snapshots = [stage_blocks]
            if pp_size > 1:
                payload = json.dumps(stage_blocks).encode()
                sizes = torch.empty(pp_size, dtype=torch.int64)
                callbacks["allgather"](
                    [sizes],
                    ct._CPP_PARALLEL_MODE_PP,
                    [torch.tensor([len(payload)], dtype=torch.int64)],
                    False,
                )
                max_size = sizes.max().item()
                send = torch.zeros(max_size, dtype=torch.uint8)
                send[: len(payload)] = torch.tensor(list(payload), dtype=torch.uint8)
                received = torch.empty((pp_size, max_size), dtype=torch.uint8)
                callbacks["allgather"](
                    [received], ct._CPP_PARALLEL_MODE_PP, [send], False
                )
                stage_snapshots = [
                    json.loads(bytes(row[:size].tolist()))
                    for row, size in zip(received, sizes.tolist())
                ]
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
            if "STAGE" in looked_up:
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
        timed_out = []
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                timed_out.append(p.pid)
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join()
    test.assertFalse(timed_out, f"workers deadlocked: {timed_out}")
    for rank, p in enumerate(procs):
        test.assertEqual(p.exitcode, 0, f"rank {rank} exited with code {p.exitcode}")
    return results


class PPGroupTopologyTest(unittest.TestCase):
    def test_pp_callbacks_route_by_group_name(self):
        import torch

        from rtp_llm.models_py.distributed import collective_torch as ct

        device_group, cpu_group, result_group = (
            object(),
            object(),
            object(),
        )
        groups = {
            ct._PP_DEVICE_GROUP_NAME: device_group,
            ct._PP_CPU_GROUP_NAME: cpu_group,
            ct._PP_RESULT_GROUP_NAME: result_group,
        }
        group_ranks = {
            device_group: [0, 1, 2],
            cpu_group: [0, 1, 2],
            result_group: [0, 2],
        }
        callbacks = {}
        cpp_ops = SimpleNamespace(
            register_comm_ops=lambda broadcast, allreduce, allgather, isend, irecv: callbacks.update(
                isend=isend, irecv=irecv
            ),
        )
        cfg = SimpleNamespace(world_size=3, pp_size=3, dp_size=1, tp_size=1, dp_rank=0)
        with patch.dict("sys.modules", {"librtp_compute_ops": cpp_ops}), patch.object(
            ct, "_parallelism_config", cfg
        ), patch.object(ct, "_group_map", {"PP_RESULT0": result_group}), patch.object(
            ct,
            "_get_group",
            side_effect=lambda group, cpu_backend=False: (
                cpu_group if cpu_backend else device_group
            ),
        ), patch.object(
            torch.distributed, "get_rank", return_value=0
        ):
            ct._register_process_groups_to_cpp()

        tensor = object()
        for operation, peer_arg in (("isend", "dst"), ("irecv", "src")):
            with patch.object(
                torch.distributed, operation, return_value=object()
            ) as p2p_op:
                for group_name, group in groups.items():
                    peer = group_ranks[group][-1]
                    with self.subTest(operation=operation, group_name=group_name):
                        p2p_op.reset_mock()
                        work = callbacks[operation](tensor, peer, group_name)
                        p2p_op.assert_called_once_with(
                            tensor, **{peer_arg: peer, "group": group}
                        )
                        self.assertIs(work, p2p_op.return_value)

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

    def test_single_rank_and_pure_tp_dp_world_aliases(self):
        for tp_size, dp_size in ((1, 1), (2, 1), (1, 2)):
            with self.subTest(tp_size=tp_size, dp_size=dp_size):
                _run_topology(self, pp_size=1, dp_size=dp_size, tp_size=tp_size)

    def test_pp1_dp2_tp2_uses_world_stage_and_local_dp_tp_groups(self):
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
        for rank in range(4):
            self.assertEqual(membership[f"DP{rank}"], [rank])
        self.assertEqual(membership["STAGE0"], [0, 1])
        self.assertEqual(membership["STAGE1"], [2, 3])
        self.assertEqual(results[0][0], 0)
        self.assertEqual(results[3][0], 1)
        self.assertEqual(results[0][4]["TP"], [0, 1])
        self.assertEqual(results[0][4]["PP"], [0, 2])
        self.assertEqual(results[3][4]["TP"], [2, 3])
        self.assertEqual(results[3][4]["PP"], [1, 3])
        self.assertEqual(results[0][4]["PP_P2P_SOURCE"], [2])
        self.assertEqual(results[2][4]["PP_P2P_SOURCE"], [0])

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
        self.assertEqual(membership["PP_RESULT0"], [0, 4])
        self.assertEqual(membership["PP_RESULT1"], [2, 6])
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

    def test_pp3_dp2_tp1_results_skip_middle_stage(self):
        results = _run_topology(self, pp_size=3, dp_size=2, tp_size=1)
        membership = self._merge(results)
        self.assertEqual(membership["PP_RESULT0"], [0, 4])
        self.assertEqual(membership["PP_RESULT1"], [1, 5])

    def test_pp2_dp2_tp1_stage_reuses_dp(self):
        results = _run_topology(self, pp_size=2, dp_size=2, tp_size=1)
        membership = self._merge(results)
        self.assertEqual(membership["DP0"], [0, 1])
        self.assertEqual(membership["DP1"], [2, 3])
        self.assertEqual(membership["STAGE0"], [0, 1])
        self.assertEqual(membership["STAGE1"], [2, 3])
        for result in results.values():
            stage_snapshots = result[5][2]
            self.assertEqual(
                [min(column) for column in zip(*stage_snapshots)], [80, 96, 112, 128]
            )

    def test_pp2_dp1_tp1_singleton_tp_dp_groups(self):
        results = _run_topology(self, pp_size=2, dp_size=1, tp_size=1)
        membership = self._merge(results)
        self.assertEqual(membership["PP0"], [0, 1])
        self.assertEqual(membership["PP0_gloo"], [0, 1])
        self.assertEqual(membership["PP_RESULT0"], [0, 1])
        for rank in range(2):
            self.assertEqual(membership[f"TP{rank}"], [rank])
            self.assertEqual(membership[f"DP{rank}"], [rank])
            self.assertEqual(membership[f"STAGE{rank}"], [rank])
            for group in ("TP", "DP", "STAGE"):
                self.assertEqual(results[rank][4][group], [rank])
        self.assertEqual(results[0][4]["PP"], [0, 1])
        for result in results.values():
            local_blocks, stage_blocks, stage_snapshots, _ = result[5]
            self.assertEqual(stage_blocks, local_blocks)
            self.assertEqual(
                [min(column) for column in zip(*stage_snapshots)], [80, 96, 112, 128]
            )


if __name__ == "__main__":
    unittest.main()
