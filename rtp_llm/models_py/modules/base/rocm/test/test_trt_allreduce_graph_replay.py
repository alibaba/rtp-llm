# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# Regression test: trtallreduce IPC fast-path under hipGraph capture + replay.
# Requires >= 2 ROCm GPUs.

import os
import socket
import sys
import unittest
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

warnings.filterwarnings(
    "ignore", message="barrier.*using the device under current context"
)
warnings.filterwarnings("ignore", message="Guessing device ID based on global rank")

REPLAY_ROUNDS = 60


def _setup(rank, world_size, port):
    os.environ.update(
        MASTER_ADDR="localhost",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{rank}"),
    )


def _teardown():
    try:
        dist.barrier()
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _launch(fn, world_size=2, timeout=180, **kw):
    port = _free_port()
    procs = []
    for r in range(world_size):
        p = mp.Process(
            target=fn, args=(r, world_size, port), kwargs=kw, name=f"rank-{r}"
        )
        p.start()
        procs.append(p)
    try:
        for p in procs:
            p.join(timeout=timeout)
            if p.exitcode is None:
                raise RuntimeError(
                    f"{p.name} timed out after {timeout}s (still running)"
                )
            if p.exitcode != 0:
                raise RuntimeError(f"{p.name} exited with code {p.exitcode}")
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join(timeout=2)


def _worker_graph_pure_allreduce(rank, world_size, port, num_replays):
    try:
        _setup(rank, world_size, port)
        from rtp_llm.models_py.modules.base.rocm.trt_allreduce import TrtllmDistEnv

        dev = torch.device(f"cuda:{rank}")
        env = TrtllmDistEnv(group=dist.group.WORLD, device_id=rank)

        torch.manual_seed(42 + rank)
        inp = torch.randn(8, 4096, dtype=torch.bfloat16, device=dev)

        ref = inp.clone()
        dist.all_reduce(ref)

        s = torch.cuda.Stream(device=dev)
        with torch.cuda.stream(s):
            env.allreduce_op(inp.clone(), torch.empty_like(inp))
            s.synchronize()
            g = torch.cuda.CUDAGraph()
            g_in, g_out = inp.clone(), torch.empty_like(inp)
            with torch.cuda.graph(g, stream=s):
                env.allreduce_op(g_in, g_out)
        s.synchronize()
        env.consume_capture_if_needed()

        for i in range(num_replays):
            g_in.copy_(inp)
            g.replay()
            s.synchronize()
            diff = (g_out - ref).abs().max().item()
            ref_max = ref.abs().max().item()
            rel = diff / ref_max if ref_max > 0 else 0
            assert (
                rel < 1e-2 or diff < 1e-3
            ), f"[Rank {rank}] replay {i}: rel={rel:.4e} abs={diff:.4e}"

        if rank == 0:
            print(f"  [graph_pure_allreduce] {num_replays} replays passed")
    except Exception as e:
        print(f"[Rank {rank}] FAILED: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        raise
    finally:
        _teardown()


def _worker_graph_fused_rmsnorm(rank, world_size, port, num_replays):
    try:
        _setup(rank, world_size, port)
        from rtp_llm.models_py.modules.base.rocm.trt_allreduce import TrtllmDistEnv

        dev = torch.device(f"cuda:{rank}")
        env = TrtllmDistEnv(group=dist.group.WORLD, device_id=rank)

        torch.manual_seed(42 + rank)
        ar_in = torch.randn(8, 4096, dtype=torch.bfloat16, device=dev)
        res_in = torch.randn(8, 4096, dtype=torch.bfloat16, device=dev)
        w = torch.randn(4096, dtype=torch.bfloat16, device=dev)
        eps = 1e-6

        ref_res, ref_norm, _ = env.allreduce_add_rms_native(
            ar_in.clone(), res_in.clone(), w, eps
        )

        s = torch.cuda.Stream(device=dev)
        with torch.cuda.stream(s):
            env.allreduce_add_rms_fused(ar_in.clone(), res_in.clone(), w, eps)
            s.synchronize()
            g = torch.cuda.CUDAGraph()
            g_ar, g_res = ar_in.clone(), res_in.clone()
            with torch.cuda.graph(g, stream=s):
                g_res_out, g_norm_out, _ = env.allreduce_add_rms_fused(
                    g_ar, g_res, w, eps
                )
        s.synchronize()
        env.consume_capture_if_needed()

        for i in range(num_replays):
            g_ar.copy_(ar_in)
            g_res.copy_(res_in)
            g.replay()
            s.synchronize()

            for name, got, want in [
                ("residual", g_res_out, ref_res),
                ("norm", g_norm_out, ref_norm),
            ]:
                diff = (got - want).abs().max().item()
                mx = want.abs().max().item()
                rel = diff / mx if mx > 0 else 0
                assert (
                    rel < 1e-2 or diff < 1e-3
                ), f"[Rank {rank}] replay {i} {name}: rel={rel:.4e} abs={diff:.4e}"

        if rank == 0:
            print(f"  [graph_fused_rmsnorm] {num_replays} replays passed")
    except Exception as e:
        print(f"[Rank {rank}] FAILED: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        raise
    finally:
        _teardown()


def _worker_collective_torch_tp_capture(rank, world_size, port, num_replays):
    """Exercise the production TP group and native ProcessGroup communicator."""
    capture_state_checker = None
    collective_torch = None
    graph = None

    try:
        _setup(rank, world_size, port)

        from rtp_llm.models_py.distributed import collective_torch, rocm_rccl
        from rtp_llm.ops import NcclCommConfig, ParallelismConfig

        # Prove that externally initialized torch.distributed callers are rebound
        # to their configured local ROCm device before TP groups are created.
        torch.cuda.set_device((rank + 1) % world_size)
        parallelism_config = ParallelismConfig()
        parallelism_config.world_rank = rank
        parallelism_config.world_size = world_size
        parallelism_config.local_rank = rank
        parallelism_config.tp_size = 2
        parallelism_config.dp_size = world_size // parallelism_config.tp_size
        nccl_comm_config = NcclCommConfig(
            nccl_ip="127.0.0.1",
            tp_nccl_port=port + 9,
            dp_tp_nccl_port=port + 1,
            ffn_tp_nccl_port=port + 6,
        )
        collective_torch.init_distributed_environment(
            parallelism_config,
            nccl_comm_config=nccl_comm_config,
            nccl_init_port=port,
            backend="nccl",
            timeout=60,
        )

        assert torch.cuda.current_device() == rank
        tp_group = collective_torch._get_group(collective_torch.Group.TP)
        tp_ranks = dist.get_process_group_ranks(tp_group)
        assert len(tp_ranks) == parallelism_config.tp_size
        assert parallelism_config.dp_size > 1
        assert rocm_rccl._rccl_comm is not None
        assert rocm_rccl._rccl_comm.value is not None
        assert rocm_rccl._rccl_world_size == parallelism_config.tp_size
        assert not rocm_rccl._rccl_comm_owned_by_python

        device = torch.device(f"cuda:{rank}")
        # Warm up the real TP ProcessGroup before stream capture.
        warmup = torch.full((2, 7), rank + 1, dtype=torch.float32, device=device)
        dist.all_reduce(warmup, group=tp_group)
        warmup_gather = torch.empty(
            (parallelism_config.tp_size * 2, 7),
            dtype=torch.float32,
            device=device,
        )
        dist.all_gather_into_tensor(warmup_gather, warmup, group=tp_group)
        torch.cuda.synchronize(device)

        graph_stream = torch.cuda.Stream(device=device)
        graph = torch.cuda.CUDAGraph()
        graph_allreduce_input = torch.empty((2, 7), dtype=torch.float32, device=device)
        graph_allgather_input = torch.empty((2, 7), dtype=torch.float32, device=device)

        # The production capture-state flag is normally owned by C++ graph
        # orchestration. This integration test controls only that flag while
        # exercising the real Python dispatch, RCCL communicator, and kernels.
        capture_state_checker = rocm_rccl._is_hipgraph_capture_active
        rocm_rccl._is_hipgraph_capture_active = lambda: True
        with torch.cuda.stream(graph_stream):
            with torch.cuda.graph(graph, stream=graph_stream):
                graph_allreduce_output = collective_torch.all_reduce(
                    graph_allreduce_input, collective_torch.Group.TP
                )
                graph_allgather_output = collective_torch.all_gather(
                    graph_allgather_input, collective_torch.Group.TP
                )
        graph_stream.synchronize()

        for replay in range(num_replays):
            with torch.cuda.stream(graph_stream):
                graph_allreduce_input.fill_(rank + replay + 1)
                graph_allgather_input.fill_(rank + replay * 10)
                graph.replay()
            graph_stream.synchronize()

            expected_sum = sum(tp_rank + replay + 1 for tp_rank in tp_ranks)
            expected_gather = torch.cat(
                [
                    torch.full(
                        (2, 7),
                        tp_rank + replay * 10,
                        dtype=torch.float32,
                        device=device,
                    )
                    for tp_rank in tp_ranks
                ]
            )
            torch.testing.assert_close(
                graph_allreduce_output,
                torch.full_like(graph_allreduce_output, expected_sum),
            )
            torch.testing.assert_close(graph_allgather_output, expected_gather)

        if rank == 0:
            print(
                "  [collective_torch_tp_capture] "
                f"dp={parallelism_config.dp_size} tp={parallelism_config.tp_size} "
                f"{num_replays} replays passed"
            )
    except Exception as e:
        print(f"[Rank {rank}] FAILED: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        raise
    finally:
        # A captured raw RCCL operation retains the ProcessGroup communicator.
        # Destroy the graph before collective_torch tears that communicator down.
        if graph is not None:
            graph.reset()
        if capture_state_checker is not None:
            from rtp_llm.models_py.distributed import rocm_rccl

            rocm_rccl._is_hipgraph_capture_active = capture_state_checker
        if collective_torch is not None and dist.is_initialized():
            collective_torch.destroy_distributed_environment()
        _teardown()


class TestTrtAllReduceGraphReplay(unittest.TestCase):

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA/ROCm not available")
        if torch.cuda.device_count() < 2:
            self.skipTest("Need >= 2 GPUs")
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    def test_graph_replay_pure_allreduce(self):
        _launch(_worker_graph_pure_allreduce, num_replays=REPLAY_ROUNDS)

    def test_graph_replay_fused_rmsnorm(self):
        _launch(_worker_graph_fused_rmsnorm, num_replays=REPLAY_ROUNDS)

    def test_collective_torch_tp_capture_with_dp_and_external_init(self):
        if torch.cuda.device_count() < 4:
            self.skipTest("Need >= 4 GPUs for dp_size=2 and tp_size=2")
        _launch(
            _worker_collective_torch_tp_capture,
            world_size=4,
            num_replays=3,
        )


if __name__ == "__main__":
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    unittest.main()
