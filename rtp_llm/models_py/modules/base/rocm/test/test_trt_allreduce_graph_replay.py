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

warnings.filterwarnings("ignore", message="barrier.*using the device under current context")
warnings.filterwarnings("ignore", message="Guessing device ID based on global rank")

REPLAY_ROUNDS = 60


def _setup(rank, world_size, port):
    os.environ.update(MASTER_ADDR="localhost", MASTER_PORT=str(port),
                      RANK=str(rank), WORLD_SIZE=str(world_size))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method="env://", world_size=world_size,
                            rank=rank, device_id=torch.device(f"cuda:{rank}"))


def _teardown():
    try:
        dist.barrier(); torch.cuda.synchronize()
    except Exception:
        pass
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0)); return s.getsockname()[1]


def _launch(fn, world_size=2, timeout=180, **kw):
    port = _free_port()
    procs = []
    for r in range(world_size):
        p = mp.Process(target=fn, args=(r, world_size, port), kwargs=kw, name=f"rank-{r}")
        p.start(); procs.append(p)
    for p in procs:
        p.join(timeout=timeout)
        if p.exitcode != 0:
            raise RuntimeError(f"{p.name} exited with code {p.exitcode}")


def _worker_graph_pure_allreduce(rank, world_size, port, num_replays):
    try:
        _setup(rank, world_size, port)
        from rtp_llm.models_py.modules.base.rocm.trt_allreduce import TrtllmDistEnv

        dev = torch.device(f"cuda:{rank}")
        env = TrtllmDistEnv(group=dist.group.WORLD, device_id=rank)

        torch.manual_seed(42 + rank)
        inp = torch.randn(8, 4096, dtype=torch.bfloat16, device=dev)

        ref = inp.clone(); dist.all_reduce(ref)

        s = torch.cuda.Stream(device=dev)
        with torch.cuda.stream(s):
            env.allreduce_op(inp.clone(), torch.empty_like(inp)); s.synchronize()
            g = torch.cuda.CUDAGraph()
            g_in, g_out = inp.clone(), torch.empty_like(inp)
            with torch.cuda.graph(g, stream=s):
                env.allreduce_op(g_in, g_out)
        s.synchronize(); env.consume_capture_if_needed()

        for i in range(num_replays):
            g_in.copy_(inp); g.replay(); s.synchronize()
            diff = (g_out - ref).abs().max().item()
            ref_max = ref.abs().max().item()
            rel = diff / ref_max if ref_max > 0 else 0
            assert rel < 1e-2 or diff < 1e-3, \
                f"[Rank {rank}] replay {i}: rel={rel:.4e} abs={diff:.4e}"

        if rank == 0:
            print(f"  [graph_pure_allreduce] {num_replays} replays passed")
    except Exception as e:
        print(f"[Rank {rank}] FAILED: {e}", file=sys.stderr)
        import traceback; traceback.print_exc(); raise
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
            ar_in.clone(), res_in.clone(), w, eps)

        s = torch.cuda.Stream(device=dev)
        with torch.cuda.stream(s):
            env.allreduce_add_rms_fused(ar_in.clone(), res_in.clone(), w, eps)
            s.synchronize()
            g = torch.cuda.CUDAGraph()
            g_ar, g_res = ar_in.clone(), res_in.clone()
            with torch.cuda.graph(g, stream=s):
                g_res_out, g_norm_out, _ = env.allreduce_add_rms_fused(
                    g_ar, g_res, w, eps)
        s.synchronize(); env.consume_capture_if_needed()

        for i in range(num_replays):
            g_ar.copy_(ar_in); g_res.copy_(res_in)
            g.replay(); s.synchronize()

            for name, got, want in [("residual", g_res_out, ref_res),
                                    ("norm", g_norm_out, ref_norm)]:
                diff = (got - want).abs().max().item()
                mx = want.abs().max().item()
                rel = diff / mx if mx > 0 else 0
                assert rel < 1e-2 or diff < 1e-3, \
                    f"[Rank {rank}] replay {i} {name}: rel={rel:.4e} abs={diff:.4e}"

        if rank == 0:
            print(f"  [graph_fused_rmsnorm] {num_replays} replays passed")
    except Exception as e:
        print(f"[Rank {rank}] FAILED: {e}", file=sys.stderr)
        import traceback; traceback.print_exc(); raise
    finally:
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


if __name__ == "__main__":
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    unittest.main()
