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
    errors = []
    try:
        dist.barrier()
        torch.cuda.synchronize()
    except Exception as exc:
        errors.append(exc)
        print(f"Distributed drain failed during teardown: {exc}", file=sys.stderr)
    try:
        dist.destroy_process_group()
    except Exception as exc:
        errors.append(exc)
        print(
            f"Process-group destruction failed during teardown: {exc}", file=sys.stderr
        )
    if errors:
        raise RuntimeError(f"Distributed teardown failed: {errors}") from errors[0]


def _shutdown_and_verify(env):
    handle = env.handle
    env.shutdown()
    if env.handle is not None or not env.disabled:
        raise RuntimeError("TRT workspace remained live after collective shutdown")
    try:
        handle.get_barrier_handle()
    except RuntimeError as exc:
        if "after workspace teardown" not in str(exc):
            raise
    else:
        raise RuntimeError("Native TRT workspace accepted access after teardown")


def _finalize_worker(env):
    primary_error = sys.exc_info()[1]
    cleanup_error = None
    try:
        if env is not None:
            _shutdown_and_verify(env)
    except Exception as exc:
        cleanup_error = exc
        print(f"TRT workspace cleanup failed: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
    try:
        _teardown()
    except Exception as exc:
        if cleanup_error is None:
            cleanup_error = exc
    if cleanup_error is not None and primary_error is None:
        raise cleanup_error


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
    env = None
    try:
        _setup(rank, world_size, port)
        from rtp_llm.models_py.modules.base.rocm.trt_allreduce import TrtllmDistEnv

        dev = torch.device(f"cuda:{rank}")
        control_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
        env = TrtllmDistEnv(
            group=dist.group.WORLD, control_group=control_group, device_id=rank
        )

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
        shutdown_env, env = env, None
        _shutdown_and_verify(shutdown_env)
    except Exception as e:
        print(f"[Rank {rank}] FAILED: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        raise
    finally:
        _finalize_worker(env)


def _worker_graph_fused_rmsnorm(rank, world_size, port, num_replays):
    env = None
    try:
        _setup(rank, world_size, port)
        from rtp_llm.models_py.modules.base.rocm.trt_allreduce import TrtllmDistEnv

        dev = torch.device(f"cuda:{rank}")
        control_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
        env = TrtllmDistEnv(
            group=dist.group.WORLD, control_group=control_group, device_id=rank
        )

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
        shutdown_env, env = env, None
        _shutdown_and_verify(shutdown_env)
    except Exception as e:
        print(f"[Rank {rank}] FAILED: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        raise
    finally:
        _finalize_worker(env)


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
