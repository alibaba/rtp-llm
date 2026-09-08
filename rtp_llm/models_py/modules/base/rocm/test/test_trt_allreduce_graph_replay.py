# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import os
import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

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


def _worker_graph_pure_allreduce(
    rank, world_size, port, num_replays, hidden_size=4096, num_tokens=8
):
    try:
        _setup(rank, world_size, port)
        from rtp_llm.models_py.modules.base.rocm.trt_allreduce import TrtllmDistEnv

        dev = torch.device(f"cuda:{rank}")
        env = TrtllmDistEnv(group=dist.group.WORLD, device_id=rank)

        torch.manual_seed(42 + rank)
        inp = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=dev)

        ref = inp.float()
        dist.all_reduce(ref)
        ref = ref.to(inp.dtype)

        s = torch.cuda.Stream(device=dev)
        s.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s):
            eager_in, eager_out = inp.clone(), torch.empty_like(inp)
            env.allreduce_op(eager_in, eager_out)
            s.synchronize()
            torch.testing.assert_close(eager_out, ref, atol=1e-3, rtol=1e-2)
            assert torch.equal(eager_in, inp)
            g = torch.cuda.CUDAGraph()
            g_in, g_out = inp.clone(), torch.empty_like(inp)
            with torch.cuda.graph(g, stream=s):
                env.allreduce_op(g_in, g_out)
        s.synchronize()
        env.consume_capture_if_needed()

        previous_out = g_out.clone()
        for i in range(num_replays):
            replay_in = torch.roll(inp, shifts=i + 1, dims=-1)
            replay_ref = replay_in.float()
            dist.all_reduce(replay_ref)
            replay_ref = replay_ref.to(replay_in.dtype)
            s.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(s):
                g_in.copy_(replay_in)
                g.replay()
            s.synchronize()
            assert torch.equal(g_in, replay_in)
            assert not torch.equal(g_out, previous_out)
            torch.testing.assert_close(g_out, replay_ref, atol=2e-2, rtol=1e-2)
            previous_out.copy_(g_out)

    finally:
        _teardown()


def _worker_graph_fused_rmsnorm(
    rank, world_size, port, num_replays, num_tokens, fp8_out
):
    try:
        _setup(rank, world_size, port)
        from rtp_llm.models_py.modules.base.rocm.trt_allreduce import TrtllmDistEnv

        dev = torch.device(f"cuda:{rank}")
        env = TrtllmDistEnv(group=dist.group.WORLD, device_id=rank)
        torch.manual_seed(42 + rank)
        ar = torch.randn(num_tokens, 4096, dtype=torch.bfloat16, device=dev)
        residual = torch.randn_like(ar)
        weight = torch.randn(4096, dtype=torch.bfloat16, device=dev)
        stream = torch.cuda.Stream(device=dev)
        stream.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(stream):
            env.allreduce_add_rms_fused(
                ar.clone(), residual.clone(), weight, 1e-6, fp8_out
            )
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            graph_ar, graph_residual = ar.clone(), residual.clone()
            with torch.cuda.graph(graph, stream=stream):
                graph_res, graph_norm, graph_scale = env.allreduce_add_rms_fused(
                    graph_ar, graph_residual, weight, 1e-6, fp8_out
                )
        stream.synchronize()
        env.consume_capture_if_needed()

        for i in range(num_replays):
            replay_ar = torch.roll(ar, i + 1, -1)
            replay_residual = torch.roll(residual, i + 1, -1)
            ref_res, ref_norm, ref_scale = env.allreduce_add_rms_native(
                replay_ar.clone(), replay_residual.clone(), weight, 1e-6, fp8_out
            )
            stream.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(stream):
                graph_ar.copy_(replay_ar)
                graph_residual.copy_(replay_residual)
                graph.replay()
            stream.synchronize()
            assert torch.equal(graph_ar, replay_ar)
            assert torch.equal(graph_residual, replay_residual)
            torch.testing.assert_close(graph_res, ref_res, atol=2e-2, rtol=1e-2)
            if fp8_out:
                torch.testing.assert_close(graph_scale, ref_scale, atol=1e-5, rtol=0.1)
                got = graph_norm.float() * graph_scale
                want = ref_norm.float() * ref_scale
                diff = (got - want).abs().max().item()
                assert diff < 0.05 or diff / want.abs().max().item() < 0.1
            else:
                torch.testing.assert_close(graph_norm, ref_norm, atol=2e-2, rtol=1e-2)
    finally:
        _teardown()


class TestTrtAllReduceGraphReplay(unittest.TestCase):

    def setUp(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("Need >= 2 GPUs")
        mp.set_start_method("spawn", force=True)

    def test_graph_replay_pure_allreduce(self):
        _launch(_worker_graph_pure_allreduce, num_replays=REPLAY_ROUNDS)

    def test_graph_replay_pure_allreduce_3072(self):
        for num_tokens in (8, 128):
            with self.subTest(num_tokens=num_tokens):
                _launch(
                    _worker_graph_pure_allreduce,
                    num_replays=REPLAY_ROUNDS,
                    hidden_size=3072,
                    num_tokens=num_tokens,
                )

    def test_graph_replay_pure_allreduce_3072_tp4(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 4)
        for num_tokens in (8, 128):
            with self.subTest(num_tokens=num_tokens):
                _launch(
                    _worker_graph_pure_allreduce,
                    world_size=4,
                    num_replays=10,
                    hidden_size=3072,
                    num_tokens=num_tokens,
                )

    def test_graph_replay_fused_rmsnorm(self):
        for num_tokens in (8, 128):
            for fp8_out in (False, True):
                with self.subTest(num_tokens=num_tokens, fp8_out=fp8_out):
                    _launch(
                        _worker_graph_fused_rmsnorm,
                        num_replays=4,
                        num_tokens=num_tokens,
                        fp8_out=fp8_out,
                    )


if __name__ == "__main__":
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    unittest.main()
