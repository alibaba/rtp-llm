# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed import pynccl_cp


def _assert_equal(output: torch.Tensor, expected: torch.Tensor) -> None:
    # Compare raw bytes, including FP8 formats without CPU arithmetic support.
    torch.testing.assert_close(
        output.cpu().view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
    )


def _assert_initialized(rank: int, group) -> None:
    # This target runs on H20 or Blackwell with NCCL symmetric-memory support.
    # An ordinary-pynccl fallback would not exercise persistent role buffers.
    roles = ("indexer_k_bf16", "mla_ckv", "mla_kpe", "mla_history")
    assert set(pynccl_cp._COMMS) == {(group, rank)}
    assert set(pynccl_cp._SYMM_BASES) == {
        (group, rank, role, torch.bfloat16) for role in roles
    }
    for base in pynccl_cp._SYMM_BASES.values():
        assert (group, rank, base.data_ptr()) in pynccl_cp._SYMM_WINDOWS


def _check_eager(rank: int, group) -> None:
    for dtype in (torch.bfloat16, torch.float32, torch.uint8, torch.float8_e4m3fn):
        for rows in (0, 1, 17, 3):
            base = torch.arange(rows * 16).reshape(rows, 16) % 16
            expected = torch.cat((base, base + 32)).to(dtype)
            input_tensor = (base + rank * 32).to(device="cuda", dtype=dtype)
            output = pynccl_cp.all_gather(input_tensor, group=group)
            _assert_equal(output, expected)

    # cKV and kPE are consumed together. Neither may overwrite the other;
    # shrinking and then growing the request must not expose stale rows.
    for rows in (17, 1, 0, 33):
        input_tensor = torch.full(
            (rows, 16), rank + 1, device="cuda", dtype=torch.bfloat16
        )
        ckv = pynccl_cp.all_gather(input_tensor, group=group, role="mla_ckv")
        kpe = pynccl_cp.all_gather(input_tensor * 10, group=group, role="mla_kpe")
        expected = torch.cat(
            [torch.full((rows, 16), value, dtype=torch.bfloat16) for value in (1, 2)]
        )
        _assert_equal(ckv, expected)
        _assert_equal(kpe, expected * 10)
        if rows:
            assert ckv.data_ptr() != kpe.data_ptr()

    # A second stream must not reuse a persistent output still owned by the
    # first stream. Explicit-stream submission must also preserve dependencies.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        alternate = pynccl_cp.all_gather(input_tensor, group=group, role="mla_ckv")
        alternate.add_(10)
    stream.synchronize()
    assert alternate.data_ptr() != ckv.data_ptr()
    _assert_equal(ckv, expected)
    _assert_equal(alternate, expected + 10)

    output = torch.empty_like(ckv)
    pynccl_cp.all_gather_into_tensor(output, input_tensor, group=group, stream=stream)
    torch.cuda.current_stream().wait_stream(stream)
    _assert_equal(output, expected)


def _check_graph(rank: int, group) -> None:
    input_tensor = torch.full((7, 16), rank + 1, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            pynccl_cp.all_gather(input_tensor, group=group, role="mla_history")
    stream.synchronize()
    dist.barrier(group=group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = pynccl_cp.all_gather(input_tensor, group=group, role="mla_history")
        # Consume the collective result on the capture stream as model callers do.
        consumed = output + 1
    for value in (2, -3, 5):
        input_tensor.fill_(value * (rank + 1))
        graph.replay()
        expected = torch.cat(
            [
                torch.full((7, 16), value * peer + 1, dtype=torch.bfloat16)
                for peer in (1, 2)
            ]
        )
        _assert_equal(consumed, expected)
    # Release graph executables before destroying their NCCL communicator.
    graph.reset()


def _worker(rank: int, rendezvous: str) -> None:
    torch.cuda.set_device(rank)
    assert pynccl_cp.enabled(), "run with GLM5_CP_OPT=1"
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    group = dist.group.WORLD
    try:
        pynccl_cp.init(group, rank)
        _assert_initialized(rank, group)
        _check_eager(rank, group)
        _check_graph(rank, group)
        torch.cuda.synchronize()
        dist.barrier(group=group)
        pynccl_cp.destroy()
        for cache in (
            pynccl_cp._COMMS,
            pynccl_cp._SYMM_BASES,
            pynccl_cp._SYMM_VIEWS,
            pynccl_cp._SYMM_WINDOWS,
            pynccl_cp._SYMM_ROLE_STREAMS,
        ):
            assert not cache, "destroy() must release all communicator/window state"
        # A fresh server lifecycle must rebuild both communicators and windows.
        pynccl_cp.init(group, rank)
        _assert_initialized(rank, group)
        _check_eager(rank, group)
        torch.cuda.synchronize()
        dist.barrier(group=group)
    finally:
        pynccl_cp.destroy()
        dist.destroy_process_group()


class PyncclCPIntegrationTest(unittest.TestCase):
    def test_cp2_eager_graph_and_reinitialization(self):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest("requires two CUDA GPUs")
        if any(
            torch.cuda.get_device_capability(rank)[0] not in (9, 10) for rank in (0, 1)
        ):
            self.skipTest("requires Hopper or Blackwell symmetric-memory support")

        context = mp.get_context("spawn")
        processes = []
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = (Path(directory) / "rendezvous").as_uri()
            try:
                for rank in range(2):
                    process = context.Process(target=_worker, args=(rank, rendezvous))
                    process.start()
                    processes.append(process)
                deadline = time.monotonic() + 180
                for process in processes:
                    process.join(timeout=max(0, deadline - time.monotonic()))
                self.assertEqual([process.exitcode for process in processes], [0, 0])
            finally:
                # A failing peer can strand a rank inside NCCL; reap only the
                # children owned by this test, including on timeout/assertion.
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                for process in processes:
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
