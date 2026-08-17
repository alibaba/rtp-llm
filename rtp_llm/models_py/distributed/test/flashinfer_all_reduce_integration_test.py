# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import os
import unittest

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed.flashinfer_all_reduce import FlashInferAllReduce
from rtp_llm.test.utils.port_util import PortManager


def _worker(rank: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["FT_DISABLE_CUSTOM_AR"] = "0"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2)
    communicator = FlashInferAllReduce(
        dist.group.WORLD, torch.device("cuda", rank), single_node=True
    )
    assert not communicator.disabled

    try:
        # Compare the eager fast path with NCCL for the shapes that bracket
        # Qwen3.5 target decode traffic.
        for tokens in (1, 4, 16, 128):
            base = torch.arange(
                tokens * 5120, device="cuda", dtype=torch.float32
            ).reshape(tokens, 5120)
            input_tensor = (
                ((base.remainder(251) - 125) / 64).to(torch.bfloat16)
                * (rank + 1)
            )
            reference = input_tensor.clone()
            dist.all_reduce(reference)
            assert communicator.should_use(input_tensor)
            output = communicator.all_reduce(input_tensor)
            torch.cuda.synchronize()
            torch.testing.assert_close(output, reference, rtol=0, atol=0)

        # Capture once and replay with changing inputs.  This catches stale
        # Lamport state/output buffers, the class of bug most relevant to the
        # steady-state decode graphs.
        static_input = torch.full(
            (16, 5120), rank + 1, device="cuda", dtype=torch.bfloat16
        )
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                communicator.all_reduce(static_input)
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()
        dist.barrier()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            graph_output = communicator.all_reduce(static_input)

        for value in (2, 5, -3, 7, 11):
            static_input.fill_(value * (rank + 1))
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            expected = torch.full_like(graph_output, value * 3)
            torch.testing.assert_close(graph_output, expected, rtol=0, atol=0)
    finally:
        communicator.destroy()
        dist.destroy_process_group()


class FlashInferAllReduceIntegrationTest(unittest.TestCase):
    def test_tp2_eager_and_graph_replay_match_nccl(self):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest("requires two CUDA GPUs")

        port_manager = PortManager()
        ports, locks = port_manager.get_consecutive_ports(1)
        try:
            context = mp.get_context("spawn")
            processes = [
                context.Process(target=_worker, args=(rank, ports[0]))
                for rank in range(2)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=120)
                self.assertEqual(process.exitcode, 0)
        finally:
            for lock in locks:
                lock.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
