"""Verify TP2 target reductions against NCCL, including shared-buffer replay."""

import datetime
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, store):
    from rtp_llm.models_py.distributed.collective_torch import Group
    from rtp_llm.models_py.modules.qwen4_exp.tp_reduce import Qwen4TargetReduce

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method="file://" + store,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=120),
    )
    reducer = Qwen4TargetReduce(SimpleNamespace(group=dist.group.WORLD, device=device))
    reducer.active = True
    try:
        for rows, layers in ((33, 2), (64, 96), (256, 4)):
            torch.manual_seed(20261113 + rank)
            source = torch.randn(rows, 2560, device=device, dtype=torch.bfloat16)

            def forward(use_ipc):
                value = source
                outputs = []
                for layer in range(layers):
                    local = value * 0.25 + (rank + 1) * (layer % 3 + 1) / 16
                    if use_ipc:
                        value = reducer(local, Group.TP)
                    else:
                        value = local.clone()
                        dist.all_reduce(value)
                    outputs.append(value)
                return outputs

            for _ in range(3):
                forward(True)
                forward(False)
            torch.cuda.synchronize()
            ipc_graph, reference_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            with torch.cuda.graph(ipc_graph):
                actual = forward(True)
            reducer.finish_graph_capture()
            with torch.cuda.graph(reference_graph):
                expected = forward(False)
            try:
                for generation in range(8):
                    source.normal_(mean=(rank * 2 - 1) * generation / 10)
                    if generation % 2:
                        source[-7:].zero_()
                    saved_source = source.clone()
                    reference_graph.replay()
                    ipc_graph.replay()
                    torch.cuda.synchronize()
                    for got, want in zip(actual, expected):
                        torch.testing.assert_close(got, want, rtol=0, atol=0)
                    torch.testing.assert_close(source, saved_source, rtol=0, atol=0)
            finally:
                ipc_graph.reset()
                reference_graph.reset()
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        reducer.workspace.destroy()
        reducer.close()
        dist.destroy_process_group()


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.device_count() >= 2,
    "requires two CUDA GPUs",
)
class TpTargetReduceIntegrationTest(unittest.TestCase):
    def test_changing_inputs_and_96_layer_graph_match_nccl(self):
        if any(torch.cuda.get_device_capability(i) != (12, 0) for i in range(2)):
            self.skipTest("requires SM120")
        if not torch.cuda.can_device_access_peer(0, 1):
            self.skipTest("requires direct peer access")
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _worker, args=(str(Path(directory) / "store"),), nprocs=2, join=True
            )


if __name__ == "__main__":
    unittest.main()
