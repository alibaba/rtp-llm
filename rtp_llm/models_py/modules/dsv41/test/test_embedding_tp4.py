"""Real TP4 embedding lookup and graph replay for target/draft token layouts."""

import faulthandler
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, rendezvous):
    from rtp_llm.models_py.distributed import collective_torch as collective
    from rtp_llm.models_py.modules.dsv41.utils import V41Embedding

    faulthandler.dump_traceback_later(60, repeat=False)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=60),
    )
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=60))
    collective._group_map[collective.Group.TP] = dist.group.WORLD
    collective._group_map[collective.Group.DP_AND_TP] = dist.group.WORLD
    collective._parallelism_config = SimpleNamespace(tp_size=4, dp_size=1, world_size=4)
    collective._initialized = True
    try:
        weight = (
            torch.arange(64 * 128, device="cuda", dtype=torch.float32)
            .reshape(64, 128)
            .to(torch.bfloat16)
        )
        embedding = V41Embedding(
            None,
            SimpleNamespace(get_attn_tp_size=lambda: 4),
            weight.chunk(4, dim=-1)[rank].contiguous(),
        )
        for shape in ((4,), (2, 3), (2, 1, 3)):
            ids = torch.arange(6 if len(shape) > 1 else 4, device="cuda").reshape(shape)
            reference = torch.nn.functional.embedding(ids, weight)
            torch.testing.assert_close(embedding(ids), reference, rtol=0, atol=0)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    embedding(ids)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            dist.barrier(group=control)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = embedding(ids)
            dist.barrier(group=control)
            ids.add_(7)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                actual, torch.nn.functional.embedding(ids, weight), rtol=0, atol=0
            )
            dist.barrier(group=control)
            graph.reset()
            del graph
        dist.barrier(group=control)
    finally:
        collective._group_map.clear()
        collective._initialized = False
        dist.destroy_process_group()
        faulthandler.cancel_dump_traceback_later()


class EmbeddingTP4Test(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.device_count() >= 4, "requires four CUDA devices")
    def test_full_hidden_width_and_live_graph_token_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            mp.spawn(
                _worker,
                args=(str(Path(temporary) / "rendezvous"),),
                nprocs=4,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
