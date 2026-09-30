import tempfile
import unittest
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.modules.base.common.embedding import EmbeddingTorch


def _check_tp_rank(rank, rendezvous):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        # Isolate the test's TP group from service/global group initialization;
        # the production all_gather implementation still executes real NCCL.
        with patch.object(
            collective_torch, "_get_group", return_value=dist.group.WORLD
        ):
            for width in (64, 4096):
                torch.manual_seed(17)
                weight = torch.randn(41, width, dtype=torch.bfloat16, device="cuda")
                shard = weight[
                    :, rank * (width // 2) : (rank + 1) * (width // 2)
                ].contiguous()
                embedding = EmbeddingTorch(shard, tp_size=2)
                assert embedding.weight is shard
                for shape in ((1,), (6,), (2, 3), (2, 2, 3)):
                    ids = torch.arange(12, device="cuda")[
                        : torch.tensor(shape).prod().item()
                    ].reshape(shape)
                    actual = embedding(ids)
                    torch.testing.assert_close(
                        actual, F.embedding(ids, weight), rtol=0, atol=0
                    )
                    assert actual.shape == (*shape, width)

                ids = torch.arange(6, device="cuda")
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        embedding(ids)
                torch.cuda.current_stream().wait_stream(stream)
                dist.barrier()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = embedding(ids)
                for offset in (7, 19):
                    ids.copy_(torch.arange(6, device="cuda") + offset)
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        captured, F.embedding(ids, weight), rtol=0, atol=0
                    )
                # NCCL graph user objects retain the communicator. Release
                # captured graphs before destroy_process_group can finalize it.
                del captured, graph
                torch.cuda.synchronize()
    finally:
        dist.destroy_process_group()


class EmbeddingTPTest(unittest.TestCase):
    def test_tp1_preserves_layout_without_collective(self):
        weight = torch.randn(23, 16)
        embedding = EmbeddingTorch(weight)
        with patch(
            "rtp_llm.models_py.modules.base.common.embedding.all_gather"
        ) as gather:
            for shape in ((), (6,), (2, 3), (1, 2, 3)):
                ids = torch.zeros(shape, dtype=torch.long)
                torch.testing.assert_close(
                    embedding(ids), F.embedding(ids, weight), rtol=0, atol=0
                )
            gather.assert_not_called()
        self.assertIs(embedding.weight, weight)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA GPUs")
    def test_tp2_hidden_shards_and_cuda_graph(self):
        with tempfile.TemporaryDirectory(prefix="embedding-tp-") as directory:
            mp.spawn(
                _check_tp_rank,
                args=("file://" + directory + "/rendezvous",),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
