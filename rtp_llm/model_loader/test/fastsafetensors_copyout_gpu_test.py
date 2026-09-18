"""Real CUDA13 wheel/ABI and two-rank selected-copyout contracts."""

import datetime
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file

from rtp_llm.utils.database import CkptDatabase

_CONFIG = {
    "loader": "fuse-shm-v2",
    "chunked_loading": True,
    "max_batch_bytes": 1024 * 1024,
    "max_io_chunk_bytes": 1024 * 1024,
    "max_broadcast_bucket_bytes": 1024 * 1024,
    "max_broadcast_tensor_bytes": 512 * 1024,
    "set_numa": False,
    "use_tqdm_on_load": False,
    "fuse-shm": {"bbuf_size_kb": 1024, "direct_io": False},
}


def _rank_load(rank, directory, rendezvous):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [
            SimpleNamespace(file_name=str(Path(directory) / "weights.safetensors"))
        ]
        expected = torch.arange(4 * 64 * 64, dtype=torch.float32).reshape(4, 64, 64)
        # A rank may require no tensors while still participating in every collective.
        for empty_rank in (False, True):
            wanted = (
                set()
                if empty_rank and rank == 1
                else {"shared", "experts.%d.weight" % rank}
            )
            tensors = dict(
                database.fastsafetensors_weights_iterator(
                    "cuda:%d" % rank,
                    False,
                    {"stacked": "experts.{expert_id}.weight"},
                    wanted.__contains__,
                )
            )
            assert set(tensors) == wanted, (rank, set(tensors), wanted)
            if wanted:
                torch.testing.assert_close(
                    tensors["shared"].cpu(), torch.arange(128, dtype=torch.float32)
                )
                torch.testing.assert_close(
                    tensors["experts.%d.weight" % rank].cpu(), expected[rank]
                )
                # Resident copyouts survive iterator close and are independently writable.
                tensors["experts.%d.weight" % rank].add_(1)
                torch.testing.assert_close(
                    tensors["shared"].cpu(), torch.arange(128, dtype=torch.float32)
                )
            del tensors
            dist.barrier()
    finally:
        dist.destroy_process_group()


class FastsafetensorsCopyoutGpuTest(unittest.TestCase):
    def test_two_rank_stacked_copyout_and_empty_rank(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        with tempfile.TemporaryDirectory() as directory:
            save_file(
                {
                    "shared": torch.arange(128, dtype=torch.float32),
                    "stacked": torch.arange(4 * 64 * 64, dtype=torch.float32).reshape(
                        4, 64, 64
                    ),
                    "unused": torch.zeros(128),
                },
                str(Path(directory) / "weights.safetensors"),
            )
            with patch.dict(
                os.environ,
                {
                    "FASTSAFETENSORS_CONFIG_JSON": json.dumps(_CONFIG),
                    "FASTSAFETENSORS_NOGDS": "0",
                },
            ):
                context = mp.spawn(
                    _rank_load,
                    args=(directory, "file://" + directory + "/rendezvous"),
                    nprocs=2,
                    join=False,
                )
                deadline = time.monotonic() + 90
                try:
                    while not context.join(timeout=1):
                        if time.monotonic() >= deadline:
                            self.fail(
                                "two-rank copyout did not finish within 90 seconds"
                            )
                finally:
                    for process in context.processes:
                        if process.is_alive():
                            process.terminate()
                        process.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
