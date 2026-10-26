"""Exercise the actual fastsafetensors private API with one and two GPUs."""

import datetime
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import fastsafetensors
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file

from rtp_llm.model_loader.per_expert_parallel_loader import PerExpertParallelLoader


def fixture(shard):
    return {
        f"layer{shard}.experts": (
            torch.arange(96, dtype=torch.bfloat16).reshape(4, 3, 8) + shard * 8
        ),
        f"layer{shard}.dense": torch.arange(16, dtype=torch.bfloat16) + shard,
    }


def verify_loader(rank, world_size, directory, use_shm):
    torch.cuda.set_device(rank)
    if world_size > 1:
        dist.init_process_group(
            "nccl",
            init_method=f"file://{directory}/rendezvous",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=120),
        )
        pg = dist.group.WORLD
    else:
        pg = fastsafetensors.SingleGroup()
    loader = None
    try:
        loader = PerExpertParallelLoader(
            {f"layer{s}.experts": f"layer{s}.expert{{expert_id}}" for s in range(4)},
            pg=pg,
            hf_weights_files=[f"{directory}/shard{s}.safetensors" for s in range(4)],
            device=f"cuda:{rank}",
            use_tqdm_on_load=False,
            use_shm=use_shm,
            nogds=not use_shm,
            bbuf_size_kb=1024,
            set_numa=False,
        )
        actual = {}
        for key, tensor in loader.iterate_weights():
            if key in actual:
                raise AssertionError(f"duplicate tensor: {key}")
            # Keep outputs through all batches and buffer reclamation.
            actual[key] = tensor.clone()
        expected = {}
        for shard in range(4):
            tensors = fixture(shard)
            expected[f"layer{shard}.dense"] = tensors[f"layer{shard}.dense"]
            for expert in range(4):
                expected[f"layer{shard}.expert{expert}"] = tensors[
                    f"layer{shard}.experts"
                ][expert]
        assert actual.keys() == expected.keys(), (actual.keys(), expected.keys())
        for key, tensor in actual.items():
            torch.testing.assert_close(tensor.cpu(), expected[key], rtol=0, atol=0)
        print(
            f"PER_EXPERT_ACCEPT version={fastsafetensors.__version__} "
            f"world_size={world_size} rank={rank} use_shm={use_shm} "
            f"tensors={len(actual)} exact=True",
            flush=True,
        )
    finally:
        if loader is not None:
            loader.close()
        if world_size > 1:
            dist.destroy_process_group()


class PerExpertParallelLoaderTest(unittest.TestCase):
    def run_loader(self, world_size, use_shm):
        self.assertGreaterEqual(torch.cuda.device_count(), world_size)
        with tempfile.TemporaryDirectory() as directory:
            for shard in range(4):
                save_file(
                    fixture(shard), str(Path(directory) / f"shard{shard}.safetensors")
                )
            if world_size == 1:
                verify_loader(0, world_size, directory, use_shm)
            else:
                mp.spawn(
                    verify_loader,
                    args=(world_size, directory, use_shm),
                    nprocs=world_size,
                    join=True,
                )

    def test_single_rank(self):
        self.run_loader(1, True)

    def test_two_ranks_shm(self):
        self.run_loader(2, True)

    def test_two_ranks_nogds(self):
        self.run_loader(2, False)

    def test_unknown_version_rejected(self):
        with mock.patch.object(fastsafetensors, "__version__", "0.1.21+ali"):
            with self.assertRaisesRegex(RuntimeError, "current version: 0.1.21"):
                PerExpertParallelLoader({})


if __name__ == "__main__":
    unittest.main()
