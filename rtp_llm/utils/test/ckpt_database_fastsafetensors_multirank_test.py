# SPDX-License-Identifier: Apache-2.0

import datetime
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file

from rtp_llm.utils.database import (
    FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
    CkptDatabase,
)


class _CheckpointFile:
    def __init__(self, file_name: str) -> None:
        self.file_name = file_name


def _run_real_fastsafetensors_rank(
    rank: int, checkpoint_path: str, init_path: str, result_dir: str
) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(minutes=2),
    )
    try:
        # Exercise the pinned wrapper/native boundary, including FuseShm
        # chunking, pre-broadcast dim0 split, rank-local copyout and bucket
        # broadcast. Tiny limits make the intended path explicit while keeping
        # the integration test cheap.
        os.environ["FASTSAFETENSORS_NOGDS"] = "0"
        os.environ.pop("FASTSAFETENSORS_CONFIG", None)
        os.environ["FASTSAFETENSORS_CONFIG_JSON"] = json.dumps(
            {
                "loader": "fuse-shm",
                "framework": "pytorch",
                "set_numa": False,
                "disable_cache": True,
                "use_pipeline": True,
                "max_concurrent_producers": 1,
                "queue_size": 0,
                "use_tqdm_on_load": False,
                "chunked_loading": True,
                "max_batch_bytes": 64 * 1024,
                "max_io_chunk_bytes": 64 * 1024,
                "broadcast_bucket": True,
                "max_broadcast_bucket_bytes": 64 * 1024,
                "max_broadcast_tensor_bytes": 32 * 1024,
                "local_rank_copy": True,
                "fuse-shm": {
                    "bbuf_size_kb": 64,
                    "direct_io": False,
                    "require_fuse_shm": True,
                },
            }
        )

        import fastsafetensors

        real_auto_loader = fastsafetensors.AutoLoader

        class TrackingAutoLoader(real_auto_loader):
            close_calls = 0

            def close(self) -> None:
                type(self).close_calls += 1
                super().close()

        wanted_keys = (
            {"direct", "experts.0.weight"} if rank == 0 else {"experts.1.weight"}
        )
        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_CheckpointFile(checkpoint_path)]

        with patch.object(fastsafetensors, "AutoLoader", TrackingAutoLoader):
            outputs = dict(
                database.fastsafetensors_weights_iterator(
                    "cuda",
                    stacked_key_config={"stacked": "experts.{expert_id}.weight"},
                    local_copyout_filter=wanted_keys.__contains__,
                    stacked_moe_mode=FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
                )
            )

        # The generator has completed and closed the real loader. Returned
        # tensors must still own valid storage after that close.
        torch.cuda.synchronize(rank)
        result = {
            "close_calls": TrackingAutoLoader.close_calls,
            "keys": sorted(outputs),
            "values": {
                key: tensor.detach().cpu().tolist() for key, tensor in outputs.items()
            },
        }
        with open(os.path.join(result_dir, f"rank-{rank}.json"), "w") as writer:
            json.dump(result, writer, sort_keys=True)
    finally:
        dist.destroy_process_group()


class InstalledFastsafetensorsMultiRankTest(unittest.TestCase):
    def test_real_two_rank_split_filter_broadcast_and_close(self) -> None:
        self.assertGreaterEqual(
            torch.cuda.device_count(),
            2,
            "Bazel target requires two H20 GPUs",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "model.safetensors")
            save_file(
                {
                    "direct": torch.tensor([90.0, 91.0]),
                    "stacked": torch.tensor(
                        [
                            [[1.0, 2.0], [3.0, 4.0]],
                            [[5.0, 6.0], [7.0, 8.0]],
                        ]
                    ),
                },
                checkpoint_path,
            )
            mp.spawn(
                _run_real_fastsafetensors_rank,
                args=(
                    checkpoint_path,
                    os.path.join(tmp_dir, "nccl-init"),
                    tmp_dir,
                ),
                nprocs=2,
                join=True,
            )

            with open(os.path.join(tmp_dir, "rank-0.json")) as reader:
                rank0 = json.load(reader)
            with open(os.path.join(tmp_dir, "rank-1.json")) as reader:
                rank1 = json.load(reader)

        self.assertEqual(rank0["close_calls"], 1)
        self.assertEqual(rank1["close_calls"], 1)
        self.assertEqual(rank0["keys"], ["direct", "experts.0.weight"])
        self.assertEqual(rank1["keys"], ["experts.1.weight"])
        self.assertEqual(rank0["values"]["direct"], [90.0, 91.0])
        self.assertEqual(
            rank0["values"]["experts.0.weight"],
            [[1.0, 2.0], [3.0, 4.0]],
        )
        self.assertEqual(
            rank1["values"]["experts.1.weight"],
            [[5.0, 6.0], [7.0, 8.0]],
        )


if __name__ == "__main__":
    unittest.main()
