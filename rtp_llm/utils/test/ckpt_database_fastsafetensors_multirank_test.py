# SPDX-License-Identifier: Apache-2.0

import datetime
import json
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file

# The production entry installs the UE8M0 dist.broadcast compatibility shim.
import rtp_llm.ops as _rtp_ops  # noqa: F401
from rtp_llm.utils.database import (
    FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
    CkptDatabase,
)

_CHUNKED_TENSOR_NUMEL = 20 * 1024  # 80 KiB in float32, above the 64 KiB limits.
_UE8M0_RAW_BYTES = [0, 1, 2, 63, 127, 128, 200, 254]


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
                },
            }
        )

        wanted_keys = (
            {"direct", "experts.0.weight", "ue8m0_scale"}
            if rank == 0
            else {"chunked", "experts.1.weight", "ue8m0_scale"}
        )
        database = object.__new__(CkptDatabase)
        database.pretrain_file_list = [_CheckpointFile(checkpoint_path)]

        outputs = dict(
            database.fastsafetensors_weights_iterator(
                "cuda",
                stacked_key_config={"stacked": "experts.{expert_id}.weight"},
                local_copyout_filter=wanted_keys.__contains__,
                stacked_moe_mode=FASTSAFETENSORS_STACKED_MOE_MODE_PER_EXPERT,
            )
        )

        # Exhausting the production generator executes its finally/close path.
        # Returned tensors must still own valid storage afterwards. Exact close
        # invocation is covered separately by the iterator failure-path unit
        # tests; this integration boundary intentionally patches no package API.
        torch.cuda.synchronize(rank)
        chunked = outputs.get("chunked")
        chunked_cpu = chunked.detach().cpu() if chunked is not None else None
        ue8m0_cpu = outputs["ue8m0_scale"].detach().cpu()
        result = {
            "keys": sorted(outputs),
            "values": {
                key: tensor.detach().cpu().tolist()
                for key, tensor in outputs.items()
                if key not in {"chunked", "ue8m0_scale"}
            },
            "chunked": (
                {
                    "numel": chunked_cpu.numel(),
                    "matches": torch.equal(
                        chunked_cpu,
                        torch.arange(_CHUNKED_TENSOR_NUMEL, dtype=torch.float32),
                    ),
                }
                if chunked_cpu is not None
                else None
            ),
            "ue8m0": {
                "dtype": str(ue8m0_cpu.dtype),
                "raw_bytes": ue8m0_cpu.view(torch.uint8).tolist(),
            },
        }
        with open(os.path.join(result_dir, f"rank-{rank}.json"), "w") as writer:
            json.dump(result, writer, sort_keys=True)
    finally:
        dist.destroy_process_group()


class InstalledFastsafetensorsMultiRankTest(unittest.TestCase):
    def test_real_two_rank_split_filter_broadcast_and_storage_ownership(self) -> None:
        self.assertGreaterEqual(
            torch.cuda.device_count(),
            2,
            "Bazel target requires two CUDA GPUs",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "model.safetensors")
            save_file(
                {
                    "direct": torch.tensor([90.0, 91.0]),
                    "chunked": torch.arange(_CHUNKED_TENSOR_NUMEL, dtype=torch.float32),
                    "ue8m0_scale": torch.tensor(
                        _UE8M0_RAW_BYTES, dtype=torch.uint8
                    ).view(torch.float8_e8m0fnu),
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

        self.assertEqual(rank0["keys"], ["direct", "experts.0.weight", "ue8m0_scale"])
        self.assertEqual(rank1["keys"], ["chunked", "experts.1.weight", "ue8m0_scale"])
        self.assertEqual(rank0["values"]["direct"], [90.0, 91.0])
        self.assertEqual(
            rank0["values"]["experts.0.weight"],
            [[1.0, 2.0], [3.0, 4.0]],
        )
        self.assertEqual(
            rank1["values"]["experts.1.weight"],
            [[5.0, 6.0], [7.0, 8.0]],
        )
        self.assertIsNone(rank0["chunked"])
        self.assertEqual(
            rank1["chunked"],
            {"matches": True, "numel": _CHUNKED_TENSOR_NUMEL},
        )
        for result in (rank0, rank1):
            self.assertEqual(result["ue8m0"]["dtype"], "torch.float8_e8m0fnu")
            self.assertEqual(result["ue8m0"]["raw_bytes"], _UE8M0_RAW_BYTES)


if __name__ == "__main__":
    unittest.main()
