"""CP4 packed SWA prefix/fresh parity with real NCCL and byte-sliced pools."""

import importlib.util
import itertools
import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_SPEC = importlib.util.spec_from_file_location(
    "packed_swa_contracts", Path(__file__).with_name("test_v41_swa_packed.py")
)
CONTRACTS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(CONTRACTS)


def gather_fresh(case, rank):
    """Independent zigzag labels exercise non-8-aligned rows through NCCL."""
    labels = [[] for _ in range(4)]
    base = 0
    for length in case.lengths:
        half = (length + 7) // 8
        for owner in range(4):
            for start in (owner * half, (7 - owner) * half):
                labels[owner].extend(
                    base + i if i < length else -1 for i in range(start, start + half)
                )
        base += length
    selected = torch.tensor(labels[rank], device="cuda")
    local = case.qkv.kv_full[selected.clamp_min(0)]
    local[selected < 0] = 0
    gathered = torch.empty((local.shape[0] * 4, 512), dtype=local.dtype, device="cuda")
    dist.all_gather_into_tensor(gathered, local)
    restore = [0] * base
    for index, label in enumerate(itertools.chain.from_iterable(labels)):
        if label >= 0:
            restore[label] = index
    restored = gathered.index_select(0, torch.tensor(restore, device="cuda"))
    torch.testing.assert_close(restored, case.qkv.kv_full, rtol=0, atol=0)
    case.qkv.kv_full = restored


def _worker(rank, rendezvous):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=180),
    )
    try:
        from rtp_llm.models_py.distributed import collective_torch

        check = unittest.TestCase()
        # Only route TP to this test's real process group; no collective/kernel
        # result is replaced. Prefix reads exchange this rank's opaque bytes.
        with patch.object(
            collective_torch, "_get_group", return_value=dist.group.WORLD
        ):
            with torch.no_grad():
                for ratio in (1, 2):
                    case = CONTRACTS.make_fixture(cp_size=4, rank=rank)
                    gather_fresh(case, rank)
                    case.owner.compress_ratio = ratio
                    check.assertTrue(
                        case.owner._can_fuse_swa_fresh(case.qkv, case.common)
                    )
                    unfused, starts = case.owner._swa_prefill_workspace(
                        case.qkv, case.common
                    )
                    CONTRACTS.assert_packed(check, case, unfused.flat)
                    fused, fused_starts = case.owner._swa_prefill_workspace(
                        case.qkv, case.common, fuse_cache_write=True
                    )
                    check.assertEqual(starts, fused_starts)
                    CONTRACTS.assert_packed(check, case, fused.flat)
                    CONTRACTS.assert_attention(check, case, fused.flat)
                    torch.testing.assert_close(fused.flat, unfused.flat, rtol=0, atol=0)
                    full = torch.empty(
                        (case.raw.shape[0] * 4, case.raw.shape[1]),
                        dtype=torch.uint8,
                        device="cuda",
                    )
                    dist.all_gather_into_tensor(full, case.raw)
                    full = (
                        full.view(4, case.raw.shape[0], -1)
                        .permute(1, 0, 2)
                        .reshape(case.raw.shape[0], -1)
                    )
                    torch.testing.assert_close(
                        full.cpu(), CONTRACTS.expected_cache_after(case), rtol=0, atol=0
                    )
                    CONTRACTS.write_measurement(
                        f"swa_packed_cp4_rank{rank}_ratio{ratio}",
                        {
                            "rank": rank,
                            "world_size": 4,
                            "compression_ratio": ratio,
                            "storage_bytes": fused.flat.untyped_storage().nbytes(),
                            "packed_rows": fused.flat.shape[0],
                            "real_nccl": True,
                            "prefix_fresh_and_cache_exact": True,
                            "attention_checked": True,
                        },
                    )
        dist.barrier()
    finally:
        dist.destroy_process_group()


class PackedSwaCP4NCCLTest(unittest.TestCase):
    def test_mixed_prefix_ring_wrap_fused_and_unfused(self):
        self.assertTrue(torch.cuda.is_available(), "This target requires CUDA")
        self.assertGreaterEqual(
            torch.cuda.device_count(), 4, "This target requires four GPUs"
        )
        with tempfile.TemporaryDirectory(dir=os.environ.get("TEST_TMPDIR")) as tmp:
            mp.spawn(_worker, args=(str(Path(tmp) / "rendezvous"),), nprocs=4)


if __name__ == "__main__":
    unittest.main()
