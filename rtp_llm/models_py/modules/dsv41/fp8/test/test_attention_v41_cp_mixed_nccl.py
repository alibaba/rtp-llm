"""CP4 fixed-tile source transport through real NCCL and native FP32 GEMM."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_SPEC = importlib.util.spec_from_file_location(
    "cp_stream_contracts",
    Path(__file__).with_name("test_attention_v41_cp_raw_groups.py"),
)
CONTRACTS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(CONTRACTS)

MIXED = (503, 159, 3263, 583, 8127, 1940, 5177, 8370, 2913, 159357, 14606, 9695)
LONG = (1048576,)
WIDE = (1, 9, 127, 129) * 32 + (2049,)
COLD_B1024 = (2,) * 1024
MIXED_B1024 = (1, 2, 3, 4, 5, 6, 7, 8) * 128
B129_FRESH257 = (257,) * 129
EXPLICIT_LIMIT = 512 * 1024**2


def _worker(rank, rendezvous, output_dir, cases):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method="file://" + rendezvous, rank=rank, world_size=4
    )
    try:
        from rtp_llm.models_py.distributed import collective_torch
        from rtp_llm.models_py.modules.dsv41.fp8.attention import _get_cp_comm_stream

        CONTRACTS.STREAM._get_cp_comm_stream = _get_cp_comm_stream
        summaries = []
        with patch.object(
            collective_torch, "_get_group", return_value=dist.group.WORLD
        ):
            for lengths in cases:
                cp, local, raw = CONTRACTS._case(lengths)
                cp.cp_rank = rank
                cp.unpad_restore = cp.unpad_restore.to("cuda")
                x = torch.zeros(
                    (len(local[rank]), 5120), device="cuda", dtype=torch.bfloat16
                )
                x[:, :4] = local[rank].to("cuda")
                del local
                for ratio in (1, 2):
                    weights = []
                    for column in range(ratio):
                        weight = torch.zeros(
                            (512, 5120), device="cuda", dtype=torch.bfloat16
                        )
                        weight[:4, :4] = torch.eye(
                            4, device="cuda", dtype=torch.bfloat16
                        ) * (1 if column == 0 else 0.5)
                        weights.append(weight)
                    expected = torch.zeros(
                        (len(raw), 512 * ratio), device="cuda", dtype=torch.float32
                    )
                    for column in range(ratio):
                        expected[:, column * 512 : column * 512 + 4] = raw.to(
                            "cuda"
                        ).float() * (1 if column == 0 else 0.5)
                    output = torch.empty_like(expected)

                    def execute(destination):
                        seen = []
                        groups = CONTRACTS.STREAM._prefill_x_group_plan(
                            x, cp, output_width=512 * ratio
                        )
                        assert groups is not None
                        tiles = CONTRACTS.STREAM._prefill_projected_x_groups(
                            x, cp, groups, tuple(weights)
                        )
                        try:
                            for start, tile in tiles:
                                seen.append((start, len(tile)))
                                destination[start : start + len(tile)].copy_(tile)
                                del tile
                        finally:
                            tiles.close()
                        assert seen == [
                            (start, min(32768, len(raw) - start))
                            for start in range(0, len(raw), 32768)
                        ]
                        return destination

                    with torch.inference_mode():
                        torch.cuda.synchronize()
                        before = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        execute(output)
                        torch.cuda.synchronize()
                        peak = torch.cuda.max_memory_allocated() - before
                        assert peak <= EXPLICIT_LIMIT, (rank, lengths, ratio, peak)
                        torch.testing.assert_close(output, expected, rtol=0, atol=0)
                        # Different data and compute stream expose stale send,
                        # receive, or producer-slab reuse across tile boundaries.
                        compute = torch.cuda.Stream()
                        compute.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(compute):
                            x[:, :4].add_(1)
                            execute(output)
                        torch.cuda.current_stream().wait_stream(compute)
                        torch.cuda.synchronize()
                        changed = expected.clone()
                        changed[:, :4].add_(1)
                        if ratio == 2:
                            changed[:, 512:516].add_(0.5)
                        torch.testing.assert_close(output, changed, rtol=0, atol=0)
                        x[:, :4].sub_(1)
                        probe = torch.tensor([rank + 1], device="cuda")
                        dist.all_reduce(probe)
                        assert probe.item() == 10
                    summaries.append(
                        {
                            "rank": rank,
                            "shape": list(lengths),
                            "ratio": ratio,
                            "rows": len(raw),
                            "peak_temporary_bytes": peak,
                            "exact_projection": True,
                        }
                    )
                    del expected, output, weights
                del x, raw
        Path(output_dir, f"stream_rank{rank}.json").write_text(
            json.dumps(summaries, indent=2)
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


class CPStreamNCCLTest(unittest.TestCase):
    def _run_cases(self, cases):
        self.assertTrue(torch.cuda.is_available())
        self.assertGreaterEqual(torch.cuda.device_count(), 4)
        output_dir = Path(
            os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", tempfile.gettempdir())
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=os.environ.get("TEST_TMPDIR")) as tmp:
            mp.spawn(
                _worker,
                args=(str(Path(tmp) / "rendezvous"), str(output_dir), cases),
                nprocs=4,
            )

    def test_mixed_and_long_cp4(self):
        self._run_cases((MIXED, LONG))

    def test_wide_and_b1024_cp4(self):
        self._run_cases((WIDE, COLD_B1024))

    def test_b129_fresh257_cp4(self):
        self._run_cases((B129_FRESH257,))

    def test_mixed_b1024_cp4(self):
        self._run_cases((MIXED_B1024,))


if __name__ == "__main__":
    unittest.main()
