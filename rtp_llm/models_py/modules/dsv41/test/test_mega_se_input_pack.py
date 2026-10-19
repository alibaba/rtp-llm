# SPDX-License-Identifier: Apache-2.0
"""Device tests for the fused MegaMoE-SE shared-L1 scale staging kernel.

The staging kernel must produce byte-identical destination contents to the
historical two-launch path (``destination[:active_rows].zero_()`` followed by
the forward-transform scatter): real scales at the UTCCP-transformed rows and
zeros everywhere else inside the active region, with the inactive tail and the
source buffer untouched. The production decode shape is tokens=192,
block_m=32, packed K=40 (dim 5120, packed UE8M0) inside CUDA-graph replay.
"""

import importlib.util
import unittest
from pathlib import Path

import torch

_PATH = Path(__file__).resolve().parent.parent / "moe" / "_mega_se_input_pack_triton.py"
_SPEC = importlib.util.spec_from_file_location("_mega_se_input_pack_under_test", _PATH)
_PACK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PACK)

stage_mega_moe_se_shared_l1_scales = _PACK.stage_mega_moe_se_shared_l1_scales


def reference_stage(source, destination, tokens, block_m):
    """Historical semantics: zero-fill the active region, then scatter."""
    if tokens == 0:
        return
    aligned_block_m = ((block_m + 127) // 128) * 128
    active_rows = ((tokens + block_m - 1) // block_m) * aligned_block_m
    destination[:active_rows].zero_()
    token_idx = torch.arange(tokens, dtype=torch.long, device=source.device)
    within = token_idx % block_m
    transformed = (
        token_idx // block_m * aligned_block_m
        + within // 128 * 128
        + (within % 32) * 4
        + (within % 128) // 32
    )
    destination[transformed] = source


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class MegaSEScaleStageFusedTest(unittest.TestCase):
    def _check(
        self,
        tokens,
        block_m,
        packed_k=40,
        capacity_rows=8192,
        dest_col_major=False,
        src_from_slice=False,
    ):
        device = "cuda"
        if dest_col_major:
            destination = torch.empty_strided(
                (capacity_rows, packed_k),
                (1, capacity_rows),
                dtype=torch.int32,
                device=device,
            )
        else:
            destination = torch.full(
                (capacity_rows, packed_k), 1234567, dtype=torch.int32, device=device
            )
        if src_from_slice:
            # Production source: buf.x_sf[:tokens], a row slice of a wider buffer.
            full = torch.randint(
                -(2**31),
                2**31 - 1,
                (capacity_rows, packed_k),
                dtype=torch.int32,
                device=device,
            )
            source = full[: max(tokens, 1)] if tokens else full[:1]
            if tokens:
                source = source.clone().reshape(tokens, packed_k)
                full[:tokens] = source
                source = full[:tokens]
        else:
            source = torch.randint(
                -(2**31),
                2**31 - 1,
                (max(tokens, 1), packed_k),
                dtype=torch.int32,
                device=device,
            )[:tokens]

        fused = destination.clone()
        stage_mega_moe_se_shared_l1_scales(source, fused, tokens, block_m)
        reference_stage(source, destination, tokens, block_m)
        self.assertTrue(
            torch.equal(fused, destination),
            f"mismatch tokens={tokens} block_m={block_m} k={packed_k} "
            f"col_major={dest_col_major} src_slice={src_from_slice}",
        )

    def test_production_decode_shape(self):
        # 40 layers/step at tokens=192 and 3 at the smaller draft bucket.
        for tokens in (192, 160, 144, 129, 256):
            with self.subTest(tokens=tokens):
                self._check(tokens, 32)

    def test_reference_mapping_all_kernel_block_sizes(self):
        for block_m in (16, 32, 64, 96, 128, 192, 256, 384):
            for tokens in (1, 31, 32, 33, 127, 128, 257, 300):
                with self.subTest(block_m=block_m, tokens=tokens):
                    self._check(tokens, block_m)

    def test_non_default_strides(self):
        for tokens, block_m in ((192, 32), (257, 192), (33, 16)):
            with self.subTest(tokens=tokens, block_m=block_m):
                self._check(tokens, block_m, dest_col_major=True)
                self._check(tokens, block_m, src_from_slice=True)

    def test_inactive_tail_untouched(self):
        tokens, block_m, packed_k = 192, 32, 40
        capacity_rows = 4096
        destination = torch.full(
            (capacity_rows, packed_k), 987654, dtype=torch.int32, device="cuda"
        )
        source = torch.randint(
            -(2**31), 2**31 - 1, (tokens, packed_k), dtype=torch.int32, device="cuda"
        )
        stage_mega_moe_se_shared_l1_scales(source, destination, tokens, block_m)
        active_rows = ((tokens + block_m - 1) // block_m) * 128
        self.assertTrue(torch.all(destination[active_rows:] == 987654).item())

    def test_zero_tokens_is_noop(self):
        destination = torch.full((512, 8), 55, dtype=torch.int32, device="cuda")
        source = torch.zeros((1, 8), dtype=torch.int32, device="cuda")
        stage_mega_moe_se_shared_l1_scales(source, destination, 0, 32)
        self.assertTrue(torch.all(destination == 55).item())

    def test_reuse_clears_active_holes(self):
        # A smaller/block_m-changed round must clear every padding row the
        # previous round wrote INSIDE the new active region (the TMA
        # stale-bytes contract); residual bytes beyond the new active_rows
        # match the historical path byte-for-byte (never read by TMA).
        packed_k, capacity_rows = 3, 1024
        destination = torch.empty(
            (capacity_rows, packed_k), dtype=torch.int32, device="cuda"
        )
        destination.fill_(99)
        first = torch.full((257, packed_k), 7, dtype=torch.int32, device="cuda")
        second = torch.full((33, packed_k), 11, dtype=torch.int32, device="cuda")
        stage_mega_moe_se_shared_l1_scales(first, destination, 257, 192)
        stage_mega_moe_se_shared_l1_scales(second, destination, 33, 16)
        expected = torch.full(
            (capacity_rows, packed_k), 99, dtype=torch.int32, device="cuda"
        )
        reference_stage(first, expected, 257, 192)
        reference_stage(second, expected, 33, 16)
        self.assertTrue(torch.equal(destination, expected))
        # Explicit hole check: no first-round byte survives inside the new
        # active region (384 rows): only zeros and second-round data there.
        self.assertFalse(torch.any(destination[:384] == 7).item())
        self.assertFalse(torch.any(destination[:384] == 99).item())

    def test_cuda_graph_replay_rewrites_padding_and_data(self):
        tokens, block_m, packed_k = 192, 32, 40
        source = torch.randint(
            -(2**31), 2**31 - 1, (tokens, packed_k), dtype=torch.int32, device="cuda"
        )
        destination = torch.empty((1024, packed_k), dtype=torch.int32, device="cuda")
        # Warm the kernel before capture (mirrors the per-bucket eager warmup).
        stage_mega_moe_se_shared_l1_scales(source, destination, tokens, block_m)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            stage_mega_moe_se_shared_l1_scales(source, destination, tokens, block_m)
        for trial in range(3):
            # Poison the whole buffer, refill the source, replay: every active
            # row (data AND padding) must be rewritten by the single kernel.
            destination.fill_(424242 + trial)
            source.copy_(
                torch.randint(
                    -(2**31),
                    2**31 - 1,
                    (tokens, packed_k),
                    dtype=torch.int32,
                    device="cuda",
                )
            )
            expected = destination.clone()
            reference_stage(source, expected, tokens, block_m)
            graph.replay()
            torch.cuda.synchronize()
            active_rows = ((tokens + block_m - 1) // block_m) * 128
            self.assertTrue(
                torch.equal(destination[:active_rows], expected[:active_rows])
            )
            self.assertTrue(
                torch.all(destination[active_rows:] == 424242 + trial).item()
            )


if __name__ == "__main__":
    unittest.main()
