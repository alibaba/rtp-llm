"""UT: ``CompressorFP8._launch`` raw_path on/off equivalence (Phase-3a part 4a).

Validates option (b) of the batched prefill plan: when the state pool has
not been overwritten within a launch (i.e. seqlen per request fits in the
state pool capacity), calling ``_launch(seq_start=sp)`` (raw_path enabled)
must produce a byte-equal KV pool to ``_launch(seq_start=None)`` (raw_path
disabled, kernel reads only from state_cache).

If this invariant holds, B>1 batched prefill can safely fall back to
``seq_start=None`` while we ship the kernel-level varlen support
(option c). If it doesn't, option (b) is unsafe and we must skip it.

Constraints:
  * B == 1 only (raw_path's scalar ``seq_start`` is well-defined here)
  * seqlen <= state pool capacity per request → no cyclic overwrite

Run:
  bazelisk test //rtp_llm/models_py/modules/dsv4/test:test_compressor_disable_raw_path \\
    --verbose_failures --config=cuda13 --test_output=all --jobs=64
"""

from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8._compressor_consts import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
    KV_ENTRY_BYTES,
    KV_HEAD_DIM,
)
from rtp_llm.models_py.modules.dsv4.fp8.compressor import (
    CompressorFP8,
    _build_prefill_positions,
)

DEVICE = "cuda"
TOKENS_PER_STATE_BLOCK = 256


def _build_compressor(
    *, dim: int, head_dim: int, rope_head_dim: int, compress_ratio: int
) -> CompressorFP8:
    coff = 1 + (compress_ratio == 4)
    weights = {
        "ape": torch.randn(compress_ratio, coff * head_dim, dtype=torch.float32) * 0.1,
        "wkv": torch.randn(coff * head_dim, dim, dtype=torch.float32) * 0.05,
        "wgate": torch.randn(coff * head_dim, dim, dtype=torch.float32) * 0.05,
        "norm": torch.ones(head_dim, dtype=torch.bfloat16),
    }
    cmp = CompressorFP8(
        dim=dim,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        max_batch_size=1,
        norm_eps=1e-6,
        compressor_weights=weights,
    ).to(DEVICE)
    if cmp._wkv_wgate_fused is not None and cmp._wkv_wgate_fused.device.type != "cuda":
        cmp._wkv_wgate_fused = cmp._wkv_wgate_fused.to(DEVICE)
    if cmp.ape.device.type != "cuda":
        cmp.ape = torch.nn.Parameter(cmp.ape.data.to(DEVICE), requires_grad=False)
    freqs_cis = torch.ones(
        4096, rope_head_dim // 2, dtype=torch.complex64, device=DEVICE
    )
    cmp.freqs_cis = freqs_cis
    return cmp


def _bind_pools(
    cmp: CompressorFP8, *, seqlen: int, head_dim: int, coff: int, compress_ratio: int
) -> tuple[torch.Tensor, torch.Tensor]:
    state_eb = TOKENS_PER_STATE_BLOCK
    state_blocks_per_req = 2
    state_total_blocks = 1 + state_blocks_per_req
    hidden = 2 * coff * head_dim
    state_view_2d = torch.zeros(
        state_total_blocks * state_eb, hidden, dtype=torch.float32, device=DEVICE
    )
    state_block_table = torch.tensor([[1, 2]], dtype=torch.int32, device=DEVICE)

    entry_bytes = KV_ENTRY_BYTES if head_dim == KV_HEAD_DIM else INDEXER_ENTRY_BYTES
    kv_eb = TOKENS_PER_STATE_BLOCK // compress_ratio
    n_compressed = (seqlen + compress_ratio - 1) // compress_ratio
    kv_blocks_needed = max(1, (n_compressed + kv_eb - 1) // kv_eb)
    kv_total_blocks = 1 + kv_blocks_needed
    kv_pool_3d = torch.zeros(
        kv_total_blocks, kv_eb, entry_bytes, dtype=torch.uint8, device=DEVICE
    )
    kv_block_table = torch.arange(
        1, 1 + kv_blocks_needed, dtype=torch.int32, device=DEVICE
    ).reshape(1, kv_blocks_needed)

    cmp.set_pool_context(
        kv_pool_view=kv_pool_3d,
        kv_block_table=kv_block_table,
        kv_eb=kv_eb,
        state_pool_view=state_view_2d,
        state_block_table=state_block_table,
        state_eb=state_eb,
    )
    return state_view_2d, kv_pool_3d


class CompressorDisableRawPathTest(unittest.TestCase):

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

    def _run_one(
        self,
        *,
        head_dim: int,
        rope_head_dim: int,
        compress_ratio: int,
        seqlen: int,
        sp: int,
        raw_path: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(42)
        dim = 64
        coff = 1 + (compress_ratio == 4)
        cmp = _build_compressor(
            dim=dim,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
        )
        state_view, kv_pool = _bind_pools(
            cmp,
            seqlen=seqlen,
            head_dim=head_dim,
            coff=coff,
            compress_ratio=compress_ratio,
        )

        torch.manual_seed(123)
        x = torch.randn(seqlen, dim, dtype=torch.bfloat16, device=DEVICE)
        out_dim = (1 + cmp.overlap) * cmp.head_dim
        # bf16 × bf16 — matches ``_wkv_wgate_fused`` storage dtype and
        # what the production forward path executes.
        fused_out = torch.nn.functional.linear(x, cmp._wkv_wgate_fused)
        kv, score = fused_out[..., :out_dim], fused_out[..., out_dim:]
        kv_flat = kv.reshape(seqlen, -1).contiguous()
        score_flat = score.reshape(seqlen, -1).contiguous()

        positions, b_idx = _build_prefill_positions(sp, 1, seqlen, DEVICE)
        meta = cmp.prepare_metadata(positions, b_idx)
        cmp._launch(kv_flat, score_flat, meta, seq_start=(sp if raw_path else None))
        torch.cuda.synchronize()
        return state_view.clone(), kv_pool.clone()

    def _check_eq(
        self,
        *,
        head_dim: int,
        rope_head_dim: int,
        compress_ratio: int,
        seqlen: int,
        sp: int,
    ) -> None:
        # seqlen must fit in state pool to avoid cyclic overwrite, otherwise
        # raw_path is the ONLY correct path and disabling it is expected to
        # diverge — the whole point of the optimization.
        assert seqlen <= TOKENS_PER_STATE_BLOCK, (
            f"seqlen={seqlen} exceeds state pool capacity "
            f"({TOKENS_PER_STATE_BLOCK}); option (b) cannot be safe here"
        )
        state_raw, kv_raw = self._run_one(
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            seqlen=seqlen,
            sp=sp,
            raw_path=True,
        )
        state_off, kv_off = self._run_one(
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            seqlen=seqlen,
            sp=sp,
            raw_path=False,
        )
        # State pool is written by run_save_partial_states unconditionally —
        # must match in both runs.
        self.assertTrue(
            torch.equal(state_raw, state_off),
            "state pool diverged between raw_path=True and raw_path=False",
        )
        # KV pool is the kernel output we care about: must be byte-equal so
        # batched prefill can fall back to raw_path=False without changing
        # downstream attention math.
        self.assertTrue(
            torch.equal(kv_raw, kv_off),
            "KV pool diverged between raw_path=True and raw_path=False — "
            "option (b) (disable_raw_path for batched) is UNSAFE",
        )

    # ---------- CSA: head_dim=512, ratio=4 (overlap=True) ----------
    def test_csa_cold_short(self) -> None:
        self._check_eq(
            head_dim=KV_HEAD_DIM,
            rope_head_dim=64,
            compress_ratio=4,
            seqlen=64,
            sp=0,
        )

    def test_csa_continuation_short(self) -> None:
        self._check_eq(
            head_dim=KV_HEAD_DIM,
            rope_head_dim=64,
            compress_ratio=4,
            seqlen=32,
            sp=128,
        )

    # ---------- Indexer's nested compressor: head_dim=128 ----------
    def test_indexer_nested_short(self) -> None:
        self._check_eq(
            head_dim=INDEXER_HEAD_DIM,
            rope_head_dim=128,
            compress_ratio=4,
            seqlen=64,
            sp=0,
        )


if __name__ == "__main__":
    unittest.main()
