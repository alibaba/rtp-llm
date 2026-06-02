"""UT: ``CompressorFP8.forward`` flat 2D input parity with legacy 3D.

Phase-3a part 2 lets the compressor accept either ``[T_total, dim]``
(varlen-native) or ``[B=1, S, dim]`` (legacy). The kernel still runs
on a flat ``[T_total, ...]`` view inside ``_launch`` either way, so the
two input shapes MUST produce byte-equal state-pool and KV-pool writes.

This UT pins that invariant down. If a future patch breaks 2D / 3D
equivalence (e.g. by routing one of them through a different reshape
path), the assertion fires before any downstream attention math
silently shifts.

Pattern follows ``test_compressor_fp8_per_token.py`` — same
``_build_compressor`` / ``_bind_pools`` pool-bind harness.

Run:
  bazelisk test //rtp_llm/models_py/modules/dsv4/test:test_compressor_forward_flat_input \\
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
from rtp_llm.models_py.modules.dsv4.fp8.compressor import CompressorFP8

DEVICE = "cuda"
TOKENS_PER_STATE_BLOCK = 256


def _build_prefill_metadata(
    compressor: CompressorFP8, sp: int, seqlen: int, device: torch.device
):
    positions = torch.arange(sp, sp + seqlen, device=device, dtype=torch.long)
    b_idx = torch.zeros(seqlen, device=device, dtype=torch.long)
    return compressor.prepare_metadata(
        positions,
        b_idx,
        seq_start_per_req=torch.tensor([sp], dtype=torch.int32, device=device),
        cu_seq_per_req=torch.tensor([0, seqlen], dtype=torch.int32, device=device),
    )


def _build_compressor(
    *,
    dim: int,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
) -> CompressorFP8:
    coff = 1 + (compress_ratio == 4)
    weights = {
        "ape": torch.randn(compress_ratio, coff * head_dim, dtype=torch.float32) * 0.1,
        "wkv": (torch.randn(coff * head_dim, dim, dtype=torch.float32) * 0.05).to(
            torch.bfloat16
        ),
        "wgate": (torch.randn(coff * head_dim, dim, dtype=torch.float32) * 0.05).to(
            torch.bfloat16
        ),
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
    # ``_wkv_wgate_fused`` is a plain attribute (not a registered buffer),
    # so ``.to(DEVICE)`` doesn't move it. The production code path comes
    # in already on-device via ``LinearFactory`` so this gap doesn't bite
    # there — only the from-scratch test fixture needs to fix it up.
    if cmp._wkv_wgate_fused is not None and cmp._wkv_wgate_fused.device.type != "cuda":
        cmp._wkv_wgate_fused = cmp._wkv_wgate_fused.to(DEVICE)
    if cmp.ape.device.type != "cuda":
        cmp.ape = torch.nn.Parameter(cmp.ape.data.to(DEVICE), requires_grad=False)
    max_pos = 4096
    freqs_cis = torch.ones(
        max_pos, rope_head_dim // 2, dtype=torch.complex64, device=DEVICE
    )
    cmp.freqs_cis = freqs_cis
    return cmp


def _bind_pools(
    cmp: CompressorFP8,
    *,
    seqlen: int,
    head_dim: int,
    coff: int,
    compress_ratio: int,
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
        state_tokens_per_block=state_eb,
        kv_tokens_per_block=kv_eb * compress_ratio,
    )
    return state_view_2d, kv_pool_3d


class CompressorFP8FlatInputParityTest(unittest.TestCase):

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        torch.manual_seed(0)

    def _run_one_shape(
        self,
        x: torch.Tensor,
        *,
        head_dim: int,
        rope_head_dim: int,
        compress_ratio: int,
        seqlen: int,
        sp: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coff = 1 + (compress_ratio == 4)
        # Re-seed so the random-init compressor weights match across runs;
        # both calls must see the SAME wkv / wgate / ape to be comparable.
        torch.manual_seed(42)
        cmp = _build_compressor(
            dim=int(x.shape[-1]),
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
        meta = _build_prefill_metadata(cmp, sp=sp, seqlen=seqlen, device=x.device)
        cmp.forward(x, sp, meta=meta)
        torch.cuda.synchronize()
        return state_view.clone(), kv_pool.clone()

    def _check_2d_eq_3d(
        self,
        *,
        head_dim: int,
        rope_head_dim: int,
        compress_ratio: int,
        seqlen: int,
        sp: int,
    ) -> None:
        dim = 64
        x_flat = torch.randn(seqlen, dim, dtype=torch.bfloat16, device=DEVICE)
        x_3d = x_flat.unsqueeze(0)  # [1, seqlen, dim]

        state_2d, kv_2d = self._run_one_shape(
            x_flat,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            seqlen=seqlen,
            sp=sp,
        )
        state_3d, kv_3d = self._run_one_shape(
            x_3d,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            seqlen=seqlen,
            sp=sp,
        )

        # State pool: fp32 — bit-equal expected (no FP8 quant in the path).
        self.assertTrue(
            torch.equal(state_2d, state_3d),
            "state pool diverges between 2D and 3D input forms",
        )
        # KV pool: uint8 packed FP8 + UE8M0 scales — also bit-equal because
        # the same kv_flat / score_flat tensors feed the same kernel.
        self.assertTrue(
            torch.equal(kv_2d, kv_3d),
            "KV pool diverges between 2D and 3D input forms",
        )

    # ------------------------------------------------------------------
    # CSA: head_dim=512, ratio=4 (overlap=True)
    # ------------------------------------------------------------------
    def test_csa_cold_prefill(self) -> None:
        self._check_2d_eq_3d(
            head_dim=KV_HEAD_DIM,
            rope_head_dim=64,
            compress_ratio=4,
            seqlen=64,
            sp=0,
        )

    def test_csa_continuation_prefill(self) -> None:
        self._check_2d_eq_3d(
            head_dim=KV_HEAD_DIM,
            rope_head_dim=64,
            compress_ratio=4,
            seqlen=32,
            sp=128,
        )

    # ------------------------------------------------------------------
    # Indexer's nested compressor: head_dim=128, ratio=4
    # ------------------------------------------------------------------
    def test_indexer_nested_cold_prefill(self) -> None:
        self._check_2d_eq_3d(
            head_dim=INDEXER_HEAD_DIM,
            rope_head_dim=128,
            compress_ratio=4,
            seqlen=64,
            sp=0,
        )


if __name__ == "__main__":
    unittest.main()
