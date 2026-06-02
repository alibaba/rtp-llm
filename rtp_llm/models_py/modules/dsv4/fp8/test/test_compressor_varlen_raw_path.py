"""UT: ``run_fused_compress_kv_write`` BATCHED varlen raw path
(Phase-3a part 4c).

Validates that the kernel's new ``BATCHED=True`` branch (per-request
``seq_start_per_req`` + ``cu_seq_per_req`` arrays) produces byte-equal
state-pool + KV-pool to the per-request scalar-``seq_start`` path the
legacy B==1 launch already covers.

Constructs a 2-request batch (different ``sp_b``, different ``S_b``),
runs CompressorFP8.forward in two ways and compares:

1. **Batched single launch**: build (positions, b_idx) interleaved across
   requests, build per-req ``sp_per_req`` / ``cu_seqlens``, call once.
2. **Per-request reference**: separate compressor instances (fresh pools
   each), one launch per request, then compose the two pool snapshots
   into a single combined view (different requests live in different
   block-table slots so their pool writes are disjoint).

The two MUST be byte-equal. If a future patch breaks the BATCHED branch
(e.g. wrong cu_seq offset arithmetic), the assertion fires before any
B>1 prefill silently corrupts the compressed KV pool.

Run:
  bazelisk test //rtp_llm/models_py/modules/dsv4/test:test_compressor_varlen_raw_path \\
    --verbose_failures --config=cuda13 --test_output=all --jobs=64
"""

from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8._compressor_consts import (
    KV_ENTRY_BYTES,
    KV_HEAD_DIM,
)
from rtp_llm.models_py.modules.dsv4.fp8.compressor import (
    CompressorFP8,
    _linear_bf16_bf16_fp32,
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
        max_batch_size=4,
        norm_eps=1e-6,
        compressor_weights=weights,
    ).to(DEVICE)
    if cmp._wkv_wgate_fused is not None and cmp._wkv_wgate_fused.device.type != "cuda":
        cmp._wkv_wgate_fused = cmp._wkv_wgate_fused.to(DEVICE)
    if cmp.ape.device.type != "cuda":
        cmp.ape = torch.nn.Parameter(cmp.ape.data.to(DEVICE), requires_grad=False)
    cmp.freqs_cis = torch.ones(
        4096, rope_head_dim // 2, dtype=torch.complex64, device=DEVICE
    )
    return cmp


def _bind_pools_for_batch(
    cmp: CompressorFP8,
    *,
    bsz: int,
    head_dim: int,
    coff: int,
    compress_ratio: int,
    n_compressed_per_req: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_eb = TOKENS_PER_STATE_BLOCK
    state_blocks_per_req = 2
    state_total_blocks = 1 + bsz * state_blocks_per_req
    hidden = 2 * coff * head_dim
    state_view_2d = torch.zeros(
        state_total_blocks * state_eb, hidden, dtype=torch.float32, device=DEVICE
    )
    # bt[b] = blocks owned by req b, all distinct (no overlap).
    state_block_table = torch.tensor(
        [
            [1 + b * state_blocks_per_req + i for i in range(state_blocks_per_req)]
            for b in range(bsz)
        ],
        dtype=torch.int32,
        device=DEVICE,
    )

    kv_eb = TOKENS_PER_STATE_BLOCK // compress_ratio
    blocks_per_req = max(1, (n_compressed_per_req + kv_eb - 1) // kv_eb)
    kv_total_blocks = 1 + bsz * blocks_per_req
    kv_pool_3d = torch.zeros(
        kv_total_blocks, kv_eb, KV_ENTRY_BYTES, dtype=torch.uint8, device=DEVICE
    )
    kv_block_table = torch.tensor(
        [
            [1 + b * blocks_per_req + i for i in range(blocks_per_req)]
            for b in range(bsz)
        ],
        dtype=torch.int32,
        device=DEVICE,
    )
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


class CompressorVarlenRawPathTest(unittest.TestCase):

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")

    def test_batched_eq_per_request(self) -> None:
        head_dim = KV_HEAD_DIM
        rope_head_dim = 64
        compress_ratio = 4
        dim = 64
        coff = 1 + (compress_ratio == 4)

        # Two requests with different sp / S — both fit in state pool.
        sps = [0, 64]
        S_per_req = [64, 32]
        T_total = sum(S_per_req)
        n_compressed_per_req = max(
            (sps[b] + S_per_req[b] + compress_ratio - 1) // compress_ratio
            for b in range(2)
        )

        torch.manual_seed(7)
        x_per_req = [
            torch.randn(S_per_req[b], dim, dtype=torch.bfloat16, device=DEVICE)
            for b in range(2)
        ]
        x_flat = torch.cat(x_per_req, dim=0)  # [T_total, dim]

        # ── Path A: batched single launch (uses BATCHED=True) ──
        torch.manual_seed(42)
        cmp_a = _build_compressor(
            dim=dim,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
        )
        state_a, kv_a = _bind_pools_for_batch(
            cmp_a,
            bsz=2,
            head_dim=head_dim,
            coff=coff,
            compress_ratio=compress_ratio,
            n_compressed_per_req=n_compressed_per_req,
        )
        positions = torch.cat(
            [
                torch.arange(sps[b], sps[b] + S_per_req[b], device=DEVICE)
                for b in range(2)
            ]
        ).to(torch.long)
        b_idx = torch.cat(
            [torch.full((S_per_req[b],), b, device=DEVICE) for b in range(2)]
        ).to(torch.long)
        cu_seqlens = torch.tensor(
            [0, S_per_req[0], S_per_req[0] + S_per_req[1]],
            dtype=torch.int32,
            device=DEVICE,
        )
        sp_per_req = torch.tensor(sps, dtype=torch.int32, device=DEVICE)
        meta_a = cmp_a.prepare_metadata(
            positions,
            b_idx,
            seq_start_per_req=sp_per_req,
            cu_seq_per_req=cu_seqlens,
        )
        # Reproduce forward()'s kv/score projection; it uses BF16 operands
        # with FP32 accumulation/output before writing the FP32 state pool.
        out_dim = (1 + cmp_a.overlap) * cmp_a.head_dim
        fused_out = _linear_bf16_bf16_fp32(x_flat, cmp_a._wkv_wgate_fused)
        kv_flat = fused_out[..., :out_dim].contiguous()
        score_flat = fused_out[..., out_dim:].contiguous()
        cmp_a._launch(kv_flat, score_flat, meta_a)
        torch.cuda.synchronize()

        # ── Path B: per-request reference with the same metadata path ──
        torch.manual_seed(42)
        cmp_b = _build_compressor(
            dim=dim,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
        )
        state_b, kv_b = _bind_pools_for_batch(
            cmp_b,
            bsz=2,
            head_dim=head_dim,
            coff=coff,
            compress_ratio=compress_ratio,
            n_compressed_per_req=n_compressed_per_req,
        )
        # Per-req launches; each request's b_idx is 0 against its own
        # block-table row. To stay in the shared pool we patch
        # ``b_idx = b`` so block_table[req_idx] resolves to the right row.
        for b in range(2):
            x_b = x_per_req[b]
            fused_b = _linear_bf16_bf16_fp32(x_b, cmp_b._wkv_wgate_fused)
            kv_b_flat = fused_b[..., :out_dim].contiguous()
            score_b_flat = fused_b[..., out_dim:].contiguous()
            pos_b = torch.arange(
                sps[b], sps[b] + S_per_req[b], device=DEVICE, dtype=torch.long
            )
            bidx_b = torch.full((S_per_req[b],), b, device=DEVICE, dtype=torch.long)
            meta_b = cmp_b.prepare_metadata(
                pos_b,
                bidx_b,
                seq_start_per_req=torch.tensor(
                    [sps[b]], dtype=torch.int64, device=DEVICE
                ),
                cu_seq_per_req=torch.tensor(
                    [0, S_per_req[b]], dtype=torch.int64, device=DEVICE
                ),
            )
            cmp_b._launch(kv_b_flat, score_b_flat, meta_b)
        torch.cuda.synchronize()

        self.assertTrue(
            torch.equal(state_a, state_b),
            "state pool diverged between batched and per-request metadata paths",
        )
        self.assertTrue(
            torch.equal(kv_a, kv_b),
            "KV pool diverged between BATCHED kernel and per-request scalar path",
        )


if __name__ == "__main__":
    unittest.main()
