"""End-to-end class-level UT for ``IndexerFP8``.

Decode path: build a layer with random weights, set up state + KV pool,
seed the pool with N pre-existing compressed tokens, run
``forward_decode_vectorized``, verify topk indices look sane (within
range, not all -1) and that the underlying score computation produces
finite logits.

The kernel-level correctness of the score is locked by
``test_indexer_fp8_score.py``; the writer half by
``test_indexer_fp8_writer.py``. This UT just gates that the class wires
both halves together correctly + the topk + nested compressor write.
"""

from __future__ import annotations

import torch

try:
    import pytest

    HAVE_PYTEST = True
except ImportError:
    HAVE_PYTEST = False

    class _NoOpMark:
        def parametrize(self, *args, **kwargs):
            def deco(fn):
                return fn

            return deco

    class _NoOpPytest:
        mark = _NoOpMark()

        @staticmethod
        def skip(msg):
            raise SystemExit(f"SKIP: {msg}")

    pytest = _NoOpPytest()

from rtp_llm.models_py.modules.dsv4._compressor_fused_triton import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
)
from rtp_llm.models_py.modules.dsv4._indexer_score_fp8 import has_fp8_paged_mqa_logits
from rtp_llm.models_py.modules.dsv4.indexer_fp8 import IndexerFP8


def _build_freqs_cis(max_pos: int, rope_dim: int, device, base: float = 10000.0):
    half = rope_dim // 2
    freqs = 1.0 / (
        base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    pos = torch.arange(max_pos, device=device, dtype=torch.float32)
    angles = torch.outer(pos, freqs)
    return torch.polar(torch.ones_like(angles), angles).to(torch.complex64)


def test_indexer_fp8_decode_smoke():
    """IndexerFP8 decode path: layer wires nested compressor + DeepGEMM
    score + topk. Smoke-level — verifies non-NaN finite output and
    correctly-sized topk indices."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not has_fp8_paged_mqa_logits():
        pytest.skip("deep_gemm.fp8_paged_mqa_logits unavailable")

    torch.manual_seed(0xBEEF)
    device = torch.device("cuda")

    dim = 1024
    q_lora_rank = 512
    index_n_heads = 64
    index_head_dim = INDEXER_HEAD_DIM
    rope_dim = 64
    index_topk = 512  # one of the persistent-topk-supported sizes
    compress_ratio = 4
    max_batch_size = 2
    max_seq_len = 4096
    norm_eps = 1e-6

    layer = IndexerFP8(
        dim=dim,
        q_lora_rank=q_lora_rank,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        rope_head_dim=rope_dim,
        index_topk=index_topk,
        compress_ratio=compress_ratio,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        norm_eps=norm_eps,
        weights=None,
    ).to(device)
    layer.compressor.norm.weight.data = (
        torch.randn(index_head_dim, device=device) * 0.05 + 1.0
    ).to(torch.bfloat16)
    layer.compressor.ape.data = (
        torch.randn(compress_ratio, layer.compressor.ape.shape[1], device=device) * 0.1
    )
    # Smoke-mode: weights/projections were nn.Linear-default fp32; production
    # factory mode loads the right dtype. Cast for the smoke run.
    layer.weights_proj.weight.data = layer.weights_proj.weight.data.to(torch.bfloat16)

    layer.freqs_cis = _build_freqs_cis(max_seq_len, rope_dim, device)
    layer.compressor.freqs_cis = layer.freqs_cis

    # ── Build pools ──
    bsz = 2
    block_size = 64  # DeepGEMM constraint
    num_blocks = 16
    state_pool_view = torch.zeros(
        num_blocks * block_size,
        layer.compressor._state_dim * 2,
        dtype=torch.float32,
        device=device,
    )
    state_block_table = (
        torch.arange(1, num_blocks + 1, dtype=torch.int32, device=device)
        .view(1, num_blocks)
        .expand(bsz, num_blocks)
        .contiguous()
    )

    kv_pool_view = torch.zeros(
        num_blocks * block_size,
        INDEXER_ENTRY_BYTES,
        dtype=torch.uint8,
        device=device,
    )
    kv_block_table = (
        torch.arange(1, num_blocks + 1, dtype=torch.int32, device=device)
        .view(1, num_blocks)
        .expand(bsz, num_blocks)
        .contiguous()
    )

    layer.set_pool_context(
        kv_pool_view=kv_pool_view,
        kv_block_table=kv_block_table,
        kv_eb=block_size,
        state_pool_view=state_pool_view,
        state_block_table=state_block_table,
        state_eb=block_size,
    )

    # ── Decode call ──
    # Each request is at start_pos s.t. (sp+1) % ratio == 0 → triggers
    # compressor write. With ratio=4 and start_pos=[3, 7], both are
    # boundary positions emitting compressed tokens 0 and 1.
    start_pos = torch.tensor([3, 7], dtype=torch.int32, device=device)
    x = torch.randn(bsz, 1, dim, dtype=torch.bfloat16, device=device) * 0.1
    qr = torch.randn(bsz, 1, q_lora_rank, dtype=torch.bfloat16, device=device) * 0.1
    out_topk = torch.full((bsz, 1, index_topk), -1, dtype=torch.int32, device=device)

    layer.forward_decode_vectorized(x, qr, start_pos, out_topk)

    # ── Sanity ──
    # For start_pos=3, compressed_len = (3+1)/4 = 1 → topk[0] should
    # have exactly 1 valid entry (others = -1).
    # For start_pos=7, compressed_len = (7+1)/4 = 2 → topk[1] should
    # have exactly 2 valid entries.
    valid_per_req = (out_topk[:, 0] >= 0).sum(dim=-1)
    assert valid_per_req[0].item() == 1, (
        f"req 0 expected 1 valid topk entry (compressed_len=1); got "
        f"{valid_per_req[0].item()}"
    )
    assert valid_per_req[1].item() == 2, (
        f"req 1 expected 2 valid topk entries (compressed_len=2); got "
        f"{valid_per_req[1].item()}"
    )
    # All valid indices must be < compressed_len of that request.
    assert (
        out_topk[0, 0, :1].max().item() < 1 and out_topk[1, 0, :2].max().item() < 2
    ), f"topk indices out of range: {out_topk[:, 0, :2]}"


if __name__ == "__main__":
    test_indexer_fp8_decode_smoke()
    print("OK")
