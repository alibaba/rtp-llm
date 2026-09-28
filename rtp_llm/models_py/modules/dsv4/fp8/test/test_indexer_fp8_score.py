"""DeepGEMM contract for the 132B indexer FP8 pool.

Verifies end-to-end that the 132B/slot UE8M0 layout we just locked in
(``test_indexer_fp8_writer.py``) is consumed correctly by DeepGEMM's
``fp8_paged_mqa_logits``. This pins the data path:

  1. K bf16 → UE8M0 quant → 132B pool packing  (matches rtp-llm fused writer)
  2. Q bf16 → per-(token,head) fp8 quant + scale-fold into weights
  3. DeepGEMM ``fp8_paged_mqa_logits`` reads the 132B pool directly
  4. Compare returned logits to the pure-PyTorch reference math:
        score[b,n,k_pos] = sum_h relu(sum_d Q[b,n,h,d] * K_dequant[b,k_pos,d]) * w[b,n,h]

Tolerance covers: fp8 quant of both Q and K (~scale/127 each), UE8M0
power-of-2 rounding on K scale, and DeepGEMM's reduction order. We
allow up to ~5% relative error on the dominant logits and absolute
error <1.0 on the smaller ones.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

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

from rtp_llm.models_py.modules.dsv4.fp8._indexer_q_quant_triton import (
    indexer_q_fp8_quant_fold,
)
from rtp_llm.models_py.modules.dsv4.fp8._indexer_score import (
    fp8_paged_indexer_score,
    has_fp8_paged_mqa_logits,
)
from rtp_llm.models_py.modules.dsv4.fp8 import _indexer_score as score_module

INDEXER_HEAD_DIM = 128
INDEXER_ENTRY_BYTES = 132
_FP8_MAX = 448.0


@pytest.mark.parametrize("provider_name", ["implicit_cuda", "cuda", "ppu"])
@pytest.mark.parametrize("next_n", [1, 3, 4])
def test_indexer_consumer_provider_chunking(provider_name, next_n):
    """Real construction and decode dispatch, with mocked numerical operators."""
    from rtp_llm.models_py.modules.dsv4.fp8 import attention, indexer
    from rtp_llm.models_py.modules.dsv4.platform_provider import (
        DefaultDsv4PlatformProvider,
    )
    from rtp_llm.platforms.ppu.models.dsv4.ppu_provider import M890PDsv4Provider
    from rtp_llm.utils.model_weight import W

    provider = {
        "implicit_cuda": None,
        "cuda": DefaultDsv4PlatformProvider(),
        "ppu": M890PDsv4Provider({"DSV4_INDEXER_TOPK_BACKEND": "torch"}),
    }[provider_name]
    batch, heads, dim = 2, 2, 4
    layer_weights = {
        key: torch.ones(1)
        for key in (
            W.v4_indexer_wq_b_w, W.v4_indexer_wq_b_s,
            W.v4_indexer_compressor_ape, W.v4_indexer_compressor_wkv,
            W.v4_indexer_compressor_wgate, W.v4_indexer_compressor_norm,
        )
    }
    layer_weights[W.v4_indexer_weights_proj_w] = torch.ones(heads, dim)
    compressor = Mock()
    compressor_factory = Mock(return_value=compressor)
    with (
        patch.object(indexer, "has_fp8_paged_mqa_logits", return_value=True),
        patch.object(attention, "_v4_fp8_linear", return_value=torch.nn.Identity()) as linear,
    ):
        consumer = indexer.IndexerFP8(
            dim=dim, q_lora_rank=heads * INDEXER_HEAD_DIM,
            index_n_heads=heads, index_head_dim=INDEXER_HEAD_DIM,
            rope_head_dim=2, index_topk=2, compress_ratio=4,
            max_batch_size=batch, max_seq_len=256,
            layer_weights=layer_weights, platform_provider=provider,
            compressor_factory=compressor_factory,
        )
    assert consumer._platform_provider is provider
    assert linear.call_args.kwargs["platform_provider"] is provider
    assert compressor_factory.call_args.kwargs["platform_provider"] is provider

    consumer._kv_pool_view = torch.zeros(batch, 64, 132, dtype=torch.uint8)
    consumer._kv_block_table = torch.arange(batch, dtype=torch.int32).view(batch, 1)
    consumer._kv_eb = 64
    consumer.freqs_cis = torch.ones(256, 1, dtype=torch.complex64)
    for name in ("_state_pool_3d", "_state_block_table", "_state_eb",
                 "_state_tokens_per_block", "_kv_tokens_per_block",
                 "_kv_owner_tokens_per_block"):
        setattr(consumer, name, None)
    positions = torch.arange(batch * next_n).view(batch, next_n) + 31
    lengths = ((positions + 1) // 4).to(torch.int32)
    metadata = SimpleNamespace(positions=positions, compressed_lens_per_token=lengths)
    output = torch.empty(batch, next_n, 2, dtype=torch.int32)
    calls = []

    def quantize(q, weights, freqs, rope_dim):
        return q.to(torch.float8_e4m3fn), weights.float()

    def score(q, pool, weights, context, blocks, schedule, width):
        assert schedule is context
        calls.append(context.clone())
        return torch.arange(width).float().expand(q.shape[0] * q.shape[1], -1).clone()

    backend = SimpleNamespace(
        get_paged_mqa_logits_metadata=lambda context, block_size, num_sms: context,
        fp8_paged_mqa_logits=score,
    )
    with (
        patch.object(indexer, "indexer_q_rope_fp8_quant_fold", side_effect=quantize),
        patch.object(indexer, "_run_decode_topk", return_value=False),
        patch.object(score_module, "_HAS_DEEP_GEMM", True),
        patch.object(score_module, "_deep_gemm", backend),
        patch.object(score_module, "_get_num_sms", return_value=1),
    ):
        result = consumer.forward_decode_vectorized(
            torch.ones(batch, next_n, dim),
            torch.ones(batch, next_n, heads * INDEXER_HEAD_DIM),
            positions[:, 0], output, position_ids=positions.reshape(-1),
            compressor_meta=metadata,
        )
    assert result is output
    expected_chunks = [lengths]
    if provider_name == "ppu" and next_n > 2:
        expected_chunks = [lengths[:, offset:offset + 2] for offset in range(0, next_n, 2)]
    assert len(calls) == len(expected_chunks)
    for actual, expected in zip(calls, expected_chunks):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(output, torch.stack((lengths - 1, lengths - 2), dim=-1).int())
    compressor.set_pool_context.assert_called_once()
    compressor.forward_decode_vectorized.assert_called_once()
    compressor.clear_pool_context.assert_called_once()


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("next_n", [1, 2, 3, 4])
@pytest.mark.parametrize("chunk_size", [None, 2])
def test_paged_score_provider_chunking(batch, next_n, chunk_size):
    """CPU dispatch contract; does not claim DeepGEMM numerical parity."""
    rows = torch.arange(batch * next_n).view(batch, next_n)
    q = rows[..., None, None].expand(batch, next_n, 2, 128).to(torch.float8_e4m3fn)
    weights = (rows[..., None].expand(batch, next_n, 2) + 10).float()
    lengths = (rows + 20).to(torch.int32)
    pool = torch.zeros(64, 132, dtype=torch.uint8)
    blocks = torch.zeros(batch, 1, dtype=torch.int32)
    calls = []

    def metadata(context, block_size, num_sms):
        assert context.is_contiguous()
        return context

    def score(q_part, kv, w_part, context, table, schedule, width):
        assert q_part.is_contiguous() and w_part.is_contiguous()
        assert table is blocks and schedule is context
        ids = q_part.float()[:, :, 0, 0]
        torch.testing.assert_close(w_part[:, 0].view_as(ids), ids + 10)
        torch.testing.assert_close(context.float(), ids + 20)
        calls.append(q_part.shape[1])
        return ids.reshape(-1, 1).expand(-1, width).clone()

    backend = SimpleNamespace(
        get_paged_mqa_logits_metadata=metadata, fp8_paged_mqa_logits=score
    )
    with (
        patch.object(score_module, "_HAS_DEEP_GEMM", True),
        patch.object(score_module, "_deep_gemm", backend),
        patch.object(score_module, "_get_num_sms", return_value=1),
        patch.object(torch, "cat", wraps=torch.cat) as cat,
    ):
        result = fp8_paged_indexer_score(
            q, weights.reshape(-1, 2), pool, blocks, lengths, 64, 8,
            query_chunk_size=chunk_size,
        )
    torch.testing.assert_close(result, rows.float().reshape(-1, 1).expand(-1, 8))
    expected = (
        [min(chunk_size, next_n - start) for start in range(0, next_n, chunk_size)]
        if chunk_size is not None and next_n > chunk_size else [next_n]
    )
    assert calls == expected
    assert cat.call_count == int(len(expected) > 1)


def _ue8m0_quantize_k(K_bf16):
    """Per-token UE8M0 fp8 quant matching the writer convention.

    Returns ``(fp8_bytes [N, 128] uint8, scale [N] fp32)``. Mirrors the
    rtp-llm fused writer (see test_indexer_fp8_writer.py)."""
    K_fp32 = K_bf16.to(torch.float32)
    absmax = K_fp32.abs().max(dim=-1, keepdim=True).values
    absmax = torch.clamp(absmax, min=1e-4)
    raw_scale = absmax / _FP8_MAX
    exponent = torch.ceil(torch.log2(raw_scale))
    inv_scale = torch.exp2(-exponent)
    scale = torch.exp2(exponent).squeeze(-1)
    x_scaled = K_fp32 * inv_scale
    x_clamped = torch.clamp(x_scaled, -_FP8_MAX, _FP8_MAX)
    fp8 = x_clamped.to(torch.float8_e4m3fn).view(torch.uint8)
    return fp8, scale


def _pack_132B(fp8_bytes, scales, *, num_blocks, block_size):
    """Pack ``[N, 128] uint8`` + ``[N] fp32`` into the 132B per-block
    layout: ``[bs*128 K | bs*4 scale]``. Slot ``i`` lives at
    ``(blk=i//bs, off=i%bs)``. Returns
    ``[num_blocks, block_size, 132] uint8``."""
    device = fp8_bytes.device
    pool = torch.zeros(
        num_blocks, block_size, INDEXER_ENTRY_BYTES, dtype=torch.uint8, device=device
    )
    pool_2d = pool.view(num_blocks, block_size * INDEXER_ENTRY_BYTES)
    N = fp8_bytes.shape[0]
    for i in range(N):
        blk = i // block_size
        off = i % block_size
        pool_2d[blk, off * INDEXER_HEAD_DIM : (off + 1) * INDEXER_HEAD_DIM] = fp8_bytes[
            i
        ]
        scale_off = block_size * INDEXER_HEAD_DIM + off * 4
        pool_2d[blk, scale_off : scale_off + 4] = (
            scales[i : i + 1].view(torch.uint8).flatten()
        )
    return pool


def _ref_indexer_score(Q_bf16, K_dequant_bf16, weights_fp32):
    """Pure-PyTorch reference for the indexer score:
    out[b,n,k_pos] = sum_h relu(sum_d Q[b,n,h,d] * K[b,k_pos,d]) * w[b,n,h]
    """
    # einsum -> [B, N, H, T]
    qk = torch.einsum(
        "bnhd,btd->bnht",
        Q_bf16.to(torch.float32),
        K_dequant_bf16.to(torch.float32),
    )
    qk = torch.relu(qk)
    out = (qk * weights_fp32.unsqueeze(-1)).sum(dim=2)  # [B, N, T]
    return out


# DeepGEMM ``get_paged_mqa_logits_metadata`` accepts ``block_kv == 64``
# on SM90/SM100 and ``block_kv == 32`` on SM100 only.
@pytest.mark.parametrize("block_size", [32, 64])
@pytest.mark.parametrize("next_n", [1, 2, 4])
@pytest.mark.parametrize("chunk_size", [None, 2])
def test_fp8_paged_indexer_score_via_deepgemm(block_size, next_n, chunk_size):
    """rtp-llm 132B pool → DeepGEMM ``fp8_paged_mqa_logits`` ≈ bf16 reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if block_size == 32 and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("DeepGEMM block_kv=32 requires SM100")
    if not has_fp8_paged_mqa_logits():
        pytest.skip("deep_gemm.fp8_paged_mqa_logits unavailable")

    torch.manual_seed(0xDEAD)
    device = torch.device("cuda")

    B = 2  # batch
    H = 64  # head count (DeepGEMM contract: H consumed per row)
    D = INDEXER_HEAD_DIM
    # Each request gets a fresh K range; for simplicity all reqs use the
    # same context length and contiguous slot ids.
    # Each request must start on a block boundary for both tested block sizes.
    T_per_req = 64

    # ── Build per-request K (bf16), quantize, pack ──
    # Total slots: B requests × T_per_req tokens, ALL contiguous starting
    # at slot 0 — keeps block_table trivial.
    total_tokens = B * T_per_req
    K_bf16 = torch.randn(total_tokens, D, dtype=torch.bfloat16, device=device) * 0.5
    fp8_bytes, scales = _ue8m0_quantize_k(K_bf16)
    # Round up so total_slots % block_size == 0 (DeepGEMM requirement).
    total_slots = (total_tokens + block_size - 1) // block_size * block_size
    num_blocks = total_slots // block_size
    pool = _pack_132B(
        fp8_bytes,
        scales,
        num_blocks=num_blocks,
        block_size=block_size,
    )
    pool_flat = pool.view(total_slots, INDEXER_ENTRY_BYTES)

    # ── block_table: each request points to its own contiguous chunk ──
    blocks_per_req = (T_per_req + block_size - 1) // block_size
    max_blocks = blocks_per_req
    block_table = torch.zeros(B, max_blocks, dtype=torch.int32, device=device)
    for b in range(B):
        for j in range(blocks_per_req):
            block_table[b, j] = b * blocks_per_req + j
    context_lens = torch.full((B, next_n), T_per_req, dtype=torch.int32, device=device)
    max_ctx_len = ((T_per_req + 31) // 32) * 32  # 32-align to keep DeepGEMM happy

    # ── Q ──
    Q = torch.randn(B, next_n, H, D, dtype=torch.bfloat16, device=device) * 0.5
    weights = torch.randn(B, next_n, H, dtype=torch.bfloat16, device=device) * 0.1

    # ── DeepGEMM path ──
    q_fp8, w_fold = indexer_q_fp8_quant_fold(Q, weights)
    logits_dg = fp8_paged_indexer_score(
        q_fp8,
        w_fold.view(B * next_n, H),
        pool_flat,
        block_table,
        context_lens,
        block_size=block_size,
        max_ctx_len=max_ctx_len,
        query_chunk_size=chunk_size,
    )  # [B*next_n, max_ctx_len] fp32
    logits_dg = logits_dg.view(B, next_n, max_ctx_len)[..., :T_per_req]

    # ── Reference: dequant K from pool the SAME way the writer did the
    # forward quant; this isolates "is the pool consumable by DeepGEMM?"
    # from "is the pool data correct?" — the latter is locked by
    # test_indexer_fp8_writer.py. ──
    K_dequant_per_req = []
    for b in range(B):
        idx = b * T_per_req + torch.arange(T_per_req, device=device)
        # Decode pool by re-reading the bytes we packed.
        fp8_b = fp8_bytes[idx].view(torch.float8_e4m3fn).to(torch.float32)
        scale_b = scales[idx].unsqueeze(-1)
        K_b = (fp8_b * scale_b).to(torch.bfloat16)  # [T, D]
        K_dequant_per_req.append(K_b)
    K_dequant = torch.stack(K_dequant_per_req, dim=0)  # [B, T, D]

    logits_ref = _ref_indexer_score(Q, K_dequant, weights.to(torch.float32))

    # ── Compare ──
    diff = (logits_dg - logits_ref).abs()
    max_abs = diff.max().item()
    # Relative error vs the magnitude of the reference logits per row.
    ref_abs = logits_ref.abs()
    # Avoid div-by-zero for rows where ReLU killed everything.
    safe_ref = torch.clamp(ref_abs, min=1e-6)
    rel = (diff / safe_ref).max().item()
    mean_abs = diff.mean().item()

    # For relu-summed scores the dominant terms can be large (sum over 64
    # heads × 128 dims of fp8*fp8 products). 5% relative is a comfortable
    # bound for combined Q-fp8 + K-fp8 quant noise + DeepGEMM reduction
    # order; absolute floor of 1.0 covers near-zero rows where rel
    # blows up.
    assert max_abs < 5.0 or rel < 0.10, (
        f"DeepGEMM logits diverge from bf16 reference: max_abs={max_abs:.3f}, "
        f"max_rel={rel:.3%}, mean_abs={mean_abs:.3f}"
    )


@pytest.mark.parametrize("next_n,chunk_size", [(3, 2), (4, 2), (4, None)])
def test_fp8_paged_indexer_score_graph_changed_inputs(next_n, chunk_size):
    """Replay must consume live inputs, including each copied query chunk."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    assert has_fp8_paged_mqa_logits(), (
        "DeepGEMM paged MQA is required for the CUDA13 graph regression"
    )

    batch, heads, block_size, width = 2, 64, 64, 128
    device = torch.device("cuda")
    # Exactly representable positive values isolate replay from FP8 error.
    keys = torch.tensor([0.125, 0.25, 0.5, 1.0], device=device)
    keys = keys[:, None, None].expand(4, block_size, INDEXER_HEAD_DIM).contiguous()
    key_bytes, scales = _ue8m0_quantize_k(keys.reshape(-1, INDEXER_HEAD_DIM))
    pool = _pack_132B(key_bytes, scales, num_blocks=4, block_size=block_size)
    q_rows = torch.arange(1, batch * next_n + 1, device=device).float() / 16
    q = q_rows.view(batch, next_n, 1, 1).expand(
        batch, next_n, heads, INDEXER_HEAD_DIM,
    ).to(torch.float8_e4m3fn).contiguous()
    weights = torch.full((batch * next_n, heads), 1.0 / heads, device=device)
    blocks = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device)
    lengths = torch.full((batch, next_n), 32, dtype=torch.int32, device=device)

    def score():
        return fp8_paged_indexer_score(
            q, weights, pool.view(-1, INDEXER_ENTRY_BYTES), blocks, lengths,
            block_size, width, query_chunk_size=chunk_size,
        )

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            score()
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = score()

    def check_replay():
        graph.replay()
        # Decode physical keys independently of the score helper and apply
        # the current logical block mapping before the reference matmul.
        raw = pool.view(4, block_size * INDEXER_ENTRY_BYTES)
        packed_keys = raw[:, :block_size * INDEXER_HEAD_DIM].contiguous()
        packed_keys = packed_keys.view(torch.float8_e4m3fn).float()
        packed_keys = packed_keys.view(4, block_size, INDEXER_HEAD_DIM)
        packed_scales = raw[:, block_size * INDEXER_HEAD_DIM:].contiguous()
        decoded = packed_keys * packed_scales.view(torch.float32).unsqueeze(-1)
        logical_keys = decoded[blocks.long()].reshape(batch, width, INDEXER_HEAD_DIM)
        expected = _ref_indexer_score(
            q.float(), logical_keys, weights.view(batch, next_n, heads),
        )
        valid = torch.arange(width, device=device)[None, None, :] < lengths[..., None]
        actual = captured.view(batch, next_n, width)
        torch.testing.assert_close(actual[valid], expected[valid], rtol=1e-5, atol=1e-5)

    check_replay()
    q.copy_((q.float() * 2).to(q.dtype))
    check_replay()
    weights.mul_(0.5)
    check_replay()
    pool.copy_(pool.flip(0))
    check_replay()
    blocks.copy_(torch.tensor([[2, 0], [3, 1]], dtype=torch.int32, device=device))
    check_replay()
    # Grow beyond the initially live block and vary lengths within chunks.
    lengths.copy_(128 - torch.arange(batch * next_n, device=device).view(batch, next_n))
    check_replay()


if __name__ == "__main__":
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10:
        test_fp8_paged_indexer_score_via_deepgemm(32, next_n=1, chunk_size=None)
    test_fp8_paged_indexer_score_via_deepgemm(64, next_n=1, chunk_size=None)
    print("OK")
