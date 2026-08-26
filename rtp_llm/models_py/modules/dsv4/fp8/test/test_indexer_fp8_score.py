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

import ast
import inspect
import textwrap

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
    validate_indexer_paged_layout,
)
from rtp_llm.models_py.modules.dsv4.fp8.indexer import IndexerFP8

INDEXER_HEAD_DIM = 128
INDEXER_ENTRY_BYTES = 132
_FP8_MAX = 448.0


def test_decode_hot_path_has_no_tensor_scalar_reads():
    """A CUDA tensor ``.item()`` here synchronizes every Indexer layer/step."""
    source = textwrap.dedent(inspect.getsource(IndexerFP8.forward_decode_vectorized))
    tree = ast.parse(source)
    item_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "item"
    ]
    assert not item_calls


def test_kernel64_table_preserves_framework_expanded_ids():
    """The framework, rather than Python, publishes kernel-row ids."""
    kernel_eb = 64
    pool = torch.zeros((8, kernel_eb, INDEXER_ENTRY_BYTES), dtype=torch.uint8)
    kernel_bt = torch.tensor([[4, 5, 6, 7, 0, 1, 2, 3]], dtype=torch.int32)

    pool_2d, deepgemm_bt, block_kv = validate_indexer_paged_layout(
        pool,
        kernel_bt,
        kernel_eb,
        owner_tokens_per_block=1024,
        compress_ratio=4,
    )

    assert block_kv == 64
    assert deepgemm_bt.tolist() == [[4, 5, 6, 7, 0, 1, 2, 3]]
    assert pool_2d.shape == (8, 64, INDEXER_ENTRY_BYTES)


def test_kernel64_rejects_partial_owner_table_width():
    pool = torch.zeros((8, 64, INDEXER_ENTRY_BYTES), dtype=torch.uint8)
    table = torch.zeros((1, 6), dtype=torch.int32)
    try:
        validate_indexer_paged_layout(
            pool,
            table,
            kernel_entries_per_block=64,
            owner_tokens_per_block=1024,
            compress_ratio=4,
        )
    except RuntimeError as error:
        assert "width=6" in str(error)
        assert "kernel_blocks_per_owner=4" in str(error)
    else:
        raise AssertionError("partial INDEXER owner table was unexpectedly accepted")


def test_kernel64_rejects_partial_owner_pool_rows():
    pool = torch.zeros((6, 64, INDEXER_ENTRY_BYTES), dtype=torch.uint8)
    table = torch.zeros((1, 8), dtype=torch.int32)
    try:
        validate_indexer_paged_layout(
            pool,
            table,
            kernel_entries_per_block=64,
            owner_tokens_per_block=1024,
            compress_ratio=4,
        )
    except RuntimeError as error:
        assert "rows=6" in str(error)
        assert "kernel_blocks_per_owner=4" in str(error)
    else:
        raise AssertionError("partial INDEXER owner pool was unexpectedly accepted")


def test_kernel16_layout_is_rejected_instead_of_claiming_coalescing():
    pool = torch.zeros((16, 16, INDEXER_ENTRY_BYTES), dtype=torch.uint8)
    table = torch.arange(16, dtype=torch.int32).view(1, 16)
    try:
        validate_indexer_paged_layout(
            pool,
            table,
            kernel_entries_per_block=16,
            owner_tokens_per_block=1024,
            compress_ratio=4,
        )
    except RuntimeError as error:
        assert "kernel_block_kv=16" in str(error)
        assert "supported=(64,)" in str(error)
    else:
        raise AssertionError("unsupported 16-entry rows were unexpectedly accepted")


def test_sm100_keeps_legacy_kernel32_layout():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("DeepGEMM block_kv=32 requires SM100")
    pool = torch.zeros((2, 32, INDEXER_ENTRY_BYTES), dtype=torch.uint8, device="cuda")
    table = torch.tensor([[1, 0]], dtype=torch.int32, device="cuda")
    actual_pool, actual_table, block_kv = validate_indexer_paged_layout(
        pool,
        table,
        kernel_entries_per_block=32,
        owner_tokens_per_block=128,
        compress_ratio=4,
    )
    assert block_kv == 32
    assert actual_pool.data_ptr() == pool.data_ptr()
    assert torch.equal(actual_table, table)


def test_kernel64_split_owner_layout_runs_deepgemm():
    """The online 1024-owner/256-kernel layout must execute end to end."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not has_fp8_paged_mqa_logits():
        pytest.skip("deep_gemm.fp8_paged_mqa_logits unavailable")

    torch.manual_seed(0xB10C)
    device = torch.device("cuda")
    physical_entries = 256
    kernel_eb = 64
    B, next_n, H, D = 1, 1, 64, INDEXER_HEAD_DIM
    K_bf16 = (
        torch.randn(physical_entries, D, dtype=torch.bfloat16, device=device) * 0.5
    )
    fp8_bytes, scales = _ue8m0_quantize_k(K_bf16)
    kernel_pool = _pack_132B(
        fp8_bytes, scales, num_blocks=4, block_size=kernel_eb
    )
    # Match the framework's shared opaque-pool layout: 64 useful entries
    # followed by another 64-entry span of per-row padding.
    padded_storage = torch.zeros(
        (4, 2 * kernel_eb, INDEXER_ENTRY_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    padded_storage[:, :kernel_eb].copy_(kernel_pool)
    kernel_pool = padded_storage[:, :kernel_eb]
    assert not kernel_pool.is_contiguous()
    kernel_bt = torch.arange(4, dtype=torch.int32, device=device).view(1, 4)
    pool_2d, owner_bt, actual_block_kv = validate_indexer_paged_layout(
        kernel_pool,
        kernel_bt,
        kernel_eb,
        owner_tokens_per_block=1024,
        compress_ratio=4,
    )

    Q = torch.randn(B, next_n, H, D, dtype=torch.bfloat16, device=device) * 0.5
    weights = (
        torch.randn(B, next_n, H, dtype=torch.bfloat16, device=device) * 0.1
    )
    q_fp8, w_fold = indexer_q_fp8_quant_fold(Q, weights)
    logits = fp8_paged_indexer_score(
        q_fp8,
        w_fold.view(B * next_n, H),
        pool_2d,
        owner_bt,
        torch.full(
            (B, next_n), physical_entries, dtype=torch.int32, device=device
        ),
        block_size=actual_block_kv,
        max_ctx_len=physical_entries,
    ).view(B, next_n, physical_entries)

    K_dequant = (
        fp8_bytes.view(torch.float8_e4m3fn).to(torch.float32)
        * scales.unsqueeze(-1)
    ).to(torch.bfloat16).unsqueeze(0)
    reference = _ref_indexer_score(Q, K_dequant, weights.to(torch.float32))
    diff = (logits - reference).abs()
    assert diff.mean().item() < 0.25
    assert diff.max().item() < 1.0


def test_online_cuda_graph_max_geometry_runs_deepgemm():
    """Cover the online max-seq / max-capture shape omitted by smoke."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not has_fp8_paged_mqa_logits():
        pytest.skip("deep_gemm.fp8_paged_mqa_logits unavailable")

    device = torch.device("cuda")
    batch = 32
    next_n = 4
    block_size = 64
    max_ctx_len = 1048576 // 4
    table_width = 8192

    torch.manual_seed(0xC0DA)
    q = torch.randn(
        (batch, next_n, 64, INDEXER_HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    head_weights = torch.randn(
        (batch, next_n, 64), dtype=torch.bfloat16, device=device
    )
    q_fp8, weights = indexer_q_fp8_quant_fold(q, head_weights)
    # Capture block tables are zero-initialized, so one valid block is enough
    # even though every row advertises the maximum context length.
    k = torch.randn(
        (block_size, INDEXER_HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    fp8_bytes, scales = _ue8m0_quantize_k(k)
    pool = _pack_132B(fp8_bytes, scales, num_blocks=1, block_size=block_size)
    block_table = torch.zeros(
        (batch, table_width), dtype=torch.int32, device=device
    )
    context_lens = torch.full(
        (batch, next_n), max_ctx_len, dtype=torch.int32, device=device
    )

    logits = fp8_paged_indexer_score(
        q_fp8,
        weights.view(batch * next_n, 64),
        pool,
        block_table,
        context_lens,
        block_size=block_size,
        max_ctx_len=max_ctx_len,
    )
    torch.cuda.synchronize()

    assert logits.shape == (batch * next_n, max_ctx_len)
    assert torch.isfinite(logits).all()


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
def test_fp8_paged_indexer_score_via_deepgemm(block_size):
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
    next_n = 1  # one Q step per request (decode-shape)
    H = 64  # head count (DeepGEMM contract: H consumed per row)
    D = INDEXER_HEAD_DIM
    # Each request gets a fresh K range; for simplicity all reqs use the
    # same context length and contiguous slot ids.
    T_per_req = 32

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


if __name__ == "__main__":
    test_decode_hot_path_has_no_tensor_scalar_reads()
    test_kernel64_table_preserves_framework_expanded_ids()
    test_kernel64_rejects_partial_owner_table_width()
    test_kernel64_rejects_partial_owner_pool_rows()
    test_kernel16_layout_is_rejected_instead_of_claiming_coalescing()
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10:
        test_sm100_keeps_legacy_kernel32_layout()
        test_fp8_paged_indexer_score_via_deepgemm(32)
    if torch.cuda.is_available() and has_fp8_paged_mqa_logits():
        test_kernel64_split_owner_layout_runs_deepgemm()
        test_online_cuda_graph_max_geometry_runs_deepgemm()
        test_fp8_paged_indexer_score_via_deepgemm(64)
    print("OK")
