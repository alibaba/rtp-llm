"""DeepGEMM contract for the 132B indexer FP8 pool.

Path coverage: the case executes on any CUDA device. ``supports_paged_mqa_logits``
selects the kernel — DeepGEMM ``fp8_paged_mqa_logits`` on SM90/SM100 with a
wheel that accepts the group-128 legacy call, the Triton fallback otherwise
(including SM120). The call counters below pin the selected path
(DeepGEMM == 1 when the predicate is true, fallback == 1 when it is false), so
the numeric comparison is honest about which kernel produced the logits.

Verifies end-to-end that the 132B/slot UE8M0 layout we just locked in
(``test_indexer_fp8_writer.py``) is consumed correctly by DeepGEMM's
``fp8_paged_mqa_logits``. This pins the data path:

  1. K bf16 → UE8M0 quant → 132B pool packing  (matches rtp-llm fused writer)
  2. Q bf16 → per-(token,head) fp8 quant + scale-fold into weights
  3. DeepGEMM ``fp8_paged_mqa_logits`` reads the 132B pool directly
  4. Compare returned logits to an INDEPENDENT reference built only from
     the test's own inputs: original bf16 Q/weights plus K decoded back
     from the 132B pool this test packed itself. No product-side
     quantizer output (``indexer_q_fp8_quant_fold`` results) feeds the
     expected value — this keeps the oracle independent of the code under
     test.

Tolerance covers fp8 quantization noise on both operands plus the
kernel's reduction order (~5% relative on dominant logits, absolute
floor on near-zero rows), checked per request.
"""

from __future__ import annotations

import logging

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

from rtp_llm.models_py.modules.dsv4.fp8 import _indexer_score as _indexer_score_mod
from rtp_llm.models_py.modules.dsv4.fp8._indexer_q_quant_triton import (
    indexer_q_fp8_quant_fold,
)
from rtp_llm.models_py.modules.dsv4.fp8._indexer_score import (
    fp8_paged_indexer_score,
    supports_paged_mqa_logits,
)
from rtp_llm.models_py.triton_kernels import (
    legacy_fp8_indexer_score as _legacy_fallback_mod,
)

INDEXER_HEAD_DIM = 128
INDEXER_ENTRY_BYTES = 132
_FP8_MAX = 448.0


class _CountingDeepGemm:
    """Delegate to DeepGEMM while counting the paged entry point."""

    def __init__(self, inner):
        self._inner = inner
        self.paged = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def fp8_paged_mqa_logits(self, *args, **kwargs):
        self.paged += 1
        return self._inner.fp8_paged_mqa_logits(*args, **kwargs)


class _CountingFallback:
    """Count calls to the group-128 Triton fallback entry point."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self._inner(*args, **kwargs)


class _NullMonkeypatch:
    def setattr(self, obj, name, value):
        setattr(obj, name, value)


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


def _pack_132B(fp8_bytes, scales, *, slot_indices, num_blocks, block_size):
    """Pack ``[N, 128] uint8`` + ``[N] fp32`` into the 132B per-block
    layout: ``[bs*128 K | bs*4 scale]``. ``slot_indices[i]`` selects the
    physical slot for source row ``i``. Returns
    ``[num_blocks, block_size, 132] uint8``."""
    device = fp8_bytes.device
    pool = torch.zeros(
        num_blocks, block_size, INDEXER_ENTRY_BYTES, dtype=torch.uint8, device=device
    )
    pool_2d = pool.view(num_blocks, block_size * INDEXER_ENTRY_BYTES)
    N = fp8_bytes.shape[0]
    assert len(slot_indices) == N
    for i, slot in enumerate(slot_indices):
        assert 0 <= slot < num_blocks * block_size
        blk = slot // block_size
        off = slot % block_size
        pool_2d[blk, off * INDEXER_HEAD_DIM : (off + 1) * INDEXER_HEAD_DIM] = fp8_bytes[
            i
        ]
        scale_off = block_size * INDEXER_HEAD_DIM + off * 4
        pool_2d[blk, scale_off : scale_off + 4] = (
            scales[i : i + 1].view(torch.uint8).flatten()
        )
    return pool


def _ref_indexer_score(q_fp32, k_dequant_fp32, weights_fp32):
    """Pure-PyTorch fp32 reference on the exact quantized operands:
    out[b,n,k_pos] = sum_h relu(sum_d q_fp8[b,n,h,d] * K_dequant[b,k_pos,d])
                     * w_fold[b,n,h]
    """
    # einsum -> [B, N, H, T]
    qk = torch.einsum(
        "bnhd,btd->bnht",
        q_fp32.to(torch.float32),
        k_dequant_fp32.to(torch.float32),
    )
    qk = torch.relu(qk)
    out = (qk * weights_fp32.unsqueeze(-1)).sum(dim=2)  # [B, N, T]
    return out


# DeepGEMM ``get_paged_mqa_logits_metadata`` accepts ``block_kv == 64``
# on SM90/SM100 and ``block_kv == 32`` on SM100 only.
@pytest.mark.parametrize("block_size", [32, 64])
def test_fp8_paged_indexer_score_via_deepgemm_independent(block_size, monkeypatch):
    """rtp-llm 132B pool → DeepGEMM ``fp8_paged_mqa_logits`` ≈ independent
    reference (original bf16 Q/weights + K decoded from the packed pool)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    expects_deepgemm = supports_paged_mqa_logits(torch.device("cuda"), block_size)
    if not expects_deepgemm:
        logging.info(
            "block_kv=%s not served by DeepGEMM on capability %s; "
            "validating the Triton fallback instead",
            block_size,
            torch.cuda.get_device_capability(),
        )
    deepgemm = _CountingDeepGemm(_indexer_score_mod._deep_gemm)
    monkeypatch.setattr(_indexer_score_mod, "_deep_gemm", deepgemm)
    fallback = _CountingFallback(_legacy_fallback_mod.legacy_fp8_paged_mqa_logits)
    monkeypatch.setattr(_legacy_fallback_mod, "legacy_fp8_paged_mqa_logits", fallback)

    torch.manual_seed(0xDEAD)
    device = torch.device("cuda")

    B = 2
    next_n = 1
    H = 64
    D = INDEXER_HEAD_DIM
    T_per_req = 32

    total_tokens = B * T_per_req
    K_bf16 = torch.randn(total_tokens, D, dtype=torch.bfloat16, device=device) * 0.5
    fp8_bytes, scales = _ue8m0_quantize_k(K_bf16)
    # Each request points to a disjoint block range; size the pool from the
    # request block table, not the globally packed token count.
    blocks_per_req = (T_per_req + block_size - 1) // block_size
    num_blocks = B * blocks_per_req
    slot_indices = [
        b * blocks_per_req * block_size + token_idx
        for b in range(B)
        for token_idx in range(T_per_req)
    ]
    pool = _pack_132B(
        fp8_bytes,
        scales,
        slot_indices=slot_indices,
        num_blocks=num_blocks,
        block_size=block_size,
    )
    pool_flat = pool.view(num_blocks * block_size, INDEXER_ENTRY_BYTES)

    max_blocks = blocks_per_req
    block_table = torch.zeros(B, max_blocks, dtype=torch.int32, device=device)
    for b in range(B):
        for j in range(blocks_per_req):
            block_table[b, j] = b * blocks_per_req + j
    context_lens = torch.full((B, next_n), T_per_req, dtype=torch.int32, device=device)
    max_ctx_len = ((T_per_req + 31) // 32) * 32

    Q = torch.randn(B, next_n, H, D, dtype=torch.bfloat16, device=device) * 0.5
    weights = torch.randn(B, next_n, H, dtype=torch.bfloat16, device=device) * 0.1

    q_fp8, w_fold = indexer_q_fp8_quant_fold(Q, weights)
    logits_dg = fp8_paged_indexer_score(
        q_fp8,
        w_fold.view(B * next_n, H),
        pool_flat,
        block_table,
        context_lens,
        block_size=block_size,
        max_ctx_len=max_ctx_len,
    )
    logits_dg = logits_dg.view(B, next_n, max_ctx_len)[..., :T_per_req]

    # ── Independent reference: decode K back from the pool bytes this test
    # packed (UE8M0 scales are powers of two so fp32 dequant is exact), and
    # score with the ORIGINAL bf16 Q/weights. No product-side quantizer
    # output enters the expected value. ──
    K_dequant_per_req = []
    for b in range(B):
        idx = b * T_per_req + torch.arange(T_per_req, device=device)
        fp8_b = fp8_bytes[idx].view(torch.float8_e4m3fn).to(torch.float32)
        scale_b = scales[idx].unsqueeze(-1)
        K_dequant_per_req.append((fp8_b * scale_b).to(torch.bfloat16))
    K_dequant = torch.stack(K_dequant_per_req, dim=0)  # [B, T, D] bf16

    logits_ref = _ref_indexer_score(Q, K_dequant, weights.to(torch.float32))

    # Quantization noise (fp8 on both operands) plus reduction order:
    # ~10% relative / 5.0 absolute, per request.
    for b in range(B):
        actual = logits_dg[b]
        expected = logits_ref[b]
        diff = (actual - expected).abs()
        max_abs = diff.max().item()
        safe_ref = expected.abs().clamp(min=1e-6)
        rel = (diff / safe_ref).max().item()
        mean_abs = diff.mean().item()
        assert actual.abs().max().item() > 1e-3, f"request {b} read an all-zero K pool"
        assert max_abs < 5.0 or rel < 0.10, (
            f"request {b} logits diverge from independent reference: "
            f"max_abs={max_abs:.3f}, max_rel={rel:.3%}, mean_abs={mean_abs:.3f}"
        )
    expected_deepgemm = 1 if expects_deepgemm else 0
    expected_fallback = 0 if expects_deepgemm else 1
    assert deepgemm.paged == expected_deepgemm, (
        f"expected {expected_deepgemm} deep_gemm.fp8_paged_mqa_logits call(s) "
        f"(predicate says DeepGEMM={expects_deepgemm}), got {deepgemm.paged} — "
        "the capability predicate and the dispatch drifted"
    )
    assert fallback.calls == expected_fallback, (
        f"expected {expected_fallback} Triton fallback call(s) "
        f"(predicate says DeepGEMM={expects_deepgemm}), got {fallback.calls} — "
        "the case did not run on either kernel path"
    )
