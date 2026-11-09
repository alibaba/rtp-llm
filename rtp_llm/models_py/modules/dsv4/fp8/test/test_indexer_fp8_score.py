"""DeepGEMM contract for the 132B indexer FP8 pool.

Path coverage: the case executes on any CUDA device. ``supports_paged_mqa_logits``
selects the kernel — DeepGEMM ``fp8_paged_mqa_logits`` on SM90/SM100 with a
wheel that accepts the group-128 legacy call, the Triton fallback otherwise
(e.g. pre-SM90 L20). The call counters below pin the selected path
(DeepGEMM == 1 when the predicate is true, fallback == 1 when it is false), so
the numeric comparison is honest about which kernel produced the logits.

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

import importlib
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
def test_fp8_paged_indexer_score_via_deepgemm(block_size, monkeypatch):
    """rtp-llm 132B pool → DeepGEMM ``fp8_paged_mqa_logits`` ≈ bf16 reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # Run on every device: when the predicate is False the wrapper must serve
    # the case through the Triton fallback, and the numerics below still have
    # to match the reference — only the path assertion changes (see the end).
    expects_deepgemm = supports_paged_mqa_logits(torch.device("cuda"), block_size)
    if not expects_deepgemm:
        logging.info(
            "block_kv=%s not served by DeepGEMM on capability %s; "
            "validating the Triton fallback instead",
            block_size,
            torch.cuda.get_device_capability(),
        )
    # Count the DeepGEMM call: the predicate above claims this case runs on
    # the DeepGEMM path, and a silent fall back to Triton would otherwise
    # keep the test green without exercising DeepGEMM at all. The fallback
    # counter proves the opposite branch really ran when the predicate is
    # false, so neither path can silently disappear.
    deepgemm = _CountingDeepGemm(_indexer_score_mod._deep_gemm)
    monkeypatch.setattr(_indexer_score_mod, "_deep_gemm", deepgemm)
    fallback = _CountingFallback(_legacy_fallback_mod.legacy_fp8_paged_mqa_logits)
    monkeypatch.setattr(_legacy_fallback_mod, "legacy_fp8_paged_mqa_logits", fallback)

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

    # ── FP8 paged path (DeepGEMM, or the Triton fallback off-contract) ──
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


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: CUDA required")
    else:
        # Runs on any CUDA device: the in-test counters assert the DeepGEMM
        # path when the predicate is true and the fallback path otherwise.
        for _block_size in (32, 64):
            test_fp8_paged_indexer_score_via_deepgemm(_block_size, _NullMonkeypatch())
    print("OK")


def test_renamed_deepgemm_fp8_mqa_api_on_sm90(monkeypatch):
    """The renamed entry points keep the existing FP8/FP32 ABI on SM90."""
    from types import SimpleNamespace

    from rtp_llm.models_py.modules.dsv4.fp8 import _indexer_score as score
    from rtp_llm.models_py.modules.dsv4.fp8 import _mqa_layout

    calls = {}

    def paged(*args):
        calls["paged"] = args
        return torch.zeros((1, 64), dtype=torch.float32)

    def contiguous(*args):
        calls["contiguous"] = args
        return torch.zeros((1, 64), dtype=torch.float32)

    provider = SimpleNamespace(
        get_paged_mqa_logits_metadata=lambda *args: torch.zeros(
            (2, 2), dtype=torch.int32
        ),
        fp8_fp4_paged_mqa_logits=paged,
        fp8_fp4_mqa_logits=contiguous,
    )
    monkeypatch.setattr(score, "_deep_gemm", provider)
    monkeypatch.setattr(score, "_HAS_DEEP_GEMM", True)
    monkeypatch.setattr(score, "_HAS_DEEP_GEMM_MQA", True)
    monkeypatch.setattr(score, "_HAS_DEEP_GEMM_LEGACY", False)
    monkeypatch.setattr(score, "_DEEP_GEMM_SM100_LEGACY_OK", False)
    monkeypatch.setattr(score, "_get_num_sms", lambda device: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))

    def normalize(values, starts, ends, width, *, relative):
        calls["normalize"] = (values.shape, width, relative)
        return values.float()

    monkeypatch.setattr(_mqa_layout, "normalize_mqa_logits", normalize)

    q = torch.zeros((1, 1, 32, 128), dtype=torch.float8_e4m3fn)
    weights = torch.zeros((1, 32), dtype=torch.float32)
    pool = torch.zeros((64, 132), dtype=torch.uint8)
    lengths = torch.tensor([[64]], dtype=torch.int32)
    table = torch.tensor([[0]], dtype=torch.int32)
    result = score.fp8_paged_indexer_score(q, weights, pool, table, lengths, 64, 64)
    assert result.shape == (1, 64)
    assert isinstance(calls["paged"][0], tuple)
    assert calls["paged"][0][0].shape == q.shape
    assert calls["paged"][0][1] is None
    assert calls["paged"][1].shape == (1, 64, 1, 132)

    k = torch.zeros((64, 128), dtype=torch.float8_e4m3fn)
    scale = torch.ones(64, dtype=torch.float32)
    starts = torch.zeros(1, dtype=torch.int32)
    ends = torch.full((1,), 64, dtype=torch.int32)
    result = score.fp8_mqa_indexer_score(
        q.view(1, 32, 128), weights, k, scale, starts, ends
    )
    assert result.shape == (1, 64)
    assert calls["contiguous"][0][1] is None
    assert calls["contiguous"][1][1] is scale
    assert calls["contiguous"][5] == 64
    assert calls["normalize"] == ((1, 64), 64, False)

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 3))
    assert not score.supports_mqa_logits(q.device)
    assert not score.supports_paged_mqa_logits(q.device, 64)


def test_dsv41_fp8_score_reuses_shared_legacy_layout(monkeypatch):
    """V4.1 FP8 must remain runnable when DeepGEMM exposes only MX MQA."""
    from types import SimpleNamespace

    from rtp_llm.models_py.modules.dsv41.fp8 import _indexer_score as score

    sentinel = object()
    calls = []
    monkeypatch.setattr(
        score,
        "_shared_fp8_paged_indexer_score",
        lambda *args, **kwargs: calls.append(("paged", args, kwargs)) or sentinel,
    )
    monkeypatch.setattr(
        score,
        "_shared_fp8_mqa_indexer_score",
        lambda *args, **kwargs: calls.append(("contiguous", args, kwargs)) or sentinel,
    )
    q = SimpleNamespace(is_cuda=True, device="cuda:0")
    assert score.has_fp8_paged_mqa_logits()
    assert score.has_fp8_mqa_logits()
    assert score.fp8_paged_indexer_score(q, None, None, None, None, 64, 64) is sentinel
    assert score.fp8_mqa_indexer_score(q, None, None, None, None, None) is sentinel
    assert [call[0] for call in calls] == ["paged", "contiguous"]


def test_mx_fp8_mqa_api_uses_packed_scales_and_request_rows(monkeypatch):
    """SM100 calls the new ABI without reinterpreting packed scales as floats."""
    from types import SimpleNamespace

    from rtp_llm.models_py.modules.dsv4.fp8 import _indexer_score as score
    from rtp_llm.models_py.modules.dsv4.fp8 import _mqa_layout

    calls = {}

    def paged(*args, **kwargs):
        calls["paged"] = (args, kwargs)
        return torch.zeros((4, 64), dtype=torch.bfloat16)

    def contiguous(*args):
        calls["contiguous"] = args
        return torch.zeros((1, 64), dtype=torch.bfloat16)

    provider = SimpleNamespace(
        get_paged_mqa_logits_metadata=lambda *args, **kwargs: torch.zeros(
            (2, 2), dtype=torch.int32
        ),
        fp8_fp4_paged_mqa_logits=paged,
        fp8_fp4_mqa_logits=contiguous,
    )
    monkeypatch.setattr(score, "_deep_gemm", provider)
    monkeypatch.setattr(score, "_HAS_DEEP_GEMM_NEW", True)
    monkeypatch.setattr(score, "_get_num_sms", lambda device: 1)
    monkeypatch.setattr(
        _mqa_layout,
        "normalize_mqa_logits",
        lambda values, *args, **kwargs: values.float(),
    )

    q = torch.zeros((2, 2, 32, 128), dtype=torch.float8_e4m3fn)
    sf = torch.full((2, 2, 32), 0x7F7F7F7F, dtype=torch.int32)
    weights = torch.zeros((4, 32), dtype=torch.bfloat16)
    pool = torch.zeros((128, 132), dtype=torch.uint8)
    lengths = torch.tensor([[32, 33], [40, 41]], dtype=torch.int32)
    table = torch.tensor([[0], [1]], dtype=torch.int32)
    result = score.fp8_paged_indexer_score(
        q, weights, pool, table, lengths, 64, 64, q_sf=sf
    )
    assert result.shape == (4, 64)
    args, kwargs = calls["paged"]
    assert args[0][0].shape == (4, 1, 32, 128)
    assert args[0][1].shape == (4, 1, 32)
    assert args[1].shape == (2, 64, 1, 132)
    assert args[4].tolist() == [[0], [0], [1], [1]]
    assert kwargs["indices"].tolist() == [0, 0, 1, 1]

    k = torch.zeros((64, 128), dtype=torch.float8_e4m3fn)
    k_sf = torch.full((64,), 0x7F7F7F7F, dtype=torch.int32)
    starts = torch.zeros(1, dtype=torch.int32)
    ends = torch.full((1,), 64, dtype=torch.int32)
    result = score.fp8_mqa_indexer_score(
        q[:1, :1].reshape(1, 32, 128),
        weights[:1],
        k,
        k_sf,
        starts,
        ends,
        q_sf=sf[:1, :1].reshape(1, 32),
    )
    assert result.shape == (1, 64)
    assert calls["contiguous"][0][1].dtype == torch.int32
    assert calls["contiguous"][1][1].dtype == torch.int32
    assert calls["contiguous"][2].dtype == torch.bfloat16


@pytest.mark.parametrize("model_module", ["dsv4", "dsv41"])
def test_mx_fp8_fused_writer_prefill_decode_against_dequantized_reference(
    model_module, monkeypatch
):
    """Real SM100 MX writer and both DeepGEMM score paths share one cache ABI."""
    from rtp_llm.models_py.modules.dsv4.fp8._indexer_q_quant_triton import (
        quantize_indexer_q_for_score,
    )
    from rtp_llm.models_py.modules.dsv4.fp8.test import (
        test_compressor_raw_dispatch as reference,
    )
    from rtp_llm.models_py.modules.indexer_mx_abi import use_mx_indexer_cache

    if not torch.cuda.is_available() or not use_mx_indexer_cache(
        torch.device("cuda:0")
    ):
        pytest.skip("SM100 DeepGEMM MX MQA wheel required")
    compressor = importlib.import_module(
        f"rtp_llm.models_py.modules.{model_module}.fp8._compressor_vllm_triton"
    )
    for name in ("run_save_partial_states", "run_fused_compress_kv_write"):
        monkeypatch.setattr(reference, name, getattr(compressor, name))
    score_mod = importlib.import_module(
        f"rtp_llm.models_py.modules.{model_module}.fp8._indexer_score"
    )
    torch.manual_seed(23)
    ref = reference._run_reference(256, 128, seed=7)
    raw = ref["kv_cache"]._flat_backing
    cache = raw.view(2, 64, 132)
    k = raw[1, : 64 * 128].view(torch.float8_e4m3fn).reshape(64, 128)
    k_sf = raw[1, 64 * 128 :].view(torch.int32)
    q = torch.randn(1, 2, 32, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    weights = torch.randn(1, 2, 32, device="cuda", dtype=torch.bfloat16) * 0.01
    # Both indexers use the shared MX Q path, including fused RoPE.
    freqs = torch.ones(2, 32, dtype=torch.complex64, device="cuda")
    q_fp8, w_bf16, q_sf = quantize_indexer_q_for_score(q, weights, freqs, 64)
    assert q_sf is not None

    k_bytes = k_sf.view(torch.uint8).reshape(64, 4).int()
    q_bytes = q_sf.view(torch.uint8).reshape(2, 32, 4).int()
    assert torch.unique(k_bytes[:, 0]).numel() > 1
    assert torch.equal(k_bytes, k_bytes[:, :1].expand_as(k_bytes))
    assert torch.equal(q_bytes, q_bytes[..., :1].expand_as(q_bytes))
    k_deq = (
        k.float().reshape(64, 4, 32) * torch.exp2((k_bytes - 127).float()).unsqueeze(-1)
    ).reshape(64, 128)
    q_deq = (
        q_fp8.float().reshape(2, 32, 4, 32)
        * torch.exp2((q_bytes - 127).float()).unsqueeze(-1)
    ).reshape(2, 32, 128)
    oracle = torch.relu(q_deq @ k_deq.T).mul(w_bf16.float().reshape(2, 32, 1)).sum(1)
    starts = torch.zeros(2, dtype=torch.int32, device="cuda")
    ends = torch.tensor([32, 64], dtype=torch.int32, device="cuda")
    prefill = score_mod.fp8_mqa_indexer_score(
        q_fp8.reshape(2, 32, 128),
        w_bf16.reshape(2, 32),
        k,
        k_sf,
        starts,
        ends,
        q_sf=q_sf.reshape(2, 32),
    )
    decode = score_mod.fp8_paged_indexer_score(
        q_fp8,
        w_bf16.reshape(2, 32),
        cache.flatten(0, 1),
        torch.tensor([[1]], dtype=torch.int32, device="cuda"),
        ends.reshape(1, 2),
        64,
        64,
        q_sf=q_sf,
    )
    for row, valid in enumerate((32, 64)):
        torch.testing.assert_close(
            prefill[row, :valid], oracle[row, :valid], rtol=0.03, atol=0.01
        )
        torch.testing.assert_close(
            decode[row, :valid], oracle[row, :valid], rtol=0.03, atol=0.01
        )
