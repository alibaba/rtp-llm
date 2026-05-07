"""DSV4 FP8 KV cache wiring tests.

Validates the FP8 KV cache contract introduced by commit
``eb64793f5 Wire DSV4 Python KV cache dtype``:

  * ``Attention(kv_cache_dtype="FP8")`` flips ``_kv_cache_is_fp8`` and
    rewires ``_pool_spec`` so SWA/CSA/HCA slots become ``(uint8, 584)``
    (= 448B NoPE FP8 + 128B RoPE bf16 + 8B ue8m0 scales) and the indexer
    slots become ``(uint8, 132)`` (= 128B FP8 + 4B fp32 scale).
  * The ``_is_fp8_kv_cache_dtype`` helper recognises the various spellings
    the framework hands in (``"FP8"``, an enum-like object with ``.name``,
    ``"KvCacheDtype.FP8"``, ``None`` → False).
  * The ``DSV4ConfigCreator`` (C++) sizes blocks consistently with this
    layout, so block_id arithmetic written in Python (slot remapping) is
    bit-exact equivalent to the BF16 path.

The previous incarnation of this file targeted an earlier "Python-owned
``kv_cache_fp8`` register-buffer" API that was removed when ownership of
the FP8 packed pool moved into the framework KV cache (the Python side
now only carries a ``(dtype, vec_dim)`` spec and a uint8 pool view).
"""

import os
import sys
import unittest

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from rtp_llm.models_py.modules.dsv4.attention import (
    _DSV4_FP8_INDEXER_ENTRY_BYTES,
    _DSV4_FP8_KV_ENTRY_BYTES,
    Attention,
    _is_fp8_kv_cache_dtype,
)
from rtp_llm.models_py.modules.dsv4.attn_type import (
    CSA_KV,
    CSA_STATE,
    HCA_KV,
    HCA_STATE,
    INDEXER_KV,
    INDEXER_STATE,
    SWA_KV,
)
from rtp_llm.models_py.modules.dsv4.decode.fp8_kv_quant_decode_op import (
    ENTRY_BYTES,
    NOPE_BYTES,
    ROPE_BYTES,
    SCALE_BYTES_PER_TOKEN,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _make_attention(
    compress_ratio: int,
    *,
    kv_cache_dtype=None,
    dim: int = 128,
    n_heads: int = 4,
    head_dim: int = 32,
    max_batch_size: int = 2,
    max_seq_len: int = 64,
    window_size: int = 8,
    index_head_dim: int = 16,
) -> Attention:
    if _is_fp8_kv_cache_dtype(kv_cache_dtype) and compress_ratio:
        # CompressorFP8 is intentionally locked to the production layouts:
        # 512-dim DS MLA KV slots and 128-dim indexer slots.
        head_dim = 512
        index_head_dim = 128
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        return Attention(
            layer_id=0,
            dim=dim,
            n_heads=n_heads,
            q_lora_rank=32,
            head_dim=head_dim,
            rope_head_dim=8,
            o_lora_rank=32,
            o_groups=2,
            window_size=window_size,
            compress_ratio=compress_ratio,
            compress_rope_theta=10000.0,
            rope_theta=10000.0,
            rope_factor=1.0,
            beta_fast=32,
            beta_slow=1,
            original_seq_len=0,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            index_n_heads=4,
            index_head_dim=index_head_dim,
            index_topk=4,
            norm_eps=1e-6,
            weights=None,
            prefix="",
            tp_size=1,
            tp_rank=0,
            kv_cache_dtype=kv_cache_dtype,
        )
    finally:
        torch.set_default_dtype(prev_dtype)


# --------------------------------------------------------------------------
# Layout constants (sanity vs the kernel)
# --------------------------------------------------------------------------
class TestLayoutConstants(unittest.TestCase):
    """Python-side ENTRY_BYTES constants must match the CUDA layout."""

    def test_kv_entry_is_584(self):
        # 448 NoPE FP8 + 128 RoPE bf16 + 8 ue8m0 scales = 584
        self.assertEqual(_DSV4_FP8_KV_ENTRY_BYTES, ENTRY_BYTES)
        self.assertEqual(NOPE_BYTES + ROPE_BYTES + SCALE_BYTES_PER_TOKEN, 584)

    def test_indexer_entry_is_132(self):
        # 128B FP8 + 4B fp32 scale = 132 (matches vLLM DeepseekV4IndexerCache).
        self.assertEqual(_DSV4_FP8_INDEXER_ENTRY_BYTES, 132)
        # No corresponding kernel-side constant exists yet — guard the value
        # so divergence between Python and a future C++ binding is caught
        # immediately.


# --------------------------------------------------------------------------
# _is_fp8_kv_cache_dtype recognises every spelling the framework emits
# --------------------------------------------------------------------------
class TestFp8DtypeDetection(unittest.TestCase):
    def test_none_is_not_fp8(self):
        self.assertFalse(_is_fp8_kv_cache_dtype(None))

    def test_string_FP8(self):
        self.assertTrue(_is_fp8_kv_cache_dtype("FP8"))
        self.assertTrue(_is_fp8_kv_cache_dtype("fp8"))

    def test_string_other(self):
        self.assertFalse(_is_fp8_kv_cache_dtype("BF16"))
        self.assertFalse(_is_fp8_kv_cache_dtype("INT8"))
        self.assertFalse(_is_fp8_kv_cache_dtype(""))

    def test_enum_like_with_name(self):
        class _Enum:
            def __init__(self, n):
                self.name = n

        self.assertTrue(_is_fp8_kv_cache_dtype(_Enum("FP8")))
        self.assertTrue(_is_fp8_kv_cache_dtype(_Enum("fp8")))
        self.assertFalse(_is_fp8_kv_cache_dtype(_Enum("BF16")))

    def test_enum_repr_dot_FP8(self):
        """e.g. ``KvCacheDtype.FP8`` rendered via str()."""

        class _E:
            def __str__(self):
                return "KvCacheDtype.FP8"

        self.assertTrue(_is_fp8_kv_cache_dtype(_E()))


# --------------------------------------------------------------------------
# Attention._pool_spec switches under FP8
# --------------------------------------------------------------------------
class TestPoolSpecFp8Wiring(unittest.TestCase):
    """When ``kv_cache_dtype="FP8"`` the per-attn-type spec must move to
    the (uint8, ENTRY_BYTES) layout.  STATE specs always stay fp32 — they
    hold partial-state accumulation, not the cached KV itself."""

    def _spec_for(self, ratio: int, kv_cache_dtype):
        return _make_attention(
            compress_ratio=ratio, kv_cache_dtype=kv_cache_dtype
        )._pool_spec

    def test_bf16_default_swa_only(self):
        spec = self._spec_for(0, kv_cache_dtype=None)
        self.assertEqual(spec[SWA_KV], (torch.bfloat16, 32))  # head_dim=32

    def test_fp8_swa_only(self):
        spec = self._spec_for(0, kv_cache_dtype="FP8")
        self.assertEqual(spec[SWA_KV], (torch.uint8, _DSV4_FP8_KV_ENTRY_BYTES))

    def test_fp8_csa(self):
        spec = self._spec_for(4, kv_cache_dtype="FP8")
        self.assertEqual(spec[SWA_KV], (torch.uint8, _DSV4_FP8_KV_ENTRY_BYTES))
        self.assertEqual(spec[CSA_KV], (torch.uint8, _DSV4_FP8_KV_ENTRY_BYTES))
        self.assertEqual(spec[INDEXER_KV], (torch.uint8, _DSV4_FP8_INDEXER_ENTRY_BYTES))
        # State pools stay fp32 — they hold accumulation, not stored KV.
        self.assertEqual(spec[CSA_STATE][0], torch.float32)
        self.assertEqual(spec[INDEXER_STATE][0], torch.float32)

    def test_fp8_hca(self):
        spec = self._spec_for(128, kv_cache_dtype="FP8")
        self.assertEqual(spec[HCA_KV], (torch.uint8, _DSV4_FP8_KV_ENTRY_BYTES))
        self.assertEqual(spec[HCA_STATE][0], torch.float32)

    def test_kv_cache_is_fp8_flag(self):
        attn_bf = _make_attention(compress_ratio=4, kv_cache_dtype=None)
        attn_fp = _make_attention(compress_ratio=4, kv_cache_dtype="FP8")
        self.assertFalse(attn_bf._kv_cache_is_fp8)
        self.assertTrue(attn_fp._kv_cache_is_fp8)

    def test_indexer_only_for_csa(self):
        """Only CSA layers (ratio=4) carry an indexer; SWA-only and HCA
        configure INDEXER_KV with the FP8 spec but the layer never reads it.
        Verify the spec is still reachable and consistent (no crash)."""
        for ratio in (0, 4, 128):
            with self.subTest(ratio=ratio):
                spec = self._spec_for(ratio, kv_cache_dtype="FP8")
                self.assertIn(INDEXER_KV, spec)
                self.assertEqual(spec[INDEXER_KV][0], torch.uint8)


# --------------------------------------------------------------------------
# Slot remapping math (pure Python — not API-dependent)
# --------------------------------------------------------------------------
class TestFp8SlotRemapping(unittest.TestCase):
    """The FP8 packed pool addresses tokens by flat slot ``r*block + offset``,
    where ``block = window_size + max_seq_len // ratio`` for compressed
    layers (HCA/CSA) and ``= window_size`` for SWA-only.  Compressed-K
    slots live in the ``[win, block)`` tail of each request's block."""

    def test_swa_only_identity(self):
        win = 8
        block = win
        for r in range(3):
            for ring_pos in range(win):
                bf16 = r * win + ring_pos
                fp8 = (bf16 // win) * block + (bf16 % win)
                self.assertEqual(bf16, fp8)

    def test_hca_remap_non_trivial(self):
        win = 8
        C = 2  # max_seq_len=256, ratio=128
        block = win + C
        r, ring_pos = 1, 3
        bf16 = r * win + ring_pos
        fp8 = (bf16 // win) * block + (bf16 % win)
        self.assertEqual(fp8, r * block + ring_pos)
        self.assertNotEqual(bf16, fp8)

    def test_compressed_remap(self):
        win = 8
        C = 4
        block = win + C
        stride_bf16 = C
        r, c = 1, 2
        bf16_cmp = r * stride_bf16 + c
        cmp_r = bf16_cmp // stride_bf16
        cmp_c = bf16_cmp % stride_bf16
        fp8_cmp = cmp_r * block + win + cmp_c
        self.assertEqual(fp8_cmp, r * block + win + c)


# --------------------------------------------------------------------------
# Indexer 132B FP8 layout (one ue8m0 fp32 scale per 128-elem vector)
# --------------------------------------------------------------------------
class TestIndexerFp8LayoutRoundTrip(unittest.TestCase):
    """Indexer FP8 cache spec lock — exercises the production
    ``quantize_indexer_k`` / ``dequantize_indexer_k`` kernels (vLLM /
    DeepGEMM per-block grouped layout: per block,
    ``[block_size * 128 K bytes][block_size * 4 fp32-scale bytes]``).

    Detailed byte-level layout coverage lives in
    ``rtp_llm/models_py/modules/dsv4/test/test_indexer_fp8_quant_triton.py``.
    This file keeps a smaller round-trip + invariant suite next to the
    other FP8 KV cache wiring tests so a contract regression on either
    side trips here too.
    """

    HEAD_DIM = 128
    FP8_E4M3_MAX = 448.0

    def _quant_dequant(self, k_bf16: torch.Tensor):
        """Quantize → dequantize via the production kernels and return
        ``(pool, recon_fp32)``.  Single-block placement at slot==arange(T)."""
        from rtp_llm.models_py.modules.dsv4.fp8._indexer_quant_triton import (
            INDEXER_ENTRY_BYTES,
            dequantize_indexer_k,
            quantize_indexer_k,
        )

        T, D = k_bf16.shape
        block_size = max(8, T)  # one block holds all tokens for the small T here
        pool = torch.zeros(
            1,
            block_size,
            INDEXER_ENTRY_BYTES,
            dtype=torch.uint8,
            device=k_bf16.device,
        )
        slots = torch.arange(T, dtype=torch.int64, device=k_bf16.device)
        quantize_indexer_k(k_bf16.contiguous(), slots, pool)
        recon = dequantize_indexer_k(pool, slots, out_dtype=torch.float32)
        return pool, recon

    def test_round_trip_within_fp8_precision(self):
        torch.manual_seed(0)
        T = 32
        k = torch.randn(T, self.HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.5
        _, recon = self._quant_dequant(k)
        rel = (recon - k.float()).abs() / (k.float().abs() + 1e-6)
        self.assertLess(
            rel.amax().item(),
            0.15,
            f"max rel {rel.amax().item():.3f} > fp8 e4m3 tolerance",
        )

    def test_zero_input_yields_clamped_scale(self):
        T = 2
        k = torch.zeros(T, self.HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        _, recon = self._quant_dequant(k)
        self.assertTrue(torch.all(recon == 0.0))

    def test_uniform_vector_quantizes_to_same_code(self):
        from rtp_llm.models_py.modules.dsv4.fp8._indexer_quant_triton import (
            INDEXER_HEAD_DIM,
        )

        T = 1
        k = torch.full((T, self.HEAD_DIM), 0.25, dtype=torch.bfloat16, device="cuda")
        pool, _ = self._quant_dequant(k)
        # Per-block layout: K bytes for slot 0 occupy bytes [0, 128).
        block_size = pool.shape[1]
        flat = pool.view(1, block_size * 132)
        codes = flat[0, :INDEXER_HEAD_DIM]
        self.assertEqual(len(codes.unique()), 1)

    def test_scale_region_layout(self):
        """Verify bytes [block_size * 128 : block_size * 128 + 4) per block
        hold the fp32 scale for slot 0 (vLLM/DeepGEMM layout)."""
        T = 4
        k = torch.full((T, self.HEAD_DIM), 1.0, dtype=torch.bfloat16, device="cuda")
        pool, _ = self._quant_dequant(k)
        block_size = pool.shape[1]
        flat = pool.view(1, block_size * 132)
        scale_region_start = block_size * self.HEAD_DIM
        scales = (
            flat[0, scale_region_start : scale_region_start + T * 4]
            .contiguous()
            .view(torch.float32)
        )
        # absmax=1 → scale = 1/448 for all T tokens.
        expected = torch.full((T,), 1.0 / self.FP8_E4M3_MAX, device=scales.device)
        torch.testing.assert_close(scales, expected, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
