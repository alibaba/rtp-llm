"""Phase 4D — Attention.forward_decode_fp8 tests.

Tests for the FP8 KV cache decode path:
  * Buffer shape / dtype checks (no CUDA required).
  * Dispatch routing: forward_decode → _forward_decode_fp8 after enable.
  * Output shape matches BF16 path (dispatch + shape only; actual FP8 quant
    is mocked because quantize_v4_kv_decode requires head_dim=512 + CUDA).
  * E2E slot-remapping formula verification (pure math, no CUDA).

The FP8 quantize kernel (quantize_v4_kv_decode) requires head_dim=512 and
the rtp_llm CUDA compute_ops .so. Tests that need it are marked skipUnless
HAS_CUDA. Tests that only need the dispatch/shape contract use mock.patch to
bypass the kernel.
"""

import os
import sys
import unittest
from unittest import mock

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from rtp_llm.models_py.modules.dsv4.attention import Attention
from rtp_llm.models_py.modules.dsv4.decode.decode_attn_metadata import (
    build_decode_metadata,
)
from rtp_llm.models_py.modules.dsv4.decode.fp8_kv_quant_decode_op import ENTRY_BYTES

HAS_CUDA = torch.cuda.is_available()


def _make_attention(
    compress_ratio: int,
    dim: int = 128,
    n_heads: int = 4,
    head_dim: int = 32,
    max_batch_size: int = 2,
    max_seq_len: int = 64,
    window_size: int = 8,
) -> Attention:
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        attn = Attention(
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
            index_head_dim=16,
            index_topk=4,
            norm_eps=1e-6,
            weights=None,
            prefix="",
            tp_size=1,
            tp_rank=0,
        )
    finally:
        torch.set_default_dtype(prev_dtype)
    return attn


def _seed_attn(attn: Attention, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    for name, p in attn.named_parameters():
        if not p.dtype.is_floating_point:
            continue
        try:
            p.data = torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.05
        except (NotImplementedError, RuntimeError):
            tmp = torch.randn(p.shape, generator=g, dtype=torch.float32) * 0.05
            p.data = tmp.to(p.dtype)
    if attn.q_norm.weight is not None:
        attn.q_norm.weight.data.fill_(1.0)
    attn.kv_norm.weight.data.fill_(1.0)
    if attn.compressor is not None:
        attn.compressor.norm.weight.data.fill_(1.0)
    attn.attn_sink.data.zero_()


# --------------------------------------------------------------------------
# Buffer / constant tests — no CUDA, no quantize_v4_kv_decode needed
# --------------------------------------------------------------------------


class TestKvCacheFp8Buffer(unittest.TestCase):
    """kv_cache_fp8 buffer is registered with the correct shape and dtype."""

    def test_swa_buffer_shape(self):
        attn = _make_attention(compress_ratio=0, window_size=8)
        # SWA-only: block_size == window_size
        self.assertEqual(attn.kv_cache_fp8.shape, (2, 8, ENTRY_BYTES))
        self.assertEqual(attn.kv_cache_fp8.dtype, torch.uint8)

    def test_hca_buffer_shape(self):
        attn = _make_attention(compress_ratio=128, window_size=8, max_seq_len=256)
        # HCA: block_size = win + max_seq_len//128 = 8 + 2 = 10
        expected_block = 8 + 256 // 128
        self.assertEqual(attn.kv_cache_fp8.shape, (2, expected_block, ENTRY_BYTES))

    def test_csa_buffer_shape(self):
        attn = _make_attention(compress_ratio=4, window_size=8, max_seq_len=64)
        expected_block = 8 + 64 // 4
        self.assertEqual(attn.kv_cache_fp8.shape, (2, expected_block, ENTRY_BYTES))

    def test_fp8_flag_initially_false(self):
        attn = _make_attention(compress_ratio=0)
        self.assertFalse(attn._fp8_kv_enabled)

    def test_fp8_block_size_equals_bf16_kv_cache_t_dim(self):
        """kv_cache_fp8 block_size must match kv_cache T dim so slot indices align."""
        for ratio in (0, 4, 128):
            with self.subTest(ratio=ratio):
                attn = _make_attention(compress_ratio=ratio, window_size=8, max_seq_len=64)
                self.assertEqual(
                    attn.kv_cache_fp8.shape[1],
                    attn.kv_cache.shape[1],
                    f"Block size mismatch for ratio={ratio}",
                )


# --------------------------------------------------------------------------
# Slot remapping math tests — pure Python/torch, no CUDA
# --------------------------------------------------------------------------


class TestFp8SlotRemapping(unittest.TestCase):
    """FP8 SWA slot remapping formula: fp8 = (bf16//win)*block + (bf16%win)."""

    def test_swa_only_identity(self):
        """For SWA-only (ratio=0): block_size==win, so fp8 slot == bf16 slot."""
        win = 8
        block = win  # SWA-only: no compressed tail
        for r in range(3):
            for ring_pos in range(win):
                bf16 = r * win + ring_pos
                fp8 = (bf16 // win) * block + (bf16 % win)
                self.assertEqual(bf16, fp8)

    def test_hca_remap_non_trivial(self):
        """For HCA (ratio=128): block_size=win+C > win, remap shifts r."""
        win = 8
        C = 2  # max_seq_len=256, ratio=128
        block = win + C  # = 10
        # Request r=1, ring_pos=3: bf16 = 1*8+3 = 11, fp8 = 1*10+3 = 13
        r, ring_pos = 1, 3
        bf16 = r * win + ring_pos
        fp8 = (bf16 // win) * block + (bf16 % win)
        self.assertEqual(fp8, r * block + ring_pos)
        self.assertNotEqual(bf16, fp8)

    def test_compressed_remap(self):
        """Compressed-K FP8 slot = r*block + win + c."""
        win = 8
        C = 4
        block = win + C
        stride_bf16 = C  # max_seq_len//ratio = C
        r, c = 1, 2
        bf16_cmp = r * stride_bf16 + c
        cmp_r = bf16_cmp // stride_bf16
        cmp_c = bf16_cmp % stride_bf16
        fp8_cmp = cmp_r * block + win + cmp_c
        self.assertEqual(fp8_cmp, r * block + win + c)


# --------------------------------------------------------------------------
# Dispatch tests — use mock.patch to bypass quantize_v4_kv_decode
# --------------------------------------------------------------------------

# The function is imported lazily inside _forward_decode_fp8 / enable_fp8_kv_cache,
# so we patch at the definition site.
_QUANT_PATH = "rtp_llm.models_py.modules.dsv4.decode.fp8_kv_quant_decode_op.quantize_v4_kv_decode"


def _noop_quantize(k_bf16, slot_mapping, kv_cache_packed):
    """No-op replacement so we can test dispatch without CUDA kernel."""
    pass


class TestFp8Dispatch(unittest.TestCase):
    """forward_decode dispatches to _forward_decode_fp8 after enable.

    The full FP8 path (quantize_v4_kv_decode, SparseAttnV4DecodeFp8Op) requires
    head_dim=512 + CUDA, so dispatch tests mock _forward_decode_fp8 itself to
    verify the routing without exercising the FP8 quantization kernels.
    The actual output-correctness tests live in the smoke suite (SM100_ARM).
    """

    def _make_meta(self, attn, compress_ratio, start_pos_val, bsz, device):
        start_pos = torch.tensor([start_pos_val] * bsz, dtype=torch.int32, device=device)
        return build_decode_metadata(
            start_pos=start_pos,
            q_len=1,
            window_size=attn.window_size,
            head_dim=attn.head_dim,
            max_seq_len=64,
            compress_ratios=[compress_ratio],
            index_topk=4,
            device=device,
        )

    def test_flag_true_routes_to_fp8(self):
        """When _fp8_kv_enabled=True, forward_decode calls _forward_decode_fp8."""
        torch.manual_seed(0)
        device, dim, bsz = "cpu", 128, 1
        attn = _make_attention(compress_ratio=0).to(device).eval()
        _seed_attn(attn)
        attn.reset_rope_cache(device=device)

        with torch.no_grad():
            attn.forward(torch.randn(bsz, 4, dim, dtype=torch.bfloat16) * 0.1, start_pos=0)

        expected = torch.zeros(bsz, 1, dim, dtype=torch.bfloat16)
        attn._fp8_kv_enabled = True
        with mock.patch.object(Attention, "_forward_decode_fp8", return_value=expected) as mock_fp8:
            meta = self._make_meta(attn, 0, 4, bsz, device)
            x = torch.randn(bsz, 1, dim, dtype=torch.bfloat16) * 0.1
            with torch.no_grad():
                out = attn.forward_decode(x, meta)
            mock_fp8.assert_called_once()
        self.assertIs(out, expected)

    def test_flag_false_skips_fp8(self):
        """When _fp8_kv_enabled=False, _forward_decode_fp8 is NOT called."""
        torch.manual_seed(1)
        device, dim, bsz = "cpu", 128, 1
        attn = _make_attention(compress_ratio=0).to(device).eval()
        _seed_attn(attn)
        attn.reset_rope_cache(device=device)

        with torch.no_grad():
            attn.forward(torch.randn(bsz, 4, dim, dtype=torch.bfloat16) * 0.1, start_pos=0)

        self.assertFalse(attn._fp8_kv_enabled)
        with mock.patch.object(Attention, "_forward_decode_fp8") as mock_fp8:
            meta = self._make_meta(attn, 0, 4, bsz, device)
            x = torch.randn(bsz, 1, dim, dtype=torch.bfloat16) * 0.1
            with torch.no_grad():
                attn.forward_decode(x, meta)
            mock_fp8.assert_not_called()

    def test_flag_true_swa_shape(self):
        """For SWA-only layer: _forward_decode_fp8 is called, output shape is right."""
        torch.manual_seed(2)
        device, dim, bsz = "cpu", 128, 1
        attn = _make_attention(compress_ratio=0).to(device).eval()
        _seed_attn(attn)
        attn.reset_rope_cache(device=device)
        with torch.no_grad():
            attn.forward(torch.randn(bsz, 4, dim, dtype=torch.bfloat16) * 0.1, start_pos=0)

        sentinel = torch.ones(bsz, 1, dim, dtype=torch.bfloat16) * 7.0
        attn._fp8_kv_enabled = True
        with mock.patch.object(Attention, "_forward_decode_fp8", return_value=sentinel):
            meta = self._make_meta(attn, 0, 4, bsz, device)
            x = torch.randn(bsz, 1, dim, dtype=torch.bfloat16) * 0.1
            with torch.no_grad():
                out = attn.forward_decode(x, meta)
        # shape and value pass through from _forward_decode_fp8
        self.assertEqual(out.shape, (bsz, 1, dim))
        self.assertTrue(torch.all(out == 7.0))

    def test_flag_true_hca_shape(self):
        """For HCA layer: _forward_decode_fp8 is called, output shape is right."""
        torch.manual_seed(3)
        device, dim, bsz = "cpu", 128, 1
        attn = _make_attention(compress_ratio=128).to(device).eval()
        _seed_attn(attn)
        attn.reset_rope_cache(device=device)
        with torch.no_grad():
            attn.forward(torch.randn(bsz, 4, dim, dtype=torch.bfloat16) * 0.1, start_pos=0)

        sentinel = torch.ones(bsz, 1, dim, dtype=torch.bfloat16) * 3.0
        attn._fp8_kv_enabled = True
        with mock.patch.object(Attention, "_forward_decode_fp8", return_value=sentinel):
            meta = self._make_meta(attn, 128, 4, bsz, device)
            x = torch.randn(bsz, 1, dim, dtype=torch.bfloat16) * 0.1
            with torch.no_grad():
                out = attn.forward_decode(x, meta)
        self.assertEqual(out.shape, (bsz, 1, dim))


# --------------------------------------------------------------------------
# enable_fp8_kv_cache integration (skipped without CUDA)
# --------------------------------------------------------------------------


class TestEnableFp8KvCacheIntegration(unittest.TestCase):
    """enable_fp8_kv_cache correctly populates kv_cache_fp8 (needs CUDA + head_dim=512)."""

    @unittest.skipUnless(HAS_CUDA, "requires CUDA for concat_and_cache_mla kernel")
    def test_enable_populates_fp8_buffer(self):
        """After enable_fp8_kv_cache, kv_cache_fp8 has non-zero bytes for written KV."""
        from rtp_llm.models_py.modules.dsv4.decode.fp8_kv_quant_decode_op import (
            NOPE_DIM, ROPE_DIM,
        )
        head_dim = NOPE_DIM + ROPE_DIM  # 512
        device = torch.device("cuda")
        attn = _make_attention(
            compress_ratio=0, head_dim=head_dim, dim=1024,
        ).to(device).eval()
        _seed_attn(attn)
        attn.reset_rope_cache(device=device)
        attn.kv_cache[:1].normal_()
        attn.enable_fp8_kv_cache(bsz=1)
        self.assertTrue(attn._fp8_kv_enabled)
        self.assertTrue(bool((attn.kv_cache_fp8[:1] != 0).any()))


if __name__ == "__main__":
    unittest.main()
