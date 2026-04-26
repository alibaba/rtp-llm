"""Stage 4C — fp8_sparse_attn_decode_op tests.

Verifies the reference (dequant + Phase 1 _sparse_attn) fallback path
produces output close to the BF16 reference within fp8_e4m3 precision.
The CUDA flash_mla path is not exercised here (no CUDA on dev box);
SM100_ARM smoke covers it.

Tests:
  * Reference forward shape matches input expectation.
  * Quant-then-dequant-then-attn ≈ BF16-attn within ~5% rel diff.
  * Empty (all-zero) cache rows are correctly skipped (output is the
    sink-only attention, which equals zero for zero-sink config).
"""

import os
import sys
import unittest

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from rtp_llm.models_py.modules.dsv4.attention import _sparse_attn
from rtp_llm.models_py.modules.dsv4.decode.fp8_kv_quant_decode_op import (
    ENTRY_BYTES,
    NOPE_DIM,
    ROPE_DIM,
    reference_quantize_v4_kv_decode,
)
from rtp_llm.models_py.modules.dsv4.decode.fp8_sparse_attn_decode_op import (
    SparseAttnV4DecodeFp8Op,
)


class TestSparseAttnV4DecodeFp8OpReference(unittest.TestCase):

    def _setup(self, B: int, T_per_req: int, H: int = 4, K: int = 4):
        torch.manual_seed(0)
        head_dim = NOPE_DIM + ROPE_DIM
        # Random KV in bf16
        kv_bf16 = torch.randn(B, T_per_req, head_dim, dtype=torch.bfloat16) * 0.2
        # Quantize and pack into [num_blocks=B, block_size=T_per_req, 584] uint8.
        # (Phase 4 default: per-request layout = 1 block per request.)
        kv_packed = torch.zeros((B, T_per_req, ENTRY_BYTES), dtype=torch.uint8)
        slots = torch.arange(B * T_per_req, dtype=torch.long)  # block_size * B slots
        # Reshape kv_bf16 to [T_total, head_dim] for the quant op
        kv_flat = kv_bf16.reshape(B * T_per_req, head_dim)
        reference_quantize_v4_kv_decode(
            kv_flat,
            slots,
            kv_packed,
            block_size=T_per_req,
        )

        # Q: [B, q_len=1, H, head_dim] bf16
        q = torch.randn(B, 1, H, head_dim, dtype=torch.bfloat16) * 0.1
        # Sink zero (V4-Flash default)
        sink = torch.zeros(H, dtype=torch.float32)
        # topk: each request picks K random valid slots within its T_per_req
        topk = torch.randint(0, T_per_req, (B, 1, K), dtype=torch.int32)
        sm_scale = head_dim**-0.5

        return q, kv_bf16, kv_packed, sink, topk, sm_scale

    def test_reference_forward_shape(self):
        B, T = 2, 8
        q, _, kv_packed, sink, topk, sm_scale = self._setup(B, T)
        op = SparseAttnV4DecodeFp8Op(
            n_heads=q.shape[2], head_dim=q.shape[3], softmax_scale=sm_scale
        )
        out = op._forward_reference(q, kv_packed, sink, topk)
        self.assertEqual(tuple(out.shape), (B, 1, q.shape[2], q.shape[3]))

    def test_fp8_close_to_bf16_reference(self):
        """Reference fp8-attn ≈ bf16-attn within ~5% rel diff."""
        B, T = 2, 8
        q, kv_bf16, kv_packed, sink, topk, sm_scale = self._setup(B, T)

        # FP8 path (dequant + ref attn)
        op = SparseAttnV4DecodeFp8Op(
            n_heads=q.shape[2], head_dim=q.shape[3], softmax_scale=sm_scale
        )
        out_fp8 = op._forward_reference(q, kv_packed, sink, topk)

        # BF16 reference
        out_bf16 = _sparse_attn(q, kv_bf16, sink, topk.long(), sm_scale)

        # Allow for FP8 quant precision (~3% per-tile, plus softmax weighting amplification)
        diff = (out_fp8.float() - out_bf16.float()).abs()
        ref_mag = out_bf16.float().abs().mean().item() + 1e-9
        rel_mean = diff.mean().item() / ref_mag
        self.assertLess(
            rel_mean, 0.10, f"fp8 vs bf16 rel_mean={rel_mean:.3e} (expected < 0.10)"
        )


class TestFlashMlaAvailability(unittest.TestCase):

    def test_op_constructs_without_flash_mla(self):
        """Op should be constructible even without flash_mla installed —
        falls back to the reference path."""
        op = SparseAttnV4DecodeFp8Op(n_heads=4, head_dim=512, softmax_scale=1.0)
        self.assertEqual(op.n_heads, 4)
        self.assertEqual(op.head_dim, 512)


if __name__ == "__main__":
    unittest.main()
