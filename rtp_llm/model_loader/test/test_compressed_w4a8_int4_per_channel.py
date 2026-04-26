"""Equivalence test for the compressed-tensors W4A8 INT4 g=32 loader.

We construct a small BF16 weight, run two paths:

1. Online quant (the existing `quantize_weight_to_int4b` path used for
   non-pre-quantized BF16 ckpt).
2. Re-emit the same weight as `weight_packed` + `weight_scale` in HF
   compressed-tensors layout, then call the new `repack_compressed_int4_to_cutlass`.

Both must produce byte-identical (kernel) and value-identical (scale)
tensors so the `CutlassExpertsW4a8Int4PerChannel` executor sees the same
inputs regardless of source.
"""

import unittest

import torch


class TestCompressedW4A8Loader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from rtp_llm.model_loader.w4a8_int4_per_channel_quant_weight import (
                quantize_weight_to_int4b,
            )

            # Wrap as staticmethod so `self._quantize_online(...)` doesn't bind
            # `self` as the first positional argument.
            cls._quantize_online = staticmethod(quantize_weight_to_int4b)
        except Exception as e:
            raise unittest.SkipTest(
                f"rtp_kernel.w4a8_group_gemm not available: {e}"
            )
        from rtp_llm.model_loader.compressed_w4a8_int4_per_channel_weight import (
            repack_compressed_int4_to_cutlass,
        )

        cls._repack = staticmethod(repack_compressed_int4_to_cutlass)

    def _online_pack(self, w_bf16: torch.Tensor, group_size: int):
        return self._quantize_online(w_bf16, group_size)

    def _hf_emulate_compressed(self, w_bf16: torch.Tensor, group_size: int):
        """Emit (weight_packed [N, K/2] int8, weight_scale [N, K/group_size] bf16)
        in the HF compressed-tensors *layout*, with scaling pinned to the online
        convention so the byte-for-byte equivalence assertion below is feasible.

        Layout matches `compressed_tensors.compressors.quantized_compressors.pack_quantized.pack_to_int32`:
        each 4-bit weight is stored offset-binary (signed value + 8 → unsigned
        0..15), packed low-bits-first; reinterpreted as little-endian int8 each
        storage byte holds two adjacent nibbles
        (low = (w_lo + 8) & 0xF, high = (w_hi + 8) & 0xF).

        Scaling deliberately diverges from upstream `calculate_qparams` (which
        uses bit_range=15 → divisor 7.5) and pins divisor 7 to match
        `quantize_weight_to_int4b`. `repack_compressed_int4_to_cutlass` does
        not reconcile the divisor difference, so the test fixes both sides to
        the same convention to validate the format-conversion logic in
        isolation. The stored bf16 scale is round-tripped through fp8 so the
        `bf16→fp8` cast inside the repack is bit-exact equal to the fp8 scale
        that the online path stores (fp8_e4m3 fits exactly in bf16).
        """
        N, K = w_bf16.shape
        assert K % group_size == 0
        n_groups = K // group_size
        wg = w_bf16.view(N, n_groups, group_size)

        finfo = torch.finfo(torch.float8_e4m3fn)
        amax = wg.abs().amax(dim=2, keepdim=True)
        scale = (amax / 7.0).clamp(min=1e-12, max=finfo.max / 8.0)
        scale_bf = scale.to(torch.float8_e4m3fn).to(torch.bfloat16)

        out_int8 = torch.round(wg / scale).clamp_(min=-8, max=7).to(torch.int8)
        # offset-binary: signed_value + 8 → unsigned [0, 15]
        out_unsigned = (out_int8 + 8).to(torch.uint8)
        out_unsigned_flat = out_unsigned.flatten()
        first = out_unsigned_flat[::2]
        second = out_unsigned_flat[1::2]
        packed_u = ((second & 0xF) << 4) | (first & 0xF)
        packed = packed_u.view(torch.int8).reshape(N, K // 2).contiguous()

        # weight_scale stored in HF as bf16, shape [N, K/group_size]
        scale_bf16 = scale_bf.squeeze(-1).contiguous()
        return packed, scale_bf16

    def test_equivalence_small(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        torch.manual_seed(0)
        device = "cuda"
        N, K = 64, 128  # divisible by group=32
        group_size = 32

        w = torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.1

        online_w, online_s = self._online_pack(w, group_size)
        packed, scale_bf16 = self._hf_emulate_compressed(w, group_size)
        compressed_w, compressed_s = self._repack(packed, scale_bf16, group_size)

        self.assertEqual(online_w.shape, compressed_w.shape)
        self.assertEqual(online_s.shape, compressed_s.shape)
        self.assertTrue(
            torch.equal(online_w, compressed_w),
            "packed int4 kernel mismatch between online and compressed paths",
        )
        # Scales are stored as fp8_e4m3fn — compare bitwise via int8 view.
        online_s_bits = online_s.view(torch.int8)
        compressed_s_bits = compressed_s.view(torch.int8)
        self.assertTrue(
            torch.equal(online_s_bits, compressed_s_bits),
            "fp8 scale mismatch between online and compressed paths",
        )


if __name__ == "__main__":
    unittest.main()
