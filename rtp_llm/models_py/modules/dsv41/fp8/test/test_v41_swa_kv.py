"""V4.1 fresh/cached SWA KV follows the reference materialization order."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_kv import swa_kv_norm_rope
from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_triton import (
    dequantize_swa_k_cache,
    quantize_and_insert_swa_k_cache,
)


def reference_norm_rope(raw, weight, frequencies, eps=1e-6):
    values = raw.float()
    normalized = (
        values
        * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
        * weight.float()
    ).bfloat16()
    tail = normalized[..., -64:].float().reshape(-1, 32, 2)
    frequencies = torch.view_as_real(frequencies.reshape(-1, 32))
    real, imag = tail[..., 0], tail[..., 1]
    cos, sin = frequencies[..., 0], frequencies[..., 1]
    rotated = normalized.clone()
    rotated[..., -64:] = (
        torch.stack(
            (
                (real.double() * cos.double() - (imag * sin).double()).float(),
                (real.double() * sin.double() + (imag * cos).double()).float(),
            ),
            dim=-1,
        )
        .reshape(*raw.shape[:-1], 64)
        .bfloat16()
    )
    return rotated


def reference_swa_kv(raw, weight, frequencies, eps=1e-6):
    rotated = reference_norm_rope(raw, weight, frequencies, eps)
    groups = rotated.float().reshape(-1, 16, 32)
    scales = torch.exp2(
        torch.ceil(
            torch.log2(groups.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448)
        )
    )
    quantized = (groups / scales).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (quantized.float() * scales).bfloat16().reshape(raw.shape)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class V41SwaKvTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(417)

    def inputs(self, batch, length, stride=1792):
        raw = torch.randn(batch, length, stride, device="cuda", dtype=torch.bfloat16)[
            ..., -512:
        ]
        weight = (torch.randn(512, device="cuda") * 0.1 + 1).bfloat16()
        angles = torch.randn(batch * length, 32, device="cuda")
        frequencies = torch.polar(torch.ones_like(angles), angles)
        return raw, weight, frequencies

    @torch.no_grad()
    def test_reference_order_and_input_ownership(self):
        for batch, length in ((1, 1), (1, 25), (3, 6), (2, 129)):
            with self.subTest(batch=batch, length=length):
                raw, weight, frequencies = self.inputs(batch, length)
                before = raw.clone()
                output = swa_kv_norm_rope(raw, weight, frequencies, 64)
                expected = reference_swa_kv(raw, weight, frequencies)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                torch.testing.assert_close(raw, before, rtol=0, atol=0)

    @torch.no_grad()
    def test_batch_chunk_and_cp_row_partition_invariance(self):
        raw, weight, frequencies = self.inputs(4, 25)
        all_rows = swa_kv_norm_rope(raw, weight, frequencies, 64).reshape(-1, 512)
        raw_rows = raw.view(-1, 512)
        for indices in (
            torch.arange(0, 100, 2, device="cuda"),
            torch.arange(1, 100, 2, device="cuda"),
            torch.arange(75, 100, device="cuda"),
        ):
            selected = swa_kv_norm_rope(
                raw_rows.index_select(0, indices),
                weight,
                frequencies.index_select(0, indices),
                64,
            )
            torch.testing.assert_close(
                selected, all_rows.index_select(0, indices), rtol=0, atol=0
            )

    @torch.no_grad()
    def test_reused_prefix_and_fresh_rows_have_identical_values(self):
        raw, weight, frequencies = self.inputs(1, 153)
        fresh = swa_kv_norm_rope(raw, weight, frequencies, 64).reshape(-1, 512)
        pool = torch.empty(2, 128, 528, device="cuda", dtype=torch.uint8)
        slots = torch.arange(153, device="cuda", dtype=torch.int64)
        quantize_and_insert_swa_k_cache(fresh, pool, slots)
        cached = dequantize_swa_k_cache(pool, slots)
        torch.testing.assert_close(cached, fresh, rtol=0, atol=0)
        for prefix in (0, 25, 128):
            mixed = torch.cat((cached[:prefix], fresh[prefix:]), dim=0)
            torch.testing.assert_close(mixed, fresh, rtol=0, atol=0)

    @torch.no_grad()
    def test_empty_rows(self):
        raw, weight, frequencies = self.inputs(1, 0)
        self.assertEqual(
            swa_kv_norm_rope(raw, weight, frequencies, 64).shape, raw.shape
        )


if __name__ == "__main__":
    unittest.main()
