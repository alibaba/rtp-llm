"""Self-contained V4.1 FP4 codec regression tests for Blackwell GPUs.

Exercise GLOBAL (288B row-interleaved) and INDEX_K (68B planar) layouts,
pool roundtrips and the DeepGEMM MX-mode score contract with synthetic data.
CPU nearest-value quantization and literal known answers supply the numerical
reference; no model weights or external reference kernels are required.
"""

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8._v41_fp4_triton import (
    FP4_GLOBAL_ENTRY_BYTES,
    FP4_INDEXER_ENTRY_BYTES,
    dequantize_indexer_k_fp4,
    dequantize_k_cache_bytes_fp4,
    dequantize_k_cache_slots_fp4,
    gather_indexer_k_fp4,
    gather_k_cache_bytes_fp4,
    quantize_and_insert_k_cache_fp4,
    quantize_indexer_k_fp4,
    quantize_rows_fp4,
)


def _reference_fp4(values, group, scale_fp8):
    """CPU numerical oracle using nearest E2M1 values with ties to even."""
    rows, dimension = values.shape
    x = values.detach().cpu().float().reshape(rows, dimension // group, group)
    maximum = x.abs().amax(dim=-1)
    if scale_fp8:
        scales = (maximum.clamp_min(6.0 * 2.0**-9) / 6.0).to(torch.float8_e4m3fn)
        scale_bytes = scales.view(torch.uint8)
        scales = scales.float()
    else:
        exponent = torch.ceil(torch.log2(maximum.clamp_min(6.0 * 2.0**-126) / 6.0))
        scales = torch.exp2(exponent)
        scale_bytes = (exponent + 127).to(torch.uint8)
    normalized = x / scales.unsqueeze(-1)
    # argmin chooses the first tie: list even codes before odd codes instead
    # of reproducing the Triton encoder's threshold comparisons.
    code_order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], dtype=torch.uint8)
    levels = torch.tensor([0.0, 1.0, 2.0, 4.0, 0.5, 1.5, 3.0, 6.0])
    nearest = (normalized.abs().unsqueeze(-1) - levels).abs().argmin(dim=-1)
    codes = code_order[nearest] | (torch.signbit(x).to(torch.uint8) << 3)
    codes = codes.reshape(rows, dimension)
    payload = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return payload.to(values.device), scale_bytes.to(values.device)


def _pool(entries, entry_bytes, blocks, device):
    stride = entries * entry_bytes
    storage = torch.full((blocks, stride), 0x5A, dtype=torch.uint8, device=device)
    return storage.as_strided((blocks, entries, entry_bytes), (stride, entry_bytes, 1))


class V41Fp4CodecGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest("FP4 codec tests require a Blackwell GPU")

    def test_known_rounding_scale_and_zero_vectors(self):
        # Every E2M1 halfway point, both signs, plus the maximum to fix the
        # scale. Expected bytes and values are literal format examples.
        halfway = torch.tensor(
            [
                0.25,
                0.75,
                1.25,
                1.75,
                2.5,
                3.5,
                5.0,
                6.0,
                -0.25,
                -0.75,
                -1.25,
                -1.75,
                -2.5,
                -3.5,
                -5.0,
                -6.0,
            ],
            device="cuda",
        )
        rounded = torch.tensor(
            [0, 1, 1, 2, 2, 4, 4, 6, -0.0, -1, -1, -2, -2, -4, -4, -6],
            device="cuda",
        )
        packed = torch.tensor(
            [0x20, 0x42, 0x64, 0x76, 0xA8, 0xCA, 0xEC, 0xFE],
            dtype=torch.uint8,
            device="cuda",
        )
        cases = (
            (512, 16, True, [0.5, 1.0, 1.5, 2.0], [0x30, 0x38, 0x3C, 0x40]),
            (128, 32, False, [0.5, 1.0, 2.0, 4.0], [126, 127, 128, 129]),
        )
        for dimension, group, scale_fp8, factors, scale_codes in cases:
            with self.subTest(scale_fp8=scale_fp8):
                scales = torch.tensor(factors, device="cuda").repeat(
                    dimension // group // 4
                )
                values = halfway.repeat(dimension // 16).reshape(-1, group)
                values = (values * scales[:, None]).flatten()
                values = torch.stack((values, torch.zeros_like(values))).to(
                    torch.bfloat16
                )
                expected_payload = packed.repeat(dimension // 16)
                expected_payload = torch.stack(
                    (expected_payload, torch.zeros_like(expected_payload))
                )
                expected_scales = torch.tensor(
                    scale_codes, dtype=torch.uint8, device="cuda"
                ).repeat(dimension // group // 4)
                # Zero groups use scale byte 1: E4M3 2^-9 or UE8M0 2^-126.
                expected_scales = torch.stack(
                    (expected_scales, torch.ones_like(expected_scales))
                )
                expected_values = rounded.repeat(dimension // 16).reshape(-1, group)
                expected_values = (expected_values * scales[:, None]).flatten()
                expected_values = torch.stack(
                    (expected_values, torch.zeros_like(expected_values))
                )
                reference = _reference_fp4(values, group, scale_fp8)
                torch.testing.assert_close(
                    reference[0], expected_payload, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    reference[1], expected_scales, rtol=0, atol=0
                )

                slots = torch.tensor([128, 129], device="cuda", dtype=torch.int64)
                if scale_fp8:
                    pool = _pool(128, FP4_GLOBAL_ENTRY_BYTES, 2, values.device)
                    quantize_and_insert_k_cache_fp4(values, pool, slots)
                    raw = gather_k_cache_bytes_fp4(pool, slots)
                    payload, sf = raw[:, :256], raw[:, 256:]
                    dequant = dequantize_k_cache_slots_fp4(pool, slots)
                else:
                    pool = _pool(128, FP4_INDEXER_ENTRY_BYTES, 2, values.device)
                    quantize_indexer_k_fp4(values, slots, pool)
                    payload, sf = gather_indexer_k_fp4(pool, slots)
                    sf = sf.view(torch.uint8).reshape(2, 4)
                    dequant = dequantize_indexer_k_fp4(pool, slots)
                    row_payload, row_sf = quantize_rows_fp4(values)
                    torch.testing.assert_close(
                        row_payload.view(torch.uint8), expected_payload, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        row_sf.view(torch.uint8).reshape(2, 4),
                        expected_scales,
                        rtol=0,
                        atol=0,
                    )
                torch.testing.assert_close(
                    payload.view(torch.uint8), expected_payload, rtol=0, atol=0
                )
                torch.testing.assert_close(sf, expected_scales, rtol=0, atol=0)
                torch.testing.assert_close(
                    dequant.float(), expected_values, rtol=0, atol=0
                )

    def test_global_pool_bytes_match_reference(self):
        torch.manual_seed(7)
        rows, entries = 257, 128
        values = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
        values[0].zero_()
        values[1].fill_(1.5)
        pool = _pool(entries, FP4_GLOBAL_ENTRY_BYTES, 4, values.device)
        slots = (
            torch.arange(rows, dtype=torch.int64, device="cuda") % (entries * 3)
            + entries
        )  # spread over three pages, page 0 untouched
        quantize_and_insert_k_cache_fp4(values, pool, slots)
        expected_payload, expected_scales = _reference_fp4(values, 16, True)
        for row in range(rows):
            slot = int(slots[row])
            block, offset = slot // entries, slot % entries
            actual = pool[block, offset]
            self.assertEqual(tuple(actual[:256].shape), (256,), "payload plane width")
            torch.testing.assert_close(
                actual[:256], expected_payload[row], rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual[256:], expected_scales[row], rtol=0, atol=0
            )
        # Untouched page zero keeps its marker bytes.
        self.assertTrue((pool[0] == 0x5A).all())

    def test_indexer_pool_bytes_match_reference(self):
        torch.manual_seed(11)
        rows, entries = 300, 128
        values = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
        values[0].zero_()
        pool = _pool(entries, FP4_INDEXER_ENTRY_BYTES, 4, values.device)
        slots = (
            torch.arange(rows, dtype=torch.int64, device="cuda") % (entries * 3)
            + entries
        )
        quantize_indexer_k_fp4(values, slots, pool)
        expected_payload, expected_scales = _reference_fp4(values, 32, False)
        raw = pool.view(pool.shape[0], -1)
        for row in range(rows):
            slot = int(slots[row])
            block, offset = slot // entries, slot % entries
            payload = raw[block, offset * 64 : offset * 64 + 64]
            scales = raw[
                block, entries * 64 + offset * 4 : entries * 64 + offset * 4 + 4
            ]
            torch.testing.assert_close(payload, expected_payload[row], rtol=0, atol=0)
            torch.testing.assert_close(scales, expected_scales[row], rtol=0, atol=0)
        self.assertTrue((pool[0] == 0x5A).all())

    def test_global_dequant_matches_reference_values(self):
        torch.manual_seed(13)
        rows, entries = 96, 128
        values = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
        pool = _pool(entries, FP4_GLOBAL_ENTRY_BYTES, 2, values.device)
        slots = torch.arange(rows, dtype=torch.int64, device="cuda")
        quantize_and_insert_k_cache_fp4(values, pool, slots)
        dequant = dequantize_k_cache_slots_fp4(pool, slots)
        payload, scales = _reference_fp4(values, 16, True)
        expected = _decode_e2m1(payload) * _decode_e4m3_scale(scales).repeat_interleave(
            16, dim=-1
        )
        torch.testing.assert_close(dequant.float(), expected, rtol=0, atol=0)

    def test_global_byte_gather_then_dequant_matches_direct_dequant(self):
        """Byte-first CP transport contract: the raw 288B entries gathered per
        owned slot (zero elsewhere) dequantize to exactly the direct per-slot
        dequant, so an owner-summing all-reduce reassembles identical rows."""
        torch.manual_seed(23)
        rows, entries = 257, 128
        values = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
        pool = _pool(entries, FP4_GLOBAL_ENTRY_BYTES, 4, values.device)
        slots = torch.arange(rows, dtype=torch.int64, device="cuda")
        quantize_and_insert_k_cache_fp4(values, pool, slots)
        direct = dequantize_k_cache_slots_fp4(pool, slots)
        raw = gather_k_cache_bytes_fp4(pool, slots)
        self.assertEqual(tuple(raw.shape), (rows, FP4_GLOBAL_ENTRY_BYTES))
        # The gathered bytes are exactly the pool's 288B rows.
        pool_flat = pool.reshape(-1, FP4_GLOBAL_ENTRY_BYTES)
        torch.testing.assert_close(raw, pool_flat[slots], rtol=0, atol=0)
        # Dequantizing the gathered bytes reproduces the direct dequant.
        from_bytes = dequantize_k_cache_bytes_fp4(raw)
        torch.testing.assert_close(from_bytes, direct, rtol=0, atol=0)
        # Sentinel slots zero-fill both the bytes and the dequant output.
        sentinel = torch.tensor([-1, -1, 5, -1], dtype=torch.int64, device="cuda")
        raw_sentinel = gather_k_cache_bytes_fp4(pool, sentinel)
        self.assertTrue((raw_sentinel[[0, 1, 3]] == 0).all())
        self.assertTrue(
            (dequantize_k_cache_bytes_fp4(raw_sentinel)[[0, 1, 3]] == 0).all()
        )
        torch.testing.assert_close(
            dequantize_k_cache_bytes_fp4(raw_sentinel)[2], direct[5], rtol=0, atol=0
        )

    def test_indexer_gather_and_dequant_roundtrip(self):
        torch.manual_seed(17)
        rows, entries = 130, 128
        values = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
        pool = _pool(entries, FP4_INDEXER_ENTRY_BYTES, 2, values.device)
        slots = torch.arange(rows, dtype=torch.int64, device="cuda")
        quantize_indexer_k_fp4(values, slots, pool)
        payload, sf = gather_indexer_k_fp4(pool, slots)
        expected_payload, expected_scales = _reference_fp4(values, 32, False)
        torch.testing.assert_close(
            payload.view(torch.uint8), expected_payload, rtol=0, atol=0
        )
        torch.testing.assert_close(
            sf.view(torch.uint8).view(rows, 4), expected_scales, rtol=0, atol=0
        )
        dequant = dequantize_indexer_k_fp4(pool, slots)
        expected = _decode_e2m1(expected_payload) * _decode_ue8m0_scale(
            expected_scales
        ).repeat_interleave(32, dim=-1)
        torch.testing.assert_close(dequant, expected, rtol=0, atol=0)
        # Sentinel slots zero-fill without touching the pool.
        sentinel = torch.full((3,), -1, dtype=torch.int64, device="cuda")
        self.assertTrue((gather_indexer_k_fp4(pool, sentinel)[0] == 0).all())
        self.assertTrue((dequantize_indexer_k_fp4(pool, sentinel) == 0).all())

    def test_row_quant_matches_reference_mx_form(self):
        torch.manual_seed(19)
        values = torch.randn(64, 32, 128, device="cuda", dtype=torch.bfloat16)
        payload, sf = quantize_rows_fp4(values)
        expected_payload, expected_scales = _reference_fp4(
            values.reshape(-1, 128), 32, False
        )
        torch.testing.assert_close(
            payload.reshape(-1, 64).view(torch.uint8),
            expected_payload,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            sf.reshape(-1).view(torch.uint8).view(-1, 4),
            expected_scales,
            rtol=0,
            atol=0,
        )

    def test_deepgemm_mx_score_matches_reference(self):
        from rtp_llm.models_py.modules.dsv4.fp8._indexer_score import (
            fp8_fp4_mqa_indexer_score,
            has_fp8_fp4_mqa_logits,
        )

        if not has_fp8_fp4_mqa_logits():
            self.skipTest("deep_gemm lacks fp8_fp4_mqa_logits")
        torch.manual_seed(23)
        queries, heads, keys = 64, 32, 1000
        q = torch.randn(queries, heads, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(keys, 128, device="cuda", dtype=torch.bfloat16)
        weights = torch.randn(queries, heads, device="cuda") / (heads * 128) ** 0.5
        q_payload, q_sf = quantize_rows_fp4(q)
        k_payload, k_sf = quantize_rows_fp4(k)
        visible = torch.full((queries,), keys, dtype=torch.int32, device="cuda")
        logits = fp8_fp4_mqa_indexer_score(
            q_payload,
            q_sf,
            k_payload,
            k_sf,
            weights,
            torch.zeros_like(visible),
            visible,
            clean_logits=True,
            max_seqlen_k=0,
        )
        q_values = (
            _decode_e2m1(q_payload.reshape(-1, 64).view(torch.uint8))
            * _decode_ue8m0_scale(
                q_sf.reshape(-1).view(torch.uint8).view(-1, 4)
            ).repeat_interleave(32, dim=-1)
        ).reshape(queries, heads, 128)
        k_values = _decode_e2m1(k_payload.view(torch.uint8)) * _decode_ue8m0_scale(
            k_sf.view(torch.uint8).view(-1, 4)
        ).repeat_interleave(32, dim=-1).reshape(keys, 128)
        reference = torch.einsum("mhd,kd->mhk", q_values, k_values)
        reference = (reference.relu_() * weights[:, :, None]).sum(1)
        torch.testing.assert_close(logits, reference, rtol=2e-4, atol=2e-4)


def _decode_e2m1(payload):
    """Decode packed uint8 e2m1 payload rows to float values."""
    low = payload & 15
    high = payload >> 4
    codes = torch.stack((low, high), dim=-1).reshape(payload.shape[0], -1)
    return _e2m1_codes_to_float(codes)


def _e2m1_codes_to_float(codes):
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=codes.device)
    values = levels[(codes & 7).long()]
    return torch.where(codes.ge(8), -values, values)


def _decode_e4m3_scale(scales):
    return scales.view(torch.float8_e4m3fn).float()


def _decode_ue8m0_scale(scales):
    exponent = scales.float() - 127.0
    values = torch.exp2(exponent)
    return torch.where(scales == 255, torch.full_like(values, float("nan")), values)


if __name__ == "__main__":
    unittest.main()
