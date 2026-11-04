"""Byte-exact codec tests for the V4.1 PPU low-level quantization module.

Every expected byte/value is produced by independent encoders built in this
file from the documented format constants (mirroring rtp-llm's codec
oracles), never by calling the module under test.
"""

import unittest

import torch

from rtp_llm.models_py.modules.dsv41.ppu import _quant

# Independent format constants (re-derived, not imported from _quant).
KV_DIM = 512
GROUP16 = 16
GROUP32 = 32
E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Tie order: even-magnitude codes come first so argmin breaks ties to even.
CODE_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)
CODE_ORDER_LEVELS = (0.0, 1.0, 2.0, 4.0, 0.5, 1.5, 3.0, 6.0)


def _nearest_codes(normalized):
    """Nearest E2M1 codes with RNE ties via the documented tie order."""
    levels = torch.tensor(CODE_ORDER_LEVELS, dtype=torch.float32)
    ax = normalized.abs().unsqueeze(-1)
    nearest = (ax - levels).abs().argmin(dim=-1)
    order = torch.tensor(CODE_ORDER, dtype=torch.uint8)
    sign = torch.signbit(normalized).to(torch.uint8) << 3
    return order[nearest] | sign


def _pack_payload(codes):
    """codes [..., K] uint8 -> packed [..., K/2] (low nibble = element 2i)."""
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)


def _reference_global(kv):
    """Independent 288B encoder (rtp fp4-codec oracle conventions)."""
    rows = kv.shape[0]
    x = kv.float().reshape(rows, KV_DIM // GROUP16, GROUP16)
    maximum = x.abs().amax(dim=-1)
    scales = (maximum.clamp_min(6.0 * 2.0**-9) / 6.0).to(torch.float8_e4m3fn)
    scale_bytes = scales.view(torch.uint8)
    normalized = x / scales.float().unsqueeze(-1)
    codes = _nearest_codes(normalized).reshape(rows, KV_DIM)
    payload = _pack_payload(codes)
    return torch.cat((payload, scale_bytes), dim=1)


def _ceil_log2(v):
    """ceil(log2(v)) for positive fp32 v, via frexp (independent of bit
    twiddling in the module under test)."""
    mantissa, exponent = torch.frexp(v)
    return exponent - (mantissa == 0.5).to(torch.int32)


def _reference_swa(kv):
    """Independent 528B encoder (rtp swa-codec oracle conventions)."""
    rows = kv.shape[0]
    x = kv.float().reshape(rows, KV_DIM // GROUP32, GROUP32)
    maximum = x.abs().amax(dim=-1).clamp_min(1e-4)
    exponent = _ceil_log2(maximum / 448.0)
    scales = torch.exp2(exponent.float())
    payload = (x / scales.unsqueeze(-1)).clamp(-448, 448)
    payload = payload.to(torch.float8_e4m3fn).view(torch.uint8).reshape(rows, KV_DIM)
    scale_bytes = (exponent + 127).to(torch.uint8)
    return torch.cat((payload, scale_bytes), dim=1), scales


def _reference_mx(x128):
    """Independent group-32 UE8M0 MX FP4 encoder (rtp convention).

    amax floored at 6*2^-126; scale = 2**ceil(log2(amax/6)); scale byte =
    unbiased exponent + 127, packed little-endian into one int32 per row.
    """
    rows = x128.shape[0]
    grouped = x128.float().reshape(rows, 4, GROUP32)
    amax = grouped.abs().amax(-1).clamp_min(6.0 * 2.0**-126)
    exponent = _ceil_log2(amax * (1.0 / 6.0))
    scales = torch.exp2(exponent.float())
    codes = _nearest_codes(grouped / scales.unsqueeze(-1)).reshape(rows, 128)
    payload = _pack_payload(codes)
    bytes_ = (exponent + 127).to(torch.uint8)
    sf = (
        bytes_[:, 0].to(torch.int64)
        | (bytes_[:, 1].to(torch.int64) << 8)
        | (bytes_[:, 2].to(torch.int64) << 16)
        | (bytes_[:, 3].to(torch.int64) << 24)
    ).to(torch.int32)
    return payload, sf, scales


def _decode_e2m1(payload):
    low = payload & 15
    high = payload >> 4
    codes = torch.stack((low, high), dim=-1).reshape(payload.shape[0], -1)
    levels = torch.tensor(E2M1_LEVELS)
    values = levels[(codes & 7).long()]
    return torch.where(codes.ge(8), -values, values)


class E2M1Test(unittest.TestCase):
    def test_ties_round_to_even_magnitude(self):
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
            ]
        )
        expected = torch.tensor(
            [
                0.0,
                1.0,
                1.0,
                2.0,
                2.0,
                4.0,
                4.0,
                6.0,
                -0.0,
                -1.0,
                -1.0,
                -2.0,
                -2.0,
                -4.0,
                -4.0,
                -6.0,
            ]
        )
        packed = _quant.encode_fp4_e2m1(halfway)
        decoded = _quant.decode_fp4_e2m1(packed)
        torch.testing.assert_close(
            decoded.float(), expected, rtol=0, atol=0, check_dtype=False
        )
        # Literal known bytes (rtp fp4-codec oracle): low nibble = element 2i,
        # high nibble = element 2i+1, sign in bit 3.
        self.assertEqual(
            packed.view(torch.uint8).tolist(),
            [0x20, 0x42, 0x64, 0x76, 0xA8, 0xCA, 0xEC, 0xFE],
        )

    def test_roundtrip_is_stable(self):
        torch.manual_seed(101)
        x = torch.randn(64, 128) * 3
        packed = _quant.encode_fp4_e2m1(x)
        values = _quant.decode_fp4_e2m1(packed)
        again = _quant.encode_fp4_e2m1(values.float())
        torch.testing.assert_close(
            again.view(torch.uint8), packed.view(torch.uint8), rtol=0, atol=0
        )


class ScaleByteTest(unittest.TestCase):
    def test_ue8m0_byte_from_scale(self):
        scales = torch.tensor([2.0**-22, 1.0, 0.5, 2.0, 3.0, 0.75, 2.0**100])
        # ceil(log2(s)): -22, 0, -1, 1, 2, 0, 100 -> +127
        expected = torch.tensor([105, 127, 126, 128, 129, 127, 227], dtype=torch.uint8)
        torch.testing.assert_close(
            _quant.ue8m0_byte_from_scale(scales).long(), expected.long(), rtol=0, atol=0
        )
        # decode is the inverse for the exact power-of-two scales.
        back = _quant.ue8m0_scale_from_byte(expected)
        torch.testing.assert_close(
            back,
            torch.tensor([2.0**-22, 1.0, 0.5, 2.0, 4.0, 1.0, 2.0**100]),
            rtol=0,
            atol=0,
        )

    def test_e4m3_scale_zero_group_is_byte_one(self):
        amax = torch.tensor([0.0, 1e-8, 0.5, 3.0, 2688.0, 6000.0])
        bytes_ = _quant.encode_e4m3_scale(amax)
        # clamp(amax/6, 2^-9, 448): zero/small -> byte 1, 0.5/6 -> e4m3(1/12),
        # 3/6 -> e4m3(0.5)=0x30, 2688/6=448 -> 0x7E, huge clamps to 448.
        self.assertEqual(bytes_[0].item(), 1)
        self.assertEqual(bytes_[1].item(), 1)
        self.assertEqual(bytes_[3].item(), 0x30)
        self.assertEqual(bytes_[4].item(), 0x7E)
        self.assertEqual(bytes_[5].item(), 0x7E)


class GlobalKv288Test(unittest.TestCase):
    def test_bytes_match_independent_reference(self):
        torch.manual_seed(7)
        kv = torch.randn(9, KV_DIM, dtype=torch.bfloat16)
        kv[0].zero_()
        kv[1].fill_(1.5)
        kv[2, 448:].fill_(448.0)
        packed = _quant.quantize_global_kv_rows(kv)
        expected = _reference_global(kv)
        torch.testing.assert_close(packed, expected, rtol=0, atol=0)
        self.assertEqual(tuple(packed.shape), (9, 288))
        # Zero groups use scale byte 1.
        self.assertTrue((packed[0, 256:] == 1).all().item())

    def test_decode_values_exact(self):
        torch.manual_seed(11)
        kv = torch.randn(6, KV_DIM, dtype=torch.bfloat16)
        packed = _quant.quantize_global_kv_rows(kv)
        decoded = _quant.dequantize_global_kv_rows(packed)
        payload, scales = packed[:, :256], packed[:, 256:]
        expected = _decode_e2m1(payload) * scales.view(
            torch.float8_e4m3fn
        ).float().repeat_interleave(16, dim=-1)
        torch.testing.assert_close(
            decoded.float(), expected, rtol=0, atol=0, check_dtype=False
        )

    def test_pool_sentinel_slots_zero_fill(self):
        torch.manual_seed(13)
        kv = torch.randn(5, KV_DIM, dtype=torch.bfloat16)
        entries = 4
        pool = torch.full((3, entries * 288), 0x5A, dtype=torch.uint8)
        slots = torch.tensor([0, 4, 9, -1, -1])
        _quant.insert_global_kv_rows(pool, slots, kv)
        rows = _quant.dequantize_global_kv_slots(pool, slots)
        # Sentinel rows zero-fill; other rows decode exactly.
        self.assertTrue((rows[[3, 4]] == 0).all().item())
        expected = _quant.dequantize_global_kv_rows(
            _quant.gather_global_kv_rows(pool, torch.tensor([0, 4, 9]))
        )
        torch.testing.assert_close(rows[:3].float(), expected.float(), rtol=0, atol=0)
        # Untouched pool bytes stay poisoned.
        self.assertTrue((pool[2, 5 * 288 :] == 0x5A).all().item())


class SwaKv528Test(unittest.TestCase):
    def test_literal_zero_and_power_of_two_scales(self):
        groups = (
            torch.tensor([0.0, 448.0, 224.0, 896.0]).repeat_interleave(32).repeat(4)
        )
        packed = _quant.quantize_swa_kv_rows(groups.reshape(1, KV_DIM))
        self.assertEqual(packed[0, 512:].tolist(), [105, 127, 126, 128] * 4)
        # Each nonzero group rescales its max to E4M3 448 = 0x7E.
        self.assertEqual(packed[0, :128].tolist(), [0] * 32 + [0x7E] * 96)

    def test_bytes_match_independent_reference(self):
        torch.manual_seed(410528)
        kv = torch.randn(8, KV_DIM, dtype=torch.bfloat16)
        kv[0].zero_()
        kv[1].fill_(1.5)
        kv[2, 448:].fill_(448.0)
        packed = _quant.quantize_swa_kv_rows(kv)
        expected, _ = _reference_swa(kv)
        torch.testing.assert_close(packed, expected, rtol=0, atol=0)
        self.assertEqual(tuple(packed.shape), (8, 528))

    def test_scale_floor_is_1e4(self):
        # amax below the floor: every group's amax treated as 1e-4 ->
        # exponent = ceil(log2(1e-4/448)) = -22 -> byte 105.
        kv = torch.full((1, KV_DIM), 1e-9, dtype=torch.bfloat16)
        packed = _quant.quantize_swa_kv_rows(kv)
        self.assertTrue((packed[0, 512:] == 105).all().item())

    def test_decode_into_strided_out_leaves_padding_untouched(self):
        torch.manual_seed(528)
        kv = torch.randn(7, KV_DIM, dtype=torch.bfloat16)
        packed = _quant.quantize_swa_kv_rows(kv)
        backing = torch.full((7, 544), -73.0, dtype=torch.bfloat16)
        out = backing[:, :512]
        returned = _quant.dequantize_swa_kv_rows(packed, out=out)
        self.assertIs(returned, out)
        _, scales = _reference_swa(kv)
        payload = packed[:, :512].view(torch.float8_e4m3fn).float().reshape(7, 16, 32)
        expected = (
            (payload * scales.unsqueeze(-1)).reshape(7, KV_DIM).to(torch.bfloat16)
        )
        torch.testing.assert_close(out.float(), expected.float(), rtol=0, atol=0)
        # The stride padding (columns 512:544) is untouched by the decode.
        self.assertTrue(backing[:, 512:].eq(-73).all().item())


class IndexerMxTest(unittest.TestCase):
    def test_quantize_indexer_q_matches_reference(self):
        torch.manual_seed(19)
        q = torch.randn(5, 32, 128, dtype=torch.bfloat16)
        payload, sf = _quant.quantize_indexer_q(q)
        expected_payload, expected_sf, _ = _reference_mx(q.reshape(-1, 128))
        torch.testing.assert_close(
            payload.reshape(-1, 64).view(torch.uint8), expected_payload, rtol=0, atol=0
        )
        torch.testing.assert_close(sf.reshape(-1), expected_sf, rtol=0, atol=0)
        self.assertEqual(payload.dtype, torch.int8)
        self.assertEqual(sf.dtype, torch.int32)

    def test_dequantize_mx_rows_exact(self):
        torch.manual_seed(23)
        k = torch.randn(9, 128, dtype=torch.bfloat16)
        payload, sf = _quant.quantize_indexer_k_rows(k)
        values = _quant.dequantize_mx_fp4_rows(payload, sf)
        _, _, scales = _reference_mx(k)
        expected = _decode_e2m1(payload.view(torch.uint8)).reshape(
            9, 4, 32
        ) * scales.unsqueeze(-1)
        torch.testing.assert_close(values.reshape(9, 4, 32), expected, rtol=0, atol=0)

    def test_sf_packing_is_little_endian_group0_low(self):
        x = torch.zeros(1, 128, dtype=torch.float32)
        # Force distinct exponents per group: group amaxes 6*2^e.
        x[0, 0:32] = 6.0 * 2.0**-1
        x[0, 32:64] = 6.0 * 2.0**3
        x[0, 64:96] = 6.0 * 2.0**10
        x[0, 96:128] = 6.0 * 2.0**20
        payload, sf = _quant.quantize_indexer_k_rows(x)
        # scale = 2^ceil(log2(amax/6)) = 2^e -> byte = e + 127, group 0 in the
        # low byte of the packed int32 (little endian).
        expected = torch.tensor(
            [127 - 1, 127 + 3, 127 + 10, 127 + 20], dtype=torch.uint8
        ).view(torch.int32)
        self.assertEqual(sf[0].item(), expected[0].item())
        self.assertEqual(sf[0].item() & 0xFF, 127 - 1)

    def test_empty_rows_do_not_trip_ambiguous_reshape(self):
        # 0-element tensors: reshape(-1, D) / reshape(R, -1) are ambiguous and
        # raise. A ratio-2 producer hits this for requests shorter than 2
        # tokens (0 global-KV entries).
        k = torch.empty(0, 128, dtype=torch.bfloat16)
        payload, sf = _quant.quantize_indexer_k_rows(k)
        self.assertEqual(tuple(payload.shape), (0, 64))
        self.assertEqual(tuple(sf.shape), (0,))
        rows = _quant.indexer_k_packed_rows(payload, sf)
        self.assertEqual(tuple(rows.shape), (0, 68))
        self.assertEqual(rows.dtype, torch.uint8)


class IndexerPlanar68Test(unittest.TestCase):
    def test_store_and_gather_planar_layout(self):
        torch.manual_seed(17)
        k = torch.randn(6, 128, dtype=torch.bfloat16)
        payload, sf = _quant.quantize_indexer_k_rows(k)
        expected_payload, expected_sf, _ = _reference_mx(k)
        entries = 4
        pool = torch.full((3, entries * 68), 0x5A, dtype=torch.uint8)
        slots = torch.tensor([0, 4, 5, 8, 9, 11])
        _quant.store_indexer_k_planar(pool, slots, payload, sf)
        # Planar layout: payload plane entries*64, then scale plane entries*4.
        raw = pool.view(3, -1)
        for row, slot in enumerate(slots.tolist()):
            block, offset = divmod(slot, entries)
            self.assertTrue(
                torch.equal(
                    raw[block, offset * 64 : offset * 64 + 64],
                    expected_payload[row],
                )
            )
            begin = entries * 64 + offset * 4
            expected_bytes = int(expected_sf[row].item() & 0xFFFFFFFF).to_bytes(
                4, "little"
            )
            self.assertEqual(
                raw[block, begin : begin + 4].tolist(), list(expected_bytes)
            )
        gathered_payload, gathered_sf = _quant.gather_indexer_k_planar(pool, slots)
        torch.testing.assert_close(
            gathered_payload.view(torch.uint8),
            payload.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(gathered_sf, sf, rtol=0, atol=0)

    def test_sentinel_slots_zero_fill_without_touching_pool(self):
        k = torch.randn(2, 128, dtype=torch.bfloat16)
        payload, sf = _quant.quantize_indexer_k_rows(k)
        pool = torch.full((1, 4 * 68), 0x5A, dtype=torch.uint8)
        _quant.store_indexer_k_planar(pool, torch.tensor([0, 1]), payload, sf)
        sentinel = torch.full((3,), -1, dtype=torch.int64)
        gathered_payload, gathered_sf = _quant.gather_indexer_k_planar(pool, sentinel)
        self.assertTrue((gathered_payload == 0).all().item())
        self.assertTrue((gathered_sf == 0).all().item())


class PpuFastPathTest(unittest.TestCase):
    """GPU-only: the copied Triton kernels must be byte-exact vs reference."""

    def setUp(self):
        if not _quant.is_ppu():
            self.skipTest("requires a PPU device")

    def test_indexer_quant_kernel_matches_reference(self):
        torch.manual_seed(29)
        q = torch.randn(64, 32, 128, dtype=torch.bfloat16, device="cuda")
        payload, sf = _quant.quantize_indexer_q(q)
        expected_payload, expected_sf, _ = _reference_mx(q.reshape(-1, 128))
        torch.testing.assert_close(
            payload.reshape(-1, 64).view(torch.uint8).cpu(),
            expected_payload,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(sf.reshape(-1).cpu(), expected_sf, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
