"""Pool-codec contract tests: descriptors, encode/decode_kv, caller-owned out.

Expected packed bytes come from the independent encoders defined in
``test__quant``-style conventions (rebuilt here from documented constants).
"""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv41.ppu import kv_codec

KV_DIM = 512


def _independent_global(kv):
    rows = kv.shape[0]
    x = kv.float().reshape(rows, 32, 16)
    maximum = x.abs().amax(dim=-1)
    scales = (maximum.clamp_min(6.0 * 2.0**-9) / 6.0).to(torch.float8_e4m3fn)
    scale_bytes = scales.view(torch.uint8)
    levels = torch.tensor([0.0, 1.0, 2.0, 4.0, 0.5, 1.5, 3.0, 6.0])
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], dtype=torch.uint8)
    # Independent FP64 calculation of the pinned NV FP32 reciprocal.
    inverse = scales.double().reciprocal().float()
    normalized = x * inverse.unsqueeze(-1)
    nearest = (normalized.abs().unsqueeze(-1) - levels).abs().argmin(dim=-1)
    sign = torch.signbit(normalized).to(torch.uint8) << 3
    codes = (order[nearest] | sign).reshape(rows, KV_DIM)
    payload = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return torch.cat((payload, scale_bytes), dim=1)


def _independent_swa(kv):
    rows = kv.shape[0]
    x = kv.float().reshape(rows, 16, 32)
    maximum = x.abs().amax(dim=-1).clamp_min(1e-4)
    mantissa, exponent = torch.frexp(maximum / 448.0)
    exponent = exponent - (mantissa == 0.5).to(torch.int32)
    scales = torch.exp2(exponent.float())
    payload = (x / scales.unsqueeze(-1)).clamp(-448, 448)
    payload = payload.to(torch.float8_e4m3fn).view(torch.uint8).reshape(rows, KV_DIM)
    scale_bytes = (exponent + 127).to(torch.uint8)
    return torch.cat((payload, scale_bytes), dim=1)


def _independent_mx_expected(k):
    rows = k.shape[0]
    grouped = k.float().reshape(rows, 4, 32)
    amax = grouped.abs().amax(-1).clamp_min(6.0 * 2.0**-126)
    mantissa, exponent = torch.frexp(amax * (1.0 / 6.0))
    exponent = exponent - (mantissa == 0.5).to(torch.int32)
    scales = torch.exp2(exponent.float())
    levels = torch.tensor([0.0, 1.0, 2.0, 4.0, 0.5, 1.5, 3.0, 6.0])
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], dtype=torch.uint8)
    normalized = grouped / scales.unsqueeze(-1)
    nearest = (normalized.abs().unsqueeze(-1) - levels).abs().argmin(dim=-1)
    sign = torch.signbit(normalized).to(torch.uint8) << 3
    codes = (order[nearest] | sign).reshape(rows, 128)
    payload = codes[:, 0::2] | (codes[:, 1::2] << 4)
    scale_bytes = (exponent + 127).to(torch.uint8)
    return torch.cat((payload, scale_bytes), dim=1)


class DescriptorDefaultsTest(unittest.TestCase):
    def test_global_tags(self):
        d1 = kv_codec.default_descriptor("global_kv_1")
        self.assertEqual(
            (d1.entry_bytes, d1.ratio, d1.planar, d1.window),
            (288, 1, False, None),
        )
        d2 = kv_codec.default_descriptor(
            "global_kv_2",
            SimpleNamespace(compression_ratio=2, entry_elems=288),
        )
        self.assertEqual((d2.entry_bytes, d2.ratio, d2.kind), (288, 2, "fp4_e2m1_e4m3"))

    def test_indexer_tag_is_planar_68b(self):
        d = kv_codec.default_descriptor("indexer_kv")
        self.assertEqual(
            (d.entry_bytes, d.planar, d.kind), (68, True, "fp4_e2m1_ue8m0")
        )
        self.assertEqual(d.row_dim, 128)

    def test_swa_tag_is_528b_window_128(self):
        d = kv_codec.default_descriptor("swa_kv")
        self.assertEqual(
            (d.entry_bytes, d.planar, d.window, d.kind),
            (528, False, 128, "fp8_e4m3_ue8m0"),
        )
        d = kv_codec.default_descriptor("decoder_swa_kv")
        self.assertEqual(d.entry_bytes, 528)

    def test_unknown_tag_raises(self):
        with self.assertRaises(ValueError):
            kv_codec.default_descriptor("kv")


class EncodeDecodeTest(unittest.TestCase):
    def test_global_bytes_match_independent_reference(self):
        torch.manual_seed(31)
        kv = torch.randn(5, KV_DIM, dtype=torch.bfloat16)
        kv[0].zero_()
        packed = kv_codec.encode_kv(kv, pool_tag="global_kv_1", descriptor=None)
        torch.testing.assert_close(packed, _independent_global(kv), rtol=0, atol=0)

    def test_swa_bytes_match_independent_reference(self):
        torch.manual_seed(32)
        kv = torch.randn(5, KV_DIM, dtype=torch.bfloat16)
        packed = kv_codec.encode_kv(kv, pool_tag="swa_kv", descriptor=None)
        torch.testing.assert_close(packed, _independent_swa(kv), rtol=0, atol=0)

    def test_indexer_bytes_match_independent_reference(self):
        torch.manual_seed(33)
        k = torch.randn(4, 128, dtype=torch.bfloat16)
        packed = kv_codec.encode_kv(k, pool_tag="indexer_kv", descriptor=None)
        torch.testing.assert_close(packed, _independent_mx_expected(k), rtol=0, atol=0)

    def test_decode_kv_fills_caller_owned_strided_out_and_returns_it(self):
        torch.manual_seed(34)
        kv = torch.randn(6, KV_DIM, dtype=torch.bfloat16)
        for tag, ref in (
            ("global_kv_1", _independent_global(kv)),
            ("swa_kv", _independent_swa(kv)),
        ):
            with self.subTest(tag=tag):
                packed = kv_codec.encode_kv(kv, pool_tag=tag, descriptor=None)
                backing = torch.full((6, 600), -5.0, dtype=torch.bfloat16)
                out = backing[:, :512]
                returned = kv_codec.decode_kv(
                    packed, pool_tag=tag, descriptor=None, out=out
                )
                self.assertIs(returned, out)
                expected = self._decode_reference(ref)
                torch.testing.assert_close(
                    out.float(), expected.float(), rtol=0, atol=0
                )
                # Stride padding untouched, no dtype/stride mutation.
                self.assertTrue(backing[:, 512:].eq(-5).all().item())
                self.assertEqual(out.dtype, torch.bfloat16)
                self.assertEqual(out.stride(), (600, 1))

    def test_decode_kv_indexer_out(self):
        torch.manual_seed(35)
        k = torch.randn(4, 128, dtype=torch.bfloat16)
        packed = kv_codec.encode_kv(k, pool_tag="indexer_kv", descriptor=None)
        out = torch.empty(4, 128, dtype=torch.float32)
        returned = kv_codec.decode_kv(
            packed, pool_tag="indexer_kv", descriptor=None, out=out
        )
        self.assertIs(returned, out)
        expected = self._decode_mx_reference(packed)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_decode_kv_validates_out(self):
        packed = kv_codec.encode_kv(
            torch.randn(3, KV_DIM, dtype=torch.bfloat16),
            pool_tag="global_kv_1",
            descriptor=None,
        )
        with self.assertRaises(ValueError):
            kv_codec.decode_kv(
                packed,
                pool_tag="global_kv_1",
                descriptor=None,
                out=torch.empty(3, 256, dtype=torch.bfloat16),
            )
        with self.assertRaises(ValueError):
            kv_codec.decode_kv(
                packed,
                pool_tag="global_kv_1",
                descriptor=None,
                out=torch.empty(3, 512, dtype=torch.float16),
            )

    @staticmethod
    def _decode_reference(packed):
        levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        if packed.shape[1] == 288:  # global: e2m1 payload, e4m3 scales
            payload, scale_bytes = packed[:, :256], packed[:, 256:]
            low = (payload & 15).long()
            high = (payload >> 4).long()
            codes = torch.stack((low, high), -1).reshape(payload.shape[0], -1)
            values = levels[codes & 7] * torch.where(codes.ge(8), -1.0, 1.0)
            return (
                values.reshape(-1, 32, 16)
                * scale_bytes.view(torch.float8_e4m3fn).float().unsqueeze(-1)
            ).reshape(-1, 512)
        # SWA: fp8 payload, ue8m0 scales
        payload, scale_bytes = packed[:, :512], packed[:, 512:]
        return (
            payload.view(torch.float8_e4m3fn).float().reshape(-1, 16, 32)
            * torch.exp2(scale_bytes.to(torch.float32) - 127.0).unsqueeze(-1)
        ).reshape(-1, 512)

    @staticmethod
    def _decode_mx_reference(packed):
        payload = packed[:, :64]
        scale_bytes = packed[:, 64:]
        low = (payload & 15).long()
        high = (payload >> 4).long()
        codes = torch.stack((low, high), -1).reshape(payload.shape[0], -1)
        levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        values = levels[codes & 7] * torch.where(codes.ge(8), -1.0, 1.0)
        scale = torch.exp2(scale_bytes.to(torch.float32) - 127.0)
        return (values.reshape(-1, 4, 32) * scale.unsqueeze(-1)).reshape(-1, 128)


class PoolInsertGatherTest(unittest.TestCase):
    def _pool(self, tag, entry_bytes, entries, blocks):
        pool = torch.full((blocks, entries * entry_bytes), 0x5A, dtype=torch.uint8)
        return pool, SimpleNamespace(tokens_per_block=entries)

    def test_global_insert_gather_sentinel(self):
        torch.manual_seed(36)
        kv = torch.randn(4, KV_DIM, dtype=torch.bfloat16)
        packed = kv_codec.encode_kv(kv, pool_tag="global_kv_1", descriptor=None)
        pool, desc = self._pool("global_kv_1", 288, 4, 3)
        slots = torch.tensor([0, 4, 7, 11])
        kv_codec.insert_rows(pool, "global_kv_1", desc, slots, packed)
        gathered = kv_codec.gather_rows(pool, "global_kv_1", desc, slots)
        torch.testing.assert_close(gathered, packed, rtol=0, atol=0)
        sentinel = torch.tensor([-1, 4])
        gathered = kv_codec.gather_rows(pool, "global_kv_1", desc, sentinel)
        self.assertTrue((gathered[0] == 0).all().item())
        torch.testing.assert_close(gathered[1], packed[1], rtol=0, atol=0)
        # Untouched entries stay poisoned.
        self.assertTrue((pool[1, 288:576] == 0x5A).all().item())

    def test_indexer_planar_insert_gather(self):
        torch.manual_seed(37)
        k = torch.randn(3, 128, dtype=torch.bfloat16)
        packed = kv_codec.encode_kv(k, pool_tag="indexer_kv", descriptor=None)
        pool, desc = self._pool("indexer_kv", 68, 4, 2)
        slots = torch.tensor([0, 5, 7])
        kv_codec.insert_rows(pool, "indexer_kv", desc, slots, packed)
        gathered = kv_codec.gather_rows(pool, "indexer_kv", desc, slots)
        torch.testing.assert_close(gathered, packed, rtol=0, atol=0)
        # Scale plane of untouched offset stays poisoned.
        self.assertTrue((pool[1, 4 * 64 + 32 : 4 * 64 + 36] == 0x5A).all().item())

    def test_planar_conversion_helpers(self):
        torch.manual_seed(38)
        k = torch.randn(5, 128, dtype=torch.bfloat16)
        packed = kv_codec.encode_kv(k, pool_tag="indexer_kv", descriptor=None)
        block = kv_codec.rows_to_planar_block(packed)
        self.assertEqual(tuple(block.shape), (5 * 68,))
        self.assertTrue(torch.equal(block[: 5 * 64], packed[:, :64].reshape(-1)))
        self.assertTrue(torch.equal(block[5 * 64 :], packed[:, 64:].reshape(-1)))
        torch.testing.assert_close(
            kv_codec.planar_block_to_rows(block), packed, rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
