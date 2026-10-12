"""Execute cache programs on CPU, excluding the interpreter's broken FP8 casts.

Triton 3.7.1's _convert_float rounds ties upward and loses mantissa carry
(1.0625 -> 1.125 and 1.9375 -> 1.0 for E4M3). Use exact E4M3 operands here;
arbitrary FP8 rounding is covered by the CPU oracle and the SM120 hardware gate.
"""

import os
import unittest

import torch

from rtp_llm.models.test.dsv41_cache_codec_test import (
    FP8_LEVELS,
    HALFWAYS,
    TAGS,
    codec,
    load_codec_source,
    make_cache,
    reference_rows,
)


@unittest.skipUnless(
    os.environ.get("TRITON_INTERPRET") == "1", "Set TRITON_INTERPRET=1"
)
class V41CacheTritonInterpreterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernels = load_codec_source(
            "_dsv41_interpreter", "fp8/_v41_cache_triton.py"
        )

    def test_encode_decode_at_page_boundaries(self):
        generator = torch.Generator().manual_seed(411)
        for tag in TAGS:
            dim = 128 if tag == "indexer_kv" else 512
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(tag=tag, dtype=dtype):
                    cache, storage = make_cache(tag, gap=64)
                    expected, expected_storage = make_cache(tag, gap=64)
                    values = torch.randn(7, dim, generator=generator).to(dtype)
                    if tag == "swa_kv":
                        # Exact FP8 lattice values still exercise all storage paths.
                        codes = torch.randint(0, 127, (7, dim), generator=generator)
                        values = torch.tensor(FP8_LEVELS)[codes].to(dtype)
                        values[:, 1::2].neg_()
                    else:
                        # Keep random FP4 rounding with exactly representable scales.
                        values.clamp_(-6, 6)
                        group = 16 if tag in ("csa_kv", "hca_kv") else 32
                        values[:, ::group] = 6
                    values[0] = torch.tensor(HALFWAYS * (dim // 16))
                    values[1].zero_()
                    values[-1].fill_(float("nan"))
                    entries = cache.layout.entries
                    slots = torch.tensor(
                        [
                            0,
                            entries - 1,
                            entries,
                            2 * entries - 1,
                            2 * entries,
                            3 * entries - 1,
                            -1,
                        ]
                    )
                    expected.scatter(
                        reference_rows(values[:-1], tag), slots[:-1], backend="eager"
                    )
                    self.kernels.encode_rows(cache, slots, values)
                    torch.testing.assert_close(
                        storage, expected_storage, rtol=0, atol=0
                    )
                    out = torch.empty_like(values)
                    self.kernels.decode_rows(cache, slots, out)
                    torch.testing.assert_close(
                        out,
                        expected.read(slots, out_dtype=dtype, backend="eager"),
                        rtol=0,
                        atol=0,
                    )
                    if tag != "swa_kv":
                        self.assertTrue(bool(torch.signbit(out[0, 8])))

    def test_global_scale_floor_and_exact_scale_values(self):
        for tag in ("csa_kv", "hca_kv"):
            for value in (0.0, 6 * 2.0**-9, 3.0, 6.0, 9.0, 12.0, 3000.0):
                with self.subTest(tag=tag, value=value):
                    cache, _ = make_cache(tag)
                    values = torch.full((1, 512), value)
                    slots = torch.tensor([0])
                    self.kernels.encode_rows(cache, slots, values)
                    torch.testing.assert_close(
                        cache.gather(slots, backend="eager"),
                        reference_rows(values, tag),
                        rtol=0,
                        atol=0,
                    )

    def test_raw_transport_relocation_and_padding(self):
        for tag in TAGS:
            source, _ = make_cache(tag, gap=64)
            target, storage = make_cache(tag, gap=64)
            expected, expected_storage = make_cache(tag, gap=64)
            width = (
                source.layout.payload_bytes_per_entry
                + source.layout.scale_bytes_per_entry
            )
            rows = (
                torch.arange(5 * width).remainder(256).to(torch.uint8).reshape(5, width)
            )
            entries = source.layout.entries
            src_slots = torch.tensor(
                [0, entries - 1, entries, 2 * entries + 1, -1], dtype=torch.int32
            )
            dst_slots = torch.tensor(
                [2 * entries - 1, 0, 2 * entries, entries + 3, -1], dtype=torch.int32
            )
            source.scatter(rows, src_slots, backend="eager")
            gathered = torch.empty_like(rows)
            self.kernels.transfer_rows(source, src_slots, gathered, scatter=False)
            rows[-1].zero_()
            torch.testing.assert_close(gathered, rows, rtol=0, atol=0)
            self.kernels.transfer_rows(target, dst_slots, gathered, scatter=True)
            expected.scatter(rows, dst_slots, backend="eager")
            torch.testing.assert_close(storage, expected_storage, rtol=0, atol=0)

    def test_decode_ue8m0_special_scale_codes(self):
        for tag in ("indexer_kv", "swa_kv"):
            cache, _ = make_cache(tag)
            payload = cache.layout.payload_bytes_per_entry
            rows = torch.zeros(
                2, payload + cache.layout.scale_bytes_per_entry, dtype=torch.uint8
            )
            rows[:, :payload] = 0x22 if tag == "indexer_kv" else 0x38
            rows[1, payload:] = 255
            slots = torch.tensor([0, 1])
            cache.scatter(rows, slots, backend="eager")
            out = torch.empty(2, 128 if tag == "indexer_kv" else 512)
            self.kernels.decode_rows(cache, slots, out)
            torch.testing.assert_close(
                out,
                codec.dequantize_rows_eager(rows, tag, torch.float32),
                rtol=0,
                atol=0,
                equal_nan=True,
            )


if __name__ == "__main__":
    unittest.main()
