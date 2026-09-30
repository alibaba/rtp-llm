"""Portable numerical and storage contracts; no native RTP libraries required."""

import bisect
import importlib.util
import math
import struct
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rtp_llm.models.dsv41_kv_cache import V41CacheBinding, dsv41_page_layouts


def load_codec_source(name, relative_path):
    # Loading by path avoids modules/__init__.py's native device imports.
    path = Path(__file__).parents[2] / "models_py/modules/dsv4" / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


codec = load_codec_source("_dsv41_codec_test_source", "v41_cache.py")
TAGS = ("csa_kv", "hca_kv", "indexer_kv", "swa_kv")
HALFWAYS = [
    0.25,
    0.75,
    1.25,
    1.75,
    2.5,
    3.5,
    5,
    6,
    -0.25,
    -0.75,
    -1.25,
    -1.75,
    -2.5,
    -3.5,
    -5,
    -6,
]
FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP8_LEVELS = tuple(
    code * 2.0**-9 if code < 8 else (1 + (code & 7) / 8) * 2.0 ** ((code >> 3) - 7)
    for code in range(127)
)


def _f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def _nearest(value, levels):
    upper = min(bisect.bisect_left(levels, value), len(levels) - 1)
    return min(
        (max(0, upper - 1), upper), key=lambda i: (abs(value - levels[i]), i & 1)
    )


def reference_rows(values, tag):
    """Scalar nearest-value oracle, independent of the threshold encoder."""
    group = 16 if tag in ("csa_kv", "hca_kv") else 32
    result = []
    for row in values.float().tolist():
        payload, scales = [], []
        for begin in range(0, len(row), group):
            chunk = row[begin : begin + group]
            maximum = max(abs(x) for x in chunk)
            if tag in ("csa_kv", "hca_kv"):
                code = _nearest(
                    min(448.0, _f32(max(maximum, 6 * 2.0**-9) / 6)), FP8_LEVELS
                )
                scale = FP8_LEVELS[code]
            else:
                floor, divisor = (
                    (1e-4, 448.0) if tag == "swa_kv" else (6 * 2.0**-126, 6.0)
                )
                product = _f32(max(maximum, _f32(floor)) * _f32(1 / divisor))
                exponent = math.ceil(math.log2(product))
                code, scale = exponent + 127, 2.0**exponent
            scales.append(code)
            levels = FP8_LEVELS if tag == "swa_kv" else FP4_LEVELS
            sign_bit = 128 if tag == "swa_kv" else 8
            codes = [
                _nearest(min(levels[-1], abs(_f32(x / scale))), levels)
                | (sign_bit if math.copysign(1.0, x) < 0 else 0)
                for x in chunk
            ]
            payload.extend(
                codes
                if tag == "swa_kv"
                else [codes[i] | (codes[i + 1] << 4) for i in range(0, group, 2)]
            )
        result.append(payload + scales)
    return torch.tensor(result, dtype=torch.uint8)


def make_cache(tag, pages=3, gen=1, gap=0):
    width = dsv41_page_layouts(gen)[tag].stride_bytes
    storage = torch.full((pages, width + gap), 173, dtype=torch.uint8)
    return codec.V41PagedCache(storage[:, :width], tag, gen), storage


class V41CacheCodecTest(unittest.TestCase):
    def test_fp4_halfway_ties_scales_and_negative_zero(self):
        expected = [0x20, 0x42, 0x64, 0x76, 0xA8, 0xCA, 0xEC, 0xFE]
        decoded = [0, 1, 1, 2, 2, 4, 4, 6, -0.0, -1, -1, -2, -2, -4, -4, -6]
        for tag, repeats, scale in (
            ("csa_kv", 32, 56),
            ("hca_kv", 32, 56),
            ("indexer_kv", 8, 127),
        ):
            rows = codec.quantize_rows_eager(torch.tensor([HALFWAYS * repeats]), tag)
            self.assertEqual(rows[0, : 8 * repeats].tolist(), expected * repeats)
            self.assertTrue(bool((rows[0, 8 * repeats :] == scale).all()))
            result = codec.dequantize_rows_eager(rows, tag, torch.float32)
            torch.testing.assert_close(
                result, torch.tensor([decoded * repeats]), rtol=0, atol=0
            )
            self.assertTrue(bool(torch.signbit(result[0, 8])))

    def test_known_scale_encodings_and_zero_floor(self):
        for tag, dim, group, scales, expected in (
            ("hca_kv", 512, 16, [0.5, 1, 1.5, 2], [0x30, 0x38, 0x3C, 0x40]),
            ("indexer_kv", 128, 32, [0.5, 1, 2, 4], [126, 127, 128, 129]),
        ):
            values = (
                torch.tensor([6 * s for s in scales])
                .repeat_interleave(group)
                .repeat(dim // (4 * group))[None]
            )
            rows = codec.quantize_rows_eager(values, tag)
            self.assertEqual(
                rows[0, dim // 2 :].tolist(), expected * (dim // (4 * group))
            )
            self.assertTrue(bool((rows[0, : dim // 2] == 0x77).all()))
            zeros = codec.quantize_rows_eager(torch.zeros(1, dim), tag)
            self.assertTrue(bool((zeros[0, dim // 2 :] == 1).all()))
            self.assertEqual(int(zeros[0, : dim // 2].sum()), 0)

    def test_swa_quantizes_all_channels_and_known_exponents(self):
        values = (
            torch.tensor([0, 448, 224, 896.0]).repeat_interleave(32).repeat(4)[None]
        )
        rows = codec.quantize_rows_eager(values, "swa_kv")
        self.assertEqual(rows[0, 512:].tolist(), [105, 127, 126, 128] * 4)
        self.assertEqual(rows[0, :512].tolist(), ([0] * 32 + [0x7E] * 96) * 4)
        torch.testing.assert_close(
            codec.dequantize_rows_eager(rows, "swa_kv", torch.float32),
            values,
            rtol=0,
            atol=0,
        )

    def test_random_bytes_against_independent_scalar_oracle(self):
        generator = torch.Generator().manual_seed(410)
        for tag in TAGS:
            dim = 128 if tag == "indexer_kv" else 512
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(tag=tag, dtype=dtype):
                    values = (
                        torch.randn(7, dim, generator=generator)
                        * torch.logspace(-4, 3, 7)[:, None]
                    ).to(dtype)
                    torch.testing.assert_close(
                        codec.quantize_rows_eager(values, tag),
                        reference_rows(values, tag),
                        rtol=0,
                        atol=0,
                    )

    def test_extreme_finite_values_and_scale_rounding_boundaries(self):
        for tag in TAGS:
            dim = 128 if tag == "indexer_kv" else 512
            for value in (0.0, 1e-38, 6 * 2.0**-9, 6.375, 7.125, 2688.0, 3000.0, 3e38):
                values = torch.full((1, dim), value)
                with self.subTest(tag=tag, value=value):
                    torch.testing.assert_close(
                        codec.quantize_rows_eager(values, tag),
                        reference_rows(values, tag),
                        rtol=0,
                        atol=0,
                    )

    def test_ue8m0_zero_and_nan_scale_codes(self):
        for tag, width, payload in (("indexer_kv", 68, 64), ("swa_kv", 528, 512)):
            rows = torch.zeros(1, width, dtype=torch.uint8)
            rows[:, :payload] = 0x22 if tag == "indexer_kv" else 0x38
            result = codec.dequantize_rows_eager(rows, tag, torch.float32)
            torch.testing.assert_close(
                result, torch.full_like(result, 2.0**-127), rtol=0, atol=0
            )
            rows[:, payload:] = 255
            self.assertTrue(
                bool(torch.isnan(codec.dequantize_rows_eager(rows, tag)).all())
            )

    def test_planar_write_preserves_unused_slots_padding_and_row_stride(self):
        for tag in TAGS:
            cache, storage = make_cache(tag, gap=64)
            layout = cache.layout
            slots = torch.tensor(
                [0, layout.entries - 1, layout.entries, 3 * layout.entries - 1, -1]
            )
            dim = 128 if tag == "indexer_kv" else 512
            values = torch.arange(5 * dim).reshape(5, dim).float() / 32
            rows = codec.quantize_rows_eager(values, tag)
            expected = storage.clone()
            for row, slot in enumerate(slots.tolist()):
                if slot < 0:
                    continue
                page, entry = divmod(slot, layout.entries)
                payload, scales = layout.entry_offsets(entry)
                p, s = layout.payload_bytes_per_entry, layout.scale_bytes_per_entry
                expected[page, payload : payload + p] = rows[row, :p]
                expected[page, scales : scales + s] = rows[row, p:]
            cache.write(values, slots, backend="eager")
            torch.testing.assert_close(storage, expected, rtol=0, atol=0)
            rows[-1].zero_()
            out = torch.empty_like(rows)
            self.assertIs(cache.gather(slots, out=out, backend="eager"), out)
            torch.testing.assert_close(out, rows, rtol=0, atol=0)
            decoded = torch.empty_like(values)
            self.assertIs(cache.read(slots, out=decoded, backend="eager"), decoded)
            torch.testing.assert_close(
                decoded,
                codec.dequantize_rows_eager(rows, tag, torch.float32),
                rtol=0,
                atol=0,
            )

    def test_relocated_transport_preserves_every_byte_without_requantizing(self):
        for tag in TAGS:
            source, _ = make_cache(tag)
            target, _ = make_cache(tag)
            width = (
                source.layout.payload_bytes_per_entry
                + source.layout.scale_bytes_per_entry
            )
            rows = (
                torch.arange(4 * width).remainder(256).to(torch.uint8).reshape(4, width)
            )
            a = torch.tensor(
                [0, source.layout.entries + 1, source.layout.entries - 1, -1]
            )
            b = torch.tensor(
                [2 * source.layout.entries + 3, 2, source.layout.entries + 4, -1]
            )
            source.scatter(rows, a, backend="eager")
            transfer = source.gather(a, backend="eager")
            target.scatter(transfer, b, backend="eager")
            rows[-1].zero_()
            torch.testing.assert_close(
                target.gather(b, backend="eager"), rows, rtol=0, atol=0
            )

    def test_ratio_two_index_mapping_and_unused_half_page(self):
        blocks = torch.tensor([2, 0])
        indices = torch.tensor([0, 127, 128, 255, -1])
        index_slots = codec.logical_to_physical_slots(blocks, indices, "indexer_kv", 2)
        self.assertEqual(index_slots.tolist(), [512, 639, 0, 127, -1])
        self.assertEqual(
            codec.logical_to_physical_slots(blocks, indices, "csa_kv", 2).tolist(),
            [256, 383, 0, 127, -1],
        )
        cache, _ = make_cache("indexer_kv")
        cache.write(torch.ones(5, 128), index_slots, backend="eager")
        self.assertTrue(bool((cache.raw[:, 128 * 64 : 256 * 64] == 173).all()))
        self.assertTrue(bool((cache.raw[:, 16384 + 128 * 4 :] == 173).all()))

    def test_empty_and_masked_batches(self):
        for tag in TAGS:
            dim = 128 if tag == "indexer_kv" else 512
            cache, _ = make_cache(tag, pages=0)
            slots = torch.empty(0, dtype=torch.int64)
            cache.write(torch.empty(0, dim), slots, backend="eager")
            self.assertEqual(cache.read(slots, backend="eager").shape, (0, dim))
            cache.scatter(cache.gather(slots, backend="eager"), slots, backend="eager")
            masked = torch.full((2,), -1)
            cache.write(torch.full((2, dim), float("nan")), masked, backend="eager")
            self.assertEqual(int(cache.gather(masked, backend="eager").sum()), 0)
            self.assertEqual(float(cache.read(masked, backend="eager").sum()), 0.0)

    def test_binding_validation_including_l20_replica(self):
        cache, _ = make_cache("hca_kv")
        binding = V41CacheBinding(20, "hca_kv", 0, True)
        fields = dict(
            tag="hca_kv",
            layer_id=0,
            read_replica_source_layer_id=20,
            seq_size_per_block=256,
            kv_cache_base=cache.raw,
            kv_scale_base=None,
        )
        view = codec.V41PagedCache.from_layer_cache(SimpleNamespace(**fields), binding)
        self.assertIs(view.raw, cache.raw)
        for key, bad in (
            ("tag", "csa_kv"),
            ("layer_id", 20),
            ("read_replica_source_layer_id", None),
            ("seq_size_per_block", 128),
            ("kv_scale_base", torch.ones(1)),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                codec.V41PagedCache.from_layer_cache(
                    SimpleNamespace(**{**fields, key: bad}), binding
                )

    def test_rejects_invalid_storage_slots_rows_and_aliases(self):
        cache, _ = make_cache("hca_kv")
        slots = torch.tensor([0, 1])
        values = torch.ones(2, 512)
        for invalid in (
            torch.tensor([-2, 0]),
            torch.tensor([768, 0]),
            slots.float(),
            slots[None],
            torch.arange(4)[::2],
        ):
            with self.assertRaises(ValueError):
                cache.write(values, invalid, backend="eager")
        for invalid in (values[:, ::2], values.to(torch.int32), values[:1]):
            with self.assertRaises(ValueError):
                cache.write(invalid, slots, backend="eager")
        with self.assertRaisesRegex(ValueError, "unique"):
            cache.write(values, torch.tensor([1, 1]), backend="eager")
        with self.assertRaisesRegex(ValueError, "CUDA"):
            cache.write(values, slots)
        with self.assertRaises(ValueError):
            cache.gather(slots, backend="other")
        with self.assertRaises(ValueError):
            cache.gather(slots, out=cache.raw[0, :576].view(2, 288), backend="eager")
        with self.assertRaises(ValueError):
            cache.gather(cache.raw[0, :16].view(torch.int64), backend="eager")
        shared = torch.zeros(576, dtype=torch.uint8)
        with self.assertRaises(ValueError):
            cache.gather(
                shared[:16].view(torch.int64), out=shared.view(2, 288), backend="eager"
            )
        for raw in (
            cache.raw.float(),
            cache.raw[:, :-1],
            cache.raw[:, ::2],
            cache.raw[:1].expand(2, -1),
        ):
            with self.assertRaises(ValueError):
                codec.V41PagedCache(raw, "hca_kv")
        with self.assertRaises(ValueError):
            codec.V41PagedCache(cache.raw, "csa_state")
        for bad in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaisesRegex(ValueError, "finite"):
                cache.write(torch.full_like(values, bad), slots, backend="eager")

    def test_rejects_invalid_logical_mappings(self):
        for indices, blocks, tag, ratio in (
            ([256], [0, 1], "indexer_kv", 2),
            ([-2], [0], "hca_kv", 1),
            ([0], [-1], "hca_kv", 1),
            ([0], [0], "hca_kv", 2),
            ([0], [0], "swa_kv", 1),
        ):
            with self.assertRaises(ValueError):
                codec.logical_to_physical_slots(
                    torch.tensor(blocks), torch.tensor(indices), tag, ratio
                )


if __name__ == "__main__":
    unittest.main()
