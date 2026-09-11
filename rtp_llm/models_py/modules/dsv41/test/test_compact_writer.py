"""Native compact bytes against the frozen official TileLang GPU quantizers."""

import hashlib
import importlib.util
import os
import sys
import unittest
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv41.cache_layout import ENCODINGS, CacheRegion
from rtp_llm.models_py.modules.dsv41.compact_reader import CompactPages, gather_compact
from rtp_llm.models_py.modules.dsv41.compact_writer import encode_compact, write_compact

OFFICIAL_KERNEL_SHA256 = (
    "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"
)


def _load_official_kernel():
    configured = os.environ.get("DSV41_OFFICIAL_KERNEL_PATH")
    if configured is None:
        model = os.environ.get("DSV41_MODEL_PATH")
        if model is None:
            raise RuntimeError("set DSV41_OFFICIAL_KERNEL_PATH or DSV41_MODEL_PATH")
        configured = str(Path(model) / "inference/kernel.py")
    path = Path(configured)
    if not path.is_file():
        raise RuntimeError(f"missing frozen official GPU quantizer: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != OFFICIAL_KERNEL_SHA256:
        raise RuntimeError(
            f"official quantizer hash mismatch: expected {OFFICIAL_KERNEL_SHA256}, got {actual}"
        )
    # Bazel separates TileLang and z3-solver into different runfiles repositories.
    from rtp_llm.models_py.modules.dsv4.tilelang_kernels import _ensure_libz3_loadable

    _ensure_libz3_loadable()
    spec = importlib.util.spec_from_file_location("_dsv41_frozen_quant_kernel", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # Importing a reference must not create a pycache beside the source snapshot.
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _official_gpu(module, values, region, inplace=False):
    rows, dimension = values.shape
    group = ENCODINGS[region].group_size
    if region == CacheRegion.SWA:
        output_dtype = torch.bfloat16 if inplace else torch.float8_e4m3fn
        output = torch.empty_like(values, dtype=output_dtype)
        scales = torch.empty(
            (rows, dimension // group), dtype=torch.float8_e8m0fnu, device=values.device
        )
        kernel = module.act_quant_kernel(
            dimension,
            group,
            in_dtype=module.BF16,
            out_dtype=module.FP8,
            scale_dtype=module.FE8M0,
            round_scale=True,
            inplace=inplace,
        )
    else:
        scale_dtype = (
            torch.float8_e4m3fn
            if region == CacheRegion.GLOBAL
            else torch.float8_e8m0fnu
        )
        output = (
            torch.empty_like(values)
            if inplace
            else torch.empty(
                (rows, dimension // 2),
                dtype=torch.float4_e2m1fn_x2,
                device=values.device,
            )
        )
        scales = torch.empty(
            (rows, dimension // group), dtype=scale_dtype, device=values.device
        )
        kernel = module.fp4_quant_kernel(
            dimension,
            group,
            in_dtype=module.BF16,
            scale_dtype=module.FP8 if region == CacheRegion.GLOBAL else module.FE8M0,
            inplace=inplace,
        )
    # Execute the original GPU prim_func. Only allocation is made explicit here.
    kernel(values, output, scales)
    if inplace:
        return output
    return torch.cat((output.view(torch.uint8), scales.view(torch.uint8)), dim=1)


def _ints(values):
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def _pages(region, rows, entries=128, marker=0x5A):
    byte_width = ENCODINGS[region].entry_bytes
    stride = ((entries * byte_width + 511) // 512) * 512
    count = 2 + (rows + entries - 1) // entries
    storage = torch.full((count, stride), marker, dtype=torch.uint8, device="cuda")
    return CompactPages(storage[:, : entries * byte_width], region, entries), storage


class CompactWriterGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            raise RuntimeError(
                "this required quantizer test needs an actual Blackwell GPU"
            )
        if (
            os.environ.get("DSV41_NATIVE_COMPACT_WRITER") != "1"
            or os.environ.get("DSV41_NATIVE_COMPACT_READER") != "1"
        ):
            raise RuntimeError(
                "enable both native compact writer and reader for the ABI test"
            )
        cls.official = _load_official_kernel()

    def test_planar_destination_is_rejected_before_any_byte_is_written(self):
        from rtp_llm.models_py.modules.dsv41.flashmla import PlanarPages

        for region in (CacheRegion.SWA, CacheRegion.GLOBAL):
            pages, storage = _pages(region, 3)
            before = storage.clone()
            planar = PlanarPages(pages.data, region, pages.entries_per_page)
            values = torch.ones(
                (3, ENCODINGS[region].head_dim), dtype=torch.bfloat16, device="cuda"
            )
            with self.assertRaisesRegex(TypeError, "row-interleaved"):
                write_compact(values, planar, _ints([128, 129, 130]))
            torch.testing.assert_close(storage, before, rtol=0, atol=0)

    def _assert_bytes(self, values, region):
        original = values.view(torch.int16).clone()
        expected = _official_gpu(self.official, values, region)
        actual = encode_compact(values, region)
        actual.check()
        torch.testing.assert_close(actual.output, expected, rtol=0, atol=0)
        torch.testing.assert_close(values.view(torch.int16), original, rtol=0, atol=0)
        return expected

    def _assert_roundtrip(self, rows, region):
        dimension = ENCODINGS[region].head_dim
        generator = torch.Generator().manual_seed(421 + rows + dimension)
        values = (
            torch.randn((rows, dimension), generator=generator)
            .mul_(3.5)
            .bfloat16()
            .cuda()
        )
        expected = self._assert_bytes(values, region)
        pages, storage = _pages(region, rows)
        entries = pages.entries_per_page
        slot_mapping = torch.arange(rows, dtype=torch.int64, device="cuda") + entries
        written = write_compact(values, pages, slot_mapping)
        written.check()
        expected_storage = torch.full_like(storage, 0x5A)
        byte_width = ENCODINGS[region].entry_bytes
        for page, start in enumerate(range(0, rows, entries), 1):
            count = min(entries, rows - start)
            expected_storage[page, : count * byte_width].copy_(
                expected[start : start + count].reshape(-1)
            )
        torch.testing.assert_close(storage, expected_storage, rtol=0, atol=0)
        table = _ints([list(range(1, 1 + (rows + entries - 1) // entries))])
        result = gather_compact(
            pages,
            table,
            _ints([0]),
            torch.arange(rows, dtype=torch.int32, device="cuda").reshape(1, rows),
            _ints([rows]),
        )
        result.check()
        fakequant = _official_gpu(self.official, values, region, inplace=True)
        torch.testing.assert_close(result.output[0], fakequant, rtol=0, atol=0)

    def test_zero_negative_zero_and_subnormal_rows(self):
        for region in CacheRegion:
            dimension = ENCODINGS[region].head_dim
            values = torch.zeros((5, dimension), dtype=torch.bfloat16, device="cuda")
            values[1].fill_(-0.0)
            values[2].fill_(2**-133)
            values[3].fill_(-(2**-133))
            if region == CacheRegion.INDEX_K:
                minimum_scale = 2**-126
                boundary = [6, -6, 0.25, -0.25, 0.75, -0.75, 1.25, -1.25]
                values[4] = (
                    torch.tensor(boundary * (dimension // len(boundary)), device="cuda")
                    .mul_(minimum_scale)
                    .bfloat16()
                )
            elif region == CacheRegion.GLOBAL:
                values[4].fill_(6 * 2**-9)
            else:
                values[4].fill_(1.0e-4)
            with self.subTest(region=region):
                self._assert_bytes(values, region)

    def test_fp4_ties_round_to_even_in_both_scale_formats(self):
        boundaries = [
            6,
            -6,
            0.25,
            -0.25,
            0.75,
            -0.75,
            1.25,
            -1.25,
            1.75,
            -1.75,
            2.5,
            -2.5,
            3.5,
            -3.5,
            5,
            -5,
        ]
        for region in (CacheRegion.GLOBAL, CacheRegion.INDEX_K):
            dimension = ENCODINGS[region].head_dim
            values = torch.tensor(
                boundaries * (dimension // 16), device="cuda"
            ).reshape(1, dimension)
            values = values.repeat(5, 1) * torch.tensor(
                [0.125, 0.5, 1, 2, 16], device="cuda"
            ).reshape(5, 1)
            self._assert_bytes(values.bfloat16(), region)

    def test_fp8_ties_and_subnormal_payloads(self):
        normalized = [
            448,
            -448,
            0.0,
            -0.0,
            1.0625,
            -1.0625,
            1.1875,
            -1.1875,
            2**-10,
            -(2**-10),
            3 * 2**-10,
            -3 * 2**-10,
            1.3125,
            -1.3125,
            1.4375,
            -1.4375,
        ]
        values = (
            torch.tensor(normalized * 32, device="cuda").reshape(1, 512).repeat(5, 1)
        )
        values *= torch.tensor([2**-22, 0.125, 1, 2, 128], device="cuda").reshape(5, 1)
        self._assert_bytes(values.bfloat16(), CacheRegion.SWA)

    def test_nonfinite_input_reports_error_and_preserves_full_destination_row(self):
        for region in CacheRegion:
            dimension = ENCODINGS[region].head_dim
            values = torch.ones((5, dimension), dtype=torch.bfloat16, device="cuda")
            values[1, 0] = torch.nan
            values[2, dimension // 2] = torch.inf
            values[3, -1] = -torch.inf
            standalone = encode_compact(values, region)
            self.assertEqual(standalone.status.cpu().tolist(), [0, 1, 1, 1, 0])
            self.assertTrue(
                torch.equal(
                    standalone.output[1:4], torch.zeros_like(standalone.output[1:4])
                )
            )
            with self.assertRaisesRegex(RuntimeError, "nonfinite input"):
                standalone.check()
            pages, storage = _pages(region, 5)
            result = write_compact(values, pages, _ints([128, 129, 130, 131, -1]))
            self.assertEqual(result.status.cpu().tolist(), [0, 1, 1, 1, 0])
            byte_width = ENCODINGS[region].entry_bytes
            self.assertTrue(
                torch.equal(
                    storage[1, byte_width:],
                    torch.full_like(storage[1, byte_width:], 0x5A),
                )
            )
            expected = _official_gpu(self.official, values[:1], region)
            torch.testing.assert_close(
                storage[1, :byte_width], expected[0], rtol=0, atol=0
            )

    def test_reserved_out_of_range_and_padding_slots(self):
        values = torch.ones((5, 512), dtype=torch.bfloat16, device="cuda")
        values[4, 0] = torch.nan
        pages, storage = _pages(CacheRegion.GLOBAL, 5)
        before = storage.clone()
        result = write_compact(
            values, pages, _ints([0, -2, storage.shape[0] * 128, 128, -1])
        )
        self.assertEqual(result.status.cpu().tolist(), [2, 2, 2, 0, 0])
        expected = _official_gpu(self.official, values[3:4], CacheRegion.GLOBAL)
        before[1, :288].copy_(expected[0])
        torch.testing.assert_close(storage, before, rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, "invalid destination"):
            result.check()

    def test_graph_reads_new_values_and_slot_mapping_without_recapture(self):
        for region in CacheRegion:
            dimension = ENCODINGS[region].head_dim
            values = torch.ones((6, dimension), dtype=torch.bfloat16, device="cuda")
            pages, storage = _pages(region, 6)
            slots = _ints(list(range(128, 134)))
            status = torch.empty(6, dtype=torch.int32, device="cuda")
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    write_compact(values, pages, slots, status=status)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                result = write_compact(values, pages, slots, status=status)
            updated = (
                torch.linspace(-7, 7, 6 * dimension, device="cuda")
                .reshape(6, dimension)
                .bfloat16()
            )
            values.copy_(updated)
            slots.copy_(_ints(list(range(256 + 17, 256 + 23))))
            before = storage.clone()
            graph.replay()
            result.check()
            expected = _official_gpu(self.official, updated, region)
            byte_width = ENCODINGS[region].entry_bytes
            before[2, 17 * byte_width : 23 * byte_width].copy_(expected.reshape(-1))
            torch.testing.assert_close(storage, before, rtol=0, atol=0)

    def test_encode_graph_updates_payload_scales_and_nonfinite_status(self):
        values = torch.ones((5, 512), dtype=torch.bfloat16, device="cuda")
        output = torch.empty((5, 288), dtype=torch.uint8, device="cuda")
        status = torch.empty(5, dtype=torch.int32, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                encode_compact(values, CacheRegion.GLOBAL, output=output, status=status)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = encode_compact(
                values, CacheRegion.GLOBAL, output=output, status=status
            )
        values.fill_(3.5)
        graph.replay()
        result.check()
        torch.testing.assert_close(
            output,
            _official_gpu(self.official, values, CacheRegion.GLOBAL),
            rtol=0,
            atol=0,
        )
        values[2, 70] = torch.nan
        graph.replay()
        self.assertEqual(status.cpu().tolist(), [0, 0, 1, 0, 0])
        with self.assertRaisesRegex(RuntimeError, "nonfinite input"):
            result.check()


def _matrix_case(rows, region):
    def run(self):
        self._assert_roundtrip(rows, region)

    return run


for _region in CacheRegion:
    for _rows in (1, 5, 6, 127, 129):
        setattr(
            CompactWriterGpuTest,
            f"test_official_gpu_bytes_and_reader_{_region.value}_m{_rows}",
            _matrix_case(_rows, _region),
        )


if __name__ == "__main__":
    unittest.main()
