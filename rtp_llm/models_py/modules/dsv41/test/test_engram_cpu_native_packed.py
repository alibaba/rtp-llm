"""CPU-only byte transport and exact parallel hashing contracts."""

import ctypes
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "engram_cpu_packed_tested", SOURCE / "engram_cpu.py"
)
cpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpu)


def numpy_hash(windows, token_map, multipliers, primes, offsets, pad_id, image_token):
    """The original vectorized hash, including signed int64 wrap/remainder."""
    values = windows.numpy().astype(np.int64, copy=False)
    mapping = token_map.numpy()
    blocked = np.logical_or.accumulate(
        (values < 0) | (values >= mapping.size) | (values == image_token), axis=1
    )
    mapped = np.where(blocked, pad_id, mapping[np.clip(values, 0, mapping.size - 1)])
    layers, _, heads = primes.shape
    hashes = np.empty((values.shape[0], layers, offsets.shape[1]), dtype=np.int64)
    rolling = np.zeros((values.shape[0], layers), dtype=np.int64)
    for shift in range(values.shape[1]):
        rolling ^= mapped[:, shift, None] * multipliers.numpy()[None, :, shift]
        if shift:
            start, end = (shift - 1) * heads, shift * heads
            hashes[:, :, start:end] = (
                rolling[:, :, None] % primes.numpy()[None, :, shift - 1]
                + offsets.numpy()[None, :, start:end]
            )
    return torch.from_numpy(hashes)


def hash_constants(ngram=4, layers=2, heads=8):
    rng = np.random.default_rng(20261012 + ngram + layers + heads)
    mapping = torch.from_numpy(rng.integers(-(2**63), 2**63 - 1, 257))
    multipliers = torch.from_numpy(rng.integers(-(2**63), 2**63 - 1, (layers, ngram)))
    primes = torch.from_numpy(rng.integers(1, 2**63 - 1, (layers, ngram - 1, heads)))
    offsets = torch.from_numpy(
        rng.integers(-(2**63), 2**63 - 1, (layers, (ngram - 1) * heads))
    )
    return mapping, multipliers, primes, offsets


class EngramCpuPackedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cpu.prepare_cpu_lookup()

    def reference(self, weight, scales, ids):
        dim = weight.shape[1]
        expected = torch.zeros((*ids.shape, dim + dim // 32), dtype=torch.uint8)
        valid = (ids >= 0) & (ids < weight.shape[0])
        expected[valid] = torch.cat(
            (weight.view(torch.uint8)[ids[valid]], scales[ids[valid]]), dim=-1
        )
        return expected.flatten(1)

    def test_all_fp8_and_scale_bytes_are_preserved(self):
        # All encodings, including signed zero, FP8 NaNs and UE8M0 0/255.
        weight = torch.arange(256, dtype=torch.uint8).repeat(256, 1)
        scales = torch.arange(256, dtype=torch.uint8)[:, None].repeat(1, 8)
        ids = torch.arange(256).reshape(32, 8)
        out = torch.empty((32, 8 * 264), dtype=torch.uint8)
        for dtype in (torch.uint8, torch.float8_e4m3fn):
            self.assertIs(
                cpu.lookup_host_rows_cpu_packed(weight.view(dtype), scales, ids, out),
                out,
            )
            self.assertTrue(torch.equal(out, self.reference(weight, scales, ids)))

    def test_strides_offsets_invalid_ids_and_explicit_threads(self):
        rng = torch.Generator().manual_seed(20261012)
        initial_threads = torch.get_num_threads()
        for dim in (32, 256, 512, 4096):
            weight = torch.randint(
                256, (23, dim + 32), dtype=torch.uint8, generator=rng
            )[1:22, 3 : dim + 3]
            scales = torch.randint(
                256, (23, dim // 32 + 3), dtype=torch.uint8, generator=rng
            )[1:22, 1 : dim // 32 + 1]
            ids = torch.randint(-3, 24, (13, 7), generator=rng)
            ids[0, :4] = torch.tensor([-(2**63), -1, 21, 2**63 - 1])
            expected = self.reference(weight, scales, ids)
            for threads in (1, 4, 8, 16):
                # Nonzero output offset and sentinels detect over/underwrites.
                backing = torch.full((expected.numel() + 34,), 77, dtype=torch.uint8)
                out = backing[17:-17].view_as(expected)
                cpu.lookup_host_rows_cpu_packed(
                    weight, scales, ids, out, num_threads=threads
                )
                self.assertTrue(torch.equal(out, expected))
                self.assertTrue(torch.all(backing[:17] == 77))
                self.assertTrue(torch.all(backing[-17:] == 77))
        self.assertEqual(torch.get_num_threads(), initial_threads)

    def test_empty_table_and_output(self):
        weight = torch.empty((0, 256), dtype=torch.uint8)
        scales = torch.empty((0, 8), dtype=torch.uint8)
        ids = torch.tensor([[0, -1, 2**63 - 1]])
        out = torch.full((1, 3 * 264), 255, dtype=torch.uint8)
        cpu.lookup_host_rows_cpu_packed(weight, scales, ids, out)
        self.assertFalse(out.any())
        for shape in ((0, 24), (3, 0)):
            ids = torch.empty(shape, dtype=torch.int64)
            out = torch.empty((shape[0], shape[1] * 264), dtype=torch.uint8)
            self.assertIs(
                cpu.lookup_host_rows_cpu_packed(weight, scales, ids, out), out
            )

    def test_cold_invalid_and_aliased_inputs_never_compile(self):
        weight = torch.zeros((4, 256), dtype=torch.uint8)
        scales = torch.zeros((4, 8), dtype=torch.uint8)
        ids = torch.tensor([[0]])
        out = torch.empty((1, 264), dtype=torch.uint8)
        with patch.object(cpu, "_build_library", side_effect=AssertionError("compile")):
            with patch.object(cpu, "_LIBRARY", None):
                with self.assertRaises(RuntimeError):
                    cpu.lookup_host_rows_cpu_packed(weight, scales, ids, out)
            for threads in (0, 17, True):
                self.assertFalse(
                    cpu.is_packed_supported(
                        weight, scales, ids, out, num_threads=threads
                    )
                )
            for args in (
                (weight, scales, ids.int(), out),
                (weight, scales, torch._neg_view(ids), out),
                (weight, scales, ids, out.bfloat16()),
                (weight, scales, ids, out[:, ::2]),
                (weight, scales, ids, weight.flatten()[:264].view(1, 264)),
                (weight, scales, ids, out.to("meta")),
            ):
                self.assertFalse(cpu.is_packed_supported(*args))
                with self.assertRaises(ValueError):
                    cpu.lookup_host_rows_cpu_packed(*args)

    def test_native_prepare_semantics_and_invalid_geometry(self):
        self.assertIn(cpu._LIBRARY.engram_cpu_prepare(), (1, 2))
        null = ctypes.c_void_p()
        for values in (
            (0, 0, 0, 0, 0, 16),
            (1, 256, 256, 8, 1, 16),  # Nonempty inputs cannot be null.
            (0, 256, 255, 8, 0, 16),
            (0, 256, 256, 8, 2**63 - 1, 16),
            (0, 256, 256, 8, 0, 17),
        ):
            self.assertEqual(
                cpu._LIBRARY.engram_cpu_lookup_packed(null, null, null, null, *values),
                1,
            )


class EngramCpuHashTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cpu.prepare_cpu_lookup()

    def check(self, windows, constants, *, pad_id=-7, image_token=71, threads=16):
        actual = cpu.hash_token_windows_cpu(
            windows,
            *constants,
            pad_id=pad_id,
            image_token=image_token,
            num_threads=threads,
        )
        expected = numpy_hash(windows, *constants, pad_id, image_token)
        self.assertTrue(torch.equal(actual, expected))
        return actual

    def test_random_int64_overflow_and_prime_boundaries(self):
        rng = np.random.default_rng(20261012)
        initial_threads = torch.get_num_threads()
        for ngram, layers, heads in ((2, 1, 1), (4, 2, 8), (9, 3, 5)):
            constants = hash_constants(ngram, layers, heads)
            primes = constants[2]
            primes.flatten()[: min(4, primes.numel())] = torch.tensor(
                [1, 2, 2**62 + 1, 2**63 - 1][: min(4, primes.numel())]
            )
            windows = torch.from_numpy(rng.integers(-3, 270, (117, ngram)))
            windows[0, 0] = -(2**63)
            windows[1, 0] = 2**63 - 1
            for threads in (1, 4, 8, 16):
                self.check(windows, constants, pad_id=-(2**63), threads=threads)
        self.assertEqual(torch.get_num_threads(), initial_threads)

    def test_strided_int32_int64_and_broadcast_windows(self):
        constants = hash_constants()
        for dtype in (torch.int32, torch.int64):
            base = torch.arange(19 * 13, dtype=dtype).view(19, 13)
            self.check(base[1:18:2, 2:10:2], constants)
            self.check(base[:4, :7].t(), constants)
            self.check(base[2:3, :4].expand(25, -1), constants)
            self.check(base[:9, :1].expand(-1, 4), constants)

    def test_image_and_invalid_token_terminate_all_longer_ngrams(self):
        constants = hash_constants()
        normal = torch.tensor([[2, 3, 4, 5]], dtype=torch.int64)
        for barrier in (-1, -(2**63), 257, 2**63 - 1, 71):
            for position in range(4):
                windows = normal.repeat(2, 1)
                windows[:, position] = barrier
                windows[1, position + 1 :] = 200
                actual = self.check(windows, constants)
                self.assertTrue(torch.equal(actual[0], actual[1]))

    def test_signed_minimum_modulo_and_offset_wrap(self):
        windows = torch.zeros((3, 2), dtype=torch.int64)
        constants = (
            torch.tensor([1]),
            torch.tensor([[0, -(2**63)]]),
            torch.tensor([[[1, 3, 2**63 - 1]]]),
            torch.tensor([[0, 2**63 - 1, -(2**63)]]),
        )
        actual = self.check(windows, constants)
        self.assertEqual(actual[0, 0, 0].item(), 0)
        self.assertEqual(actual[0, 0, 1].item(), -(2**63))
        self.assertEqual(actual[0, 0, 2].item(), -2)

    def test_empty_output_and_no_retained_result(self):
        constants = hash_constants()
        windows = torch.empty((0, 4), dtype=torch.int32)
        self.assertEqual(tuple(self.check(windows, constants).shape), (0, 2, 24))
        windows = torch.zeros((3, 4), dtype=torch.int32)
        original = self.check(windows, constants)
        copied = original.clone()
        windows.fill_(42)
        updated = self.check(windows, constants)
        self.assertTrue(torch.equal(original, copied))
        self.assertNotEqual(original.data_ptr(), updated.data_ptr())

    def test_bad_constants_and_cold_call(self):
        constants = hash_constants()
        windows = torch.zeros((3, 4), dtype=torch.int64)
        with patch.object(cpu, "_build_library", side_effect=AssertionError("compile")):
            with patch.object(cpu, "_LIBRARY", None):
                with self.assertRaises(RuntimeError):
                    self.check(windows, constants)
            for bad in (windows.float(), windows.to("meta"), torch._neg_view(windows)):
                with self.assertRaises(ValueError):
                    self.check(bad, constants)
            for i, bad in (
                (0, constants[0][:0]),
                (0, constants[0].int()),
                (1, constants[1][:, ::2]),
                (2, constants[2][:, :, :0]),
                (2, torch.zeros_like(constants[2])),
                (2, -constants[2]),
                (3, constants[3][:, :-1]),
                (3, constants[3].to("meta")),
            ):
                edited = list(constants)
                edited[i] = bad
                with self.assertRaises(ValueError):
                    self.check(windows, edited)
            for kwargs in (
                {"threads": 0},
                {"threads": True},
                {"pad_id": 2**63},
                {"image_token": -(2**63) - 1},
            ):
                with self.assertRaises(ValueError):
                    self.check(windows, constants, **kwargs)


if __name__ == "__main__":
    unittest.main()
