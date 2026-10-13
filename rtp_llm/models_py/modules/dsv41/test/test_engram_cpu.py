"""CPU native conversion contract, plus explicitly enabled CUDA oracle."""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "engram_cpu_tested", SOURCE / "engram_cpu.py"
)
cpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpu)


def exhaustive_inputs():
    weight = torch.arange(256, dtype=torch.uint8).repeat(256, 1)
    scales = torch.arange(256, dtype=torch.uint8)[:, None].repeat(1, 8)
    return weight, scales, torch.arange(256).reshape(-1, 1)


def reference(weight, scales, ids):
    n, dim = weight.shape
    flat = ids.flatten()
    valid = (flat >= 0) & (flat < n)
    output = torch.zeros((flat.numel(), dim), dtype=torch.bfloat16)
    if n:
        rows = weight.view(torch.uint8).index_select(0, flat.clamp(0, n - 1))
        exponent = scales.index_select(0, flat.clamp(0, n - 1)).to(torch.int32)
        scale = (exponent << 23).view(torch.float32).repeat_interleave(32, 1)
        output = (rows.view(torch.float8_e4m3fn).float() * scale).bfloat16()
        output[~valid] = 0
    return output.reshape(ids.shape[0], ids.shape[1] * dim)


class EngramCpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cpu.prepare_cpu_lookup()

    def assertEquivalent(self, output, expected):
        self.assertTrue(torch.equal(output.isnan(), expected.isnan()))
        finite = ~expected.isnan()
        self.assertTrue(
            torch.equal(
                output.view(torch.int16)[finite], expected.view(torch.int16)[finite]
            )
        )

    def test_exhaustive_fp8_scale_special_values(self):
        weight, scales, ids = exhaustive_inputs()
        out = torch.empty((256, 256), dtype=torch.bfloat16)
        cpu.lookup_host_rows_cpu(weight, scales, ids, out)
        self.assertEquivalent(out, reference(weight, scales, ids))

    def test_strides_offsets_threads_and_global_thread_setting(self):
        generator = torch.Generator().manual_seed(20261012)
        initial_threads = torch.get_num_threads()
        for dim in (32, 256, 512):
            weight = torch.randint(
                256, (23, dim + 32), dtype=torch.uint8, generator=generator
            )[1:22, 3 : dim + 3]
            scales = torch.randint(
                256, (23, dim // 32 + 3), dtype=torch.uint8, generator=generator
            )[1:22, 1 : dim // 32 + 1]
            ids = torch.randint(-3, 24, (13, 7), generator=generator)
            expected = reference(weight, scales, ids)
            for threads in (1, 4, 8, 16):
                out = torch.empty_like(expected)
                cpu.lookup_host_rows_cpu(weight, scales, ids, out, num_threads=threads)
                self.assertEquivalent(out, expected)
        self.assertEqual(torch.get_num_threads(), initial_threads)

    def test_invalid_extreme_ids_and_empty_table(self):
        weight, scales, _ = exhaustive_inputs()
        ids = torch.tensor([[-(2**63), -1, 256, 2**63 - 1, 0, 255]])
        out = torch.empty((1, 6 * 256), dtype=torch.bfloat16)
        cpu.lookup_host_rows_cpu(weight, scales, ids, out)
        self.assertEquivalent(out, reference(weight, scales, ids))
        cpu.lookup_host_rows_cpu(weight[:0], scales[:0], ids, out)
        self.assertEqual(out.count_nonzero().item(), 0)

    def test_empty_output(self):
        weight, scales, _ = exhaustive_inputs()
        ids = torch.empty((0, 24), dtype=torch.int64)
        out = torch.empty((0, 24 * 256), dtype=torch.bfloat16)
        self.assertIs(cpu.lookup_host_rows_cpu(weight, scales, ids, out), out)

    def test_cold_and_unsupported_never_compile(self):
        weight, scales, ids = exhaustive_inputs()
        out = torch.empty((256, 256), dtype=torch.bfloat16)
        with patch.object(
            cpu, "_build_library", side_effect=AssertionError("hot compile")
        ):
            with patch.object(cpu, "_LIBRARY", None):
                with self.assertRaises(RuntimeError):
                    cpu.lookup_host_rows_cpu(weight, scales, ids, out)
            for bad_threads in (0, 17, True):
                self.assertFalse(
                    cpu.is_supported(weight, scales, ids, out, num_threads=bad_threads)
                )
            self.assertFalse(cpu.is_supported(weight, scales, ids.int(), out))
            self.assertFalse(
                cpu.is_supported(weight, scales, torch._neg_view(ids), out)
            )
            self.assertFalse(
                cpu.is_supported(weight, scales, ids, torch._neg_view(out))
            )
            self.assertFalse(cpu.is_supported(weight, scales, ids, out[:, ::2]))
            self.assertFalse(cpu.is_supported(weight[:, ::2], scales, ids, out))


@unittest.skipUnless(
    os.environ.get("RTP_ENGRAM_CPU_TEST_CUDA") == "1", "explicit CUDA oracle only"
)
class EngramCpuCudaOracle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cpu.prepare_cpu_lookup()
        spec = importlib.util.spec_from_file_location(
            "engram_gpu_oracle", SOURCE / "_engram_triton.py"
        )
        cls.gpu = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.gpu)

    def test_all_65536_conversions_bitwise(self):
        weight, scales, ids = exhaustive_inputs()
        out = torch.empty((256, 256), dtype=torch.bfloat16, pin_memory=True)
        cpu.lookup_host_rows_cpu(weight, scales, ids, out)
        actual = self.gpu.lookup_host_rows(
            weight.cuda(), scales.cuda(), ids.cuda(), 148
        )
        self.assertTrue(
            torch.equal(
                out.view(torch.uint8), actual.cpu().reshape_as(out).view(torch.uint8)
            )
        )

    def test_random_invalid_ids_and_nonblocking_h2d(self):
        generator = torch.Generator().manual_seed(4242)
        weight = torch.randint(256, (127, 256), dtype=torch.uint8, generator=generator)
        scales = torch.randint(256, (127, 8), dtype=torch.uint8, generator=generator)
        ids = torch.randint(-2, 130, (41, 24), generator=generator)
        out = torch.empty((41, 24 * 256), dtype=torch.bfloat16, pin_memory=True)
        cpu.lookup_host_rows_cpu(weight, scales, ids, out)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            copied = out.to("cuda", non_blocking=True)
        stream.synchronize()
        actual = self.gpu.lookup_host_rows(
            weight.cuda(), scales.cuda(), ids.cuda(), 148
        )
        self.assertTrue(
            torch.equal(
                copied.view(torch.uint8), actual.reshape_as(copied).view(torch.uint8)
            )
        )


if __name__ == "__main__":
    unittest.main()
