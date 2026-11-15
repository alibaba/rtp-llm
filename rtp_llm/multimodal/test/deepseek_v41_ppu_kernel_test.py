"""Stdlib contracts, plus explicit --device PPU vision kernel/replay tests.

The default suite needs neither Torch nor Triton. Metadata doubles exercise
admission and launch boundaries only; they never establish device numerics.
Set RTP_V41_TEST_VISION_PPU_DEVICE=1 or pass --device to require real PPU tests.
An explicitly requested device run fails when the device/dependencies are absent.
"""

import contextlib
import hashlib
import importlib.util
import json
import os
import re
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
MODULES = ROOT / "rtp_llm/models_py/modules/dsv41"
JIT = ROOT / "rtp_llm/models_py/kernels/ppu/jit"
BUNDLE = JIT / "dsv4_ppu/vision_activation"
DEVICE = (
    "--device" in sys.argv
    or os.environ.get("RTP_V41_TEST_VISION_PPU_DEVICE") == "1"
)
if "--device" in sys.argv:
    sys.argv.remove("--device")


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TensorMetadata:
    """Host metadata only: no tensor arithmetic or device execution."""

    next_pointer = 1 << 20

    def __init__(
        self,
        shape,
        dtype="bf16",
        device="cuda:0",
        strides=None,
        pointer=None,
        storage=None,
    ):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype, self.device = dtype, device
        self.is_cuda = device.startswith("cuda")
        self.layout = "strided"
        self.requires_grad = False
        self.negative = self.conjugate = False
        self.contiguous = True
        self._strides = strides
        if pointer is None:
            TensorMetadata.next_pointer += 1 << 20
            pointer = TensorMetadata.next_pointer
        self.pointer = pointer
        self.storage = pointer if storage is None else storage

    def stride(self, axis=None):
        strides = self._strides
        if strides is None:
            strides, running = [], 1
            for size in reversed(self.shape):
                strides.insert(0, running)
                running *= size
            strides = tuple(strides)
        return strides if axis is None else strides[axis]

    def is_contiguous(self):
        return self.contiguous

    def is_neg(self):
        return self.negative

    def is_conj(self):
        return self.conjugate

    def data_ptr(self):
        return self.pointer

    def element_size(self):
        return 2 if self.dtype == "bf16" else 4

    def untyped_storage(self):
        return types.SimpleNamespace(data_ptr=lambda: self.storage)

    def as_strided(self, shape, strides):
        return TensorMetadata(
            shape, self.dtype, self.device, strides, self.pointer, self.storage
        )

    def __getitem__(self, key):
        # Only the fixed fresh Q/K output views are needed by the dispatch test.
        heads = 16
        pointer = self.pointer + (heads * 64 * 2 if key[1].start else 0)
        return TensorMetadata(
            (self.shape[0], heads, 64),
            self.dtype,
            self.device,
            (2048, 64, 1),
            pointer,
            self.storage,
        )


class LaunchBoundary:
    def __init__(self):
        self.call = Mock()
        self.grid = None

    def __getitem__(self, grid):
        self.grid = grid
        return self.call


class VisionMetadataTest(unittest.TestCase):
    def setUp(self):
        torch = types.ModuleType("torch")
        torch.Tensor = TensorMetadata
        torch.strided, torch.bfloat16, torch.float32 = "strided", "bf16", "fp32"
        torch.version = types.SimpleNamespace(hip=None)
        torch.is_grad_enabled = Mock(return_value=False)
        torch.cuda = types.SimpleNamespace(
            get_device_capability=Mock(return_value=(8, 9)),
            is_current_stream_capturing=Mock(return_value=False),
            device=lambda _: contextlib.nullcontext(),
        )
        torch.empty = Mock(
            side_effect=lambda shape, dtype, device: TensorMetadata(shape, dtype, device)
        )
        torch.empty_like = Mock(
            side_effect=lambda value: TensorMetadata(
                value.shape, value.dtype, value.device
            )
        )
        triton = types.ModuleType("triton")
        triton.jit = lambda *args, **kwargs: (lambda fn: fn)
        triton.cdiv = (
            lambda numerator, denominator: (numerator + denominator - 1) // denominator
        )
        triton.next_power_of_2 = lambda value: 1 << (value - 1).bit_length()
        language = types.ModuleType("triton.language")
        language.constexpr = object
        triton.language = language
        quant = types.ModuleType("rtp_llm.models_py.modules.dsv41.ppu._quant")
        quant.is_ppu = Mock(return_value=True)
        loader = types.ModuleType("rtp_llm.models_py.kernels.ppu.jit.sglang_jit")
        loader.load_sglang_kernel = Mock(
            return_value=types.SimpleNamespace(forward=Mock())
        )
        module_patch = patch.dict(
            sys.modules,
            {
                "torch": torch,
                "triton": triton,
                "triton.language": language,
                quant.__name__: quant,
                loader.__name__: loader,
            },
        )
        module_patch.start()
        self.addCleanup(module_patch.stop)
        self.torch, self.quant, self.loader = torch, quant, loader
        self.rope = load_file(
            "vision_rope_metadata", MODULES / "_vision_rope_triton.py"
        )
        self.norm = load_file(
            "vision_norm_metadata", MODULES / "vision_residual_add.py"
        )
        self.downsample = load_file(
            "vision_downsample_metadata", MODULES / "vision_downsample_triton.py"
        )
        self.activation = load_file(
            "vision_activation_metadata", MODULES / "vision_ppu.py"
        )

    @staticmethod
    def rope_inputs(rows=7):
        q = TensorMetadata(
            (rows, 16, 64), strides=(3072, 64, 1), pointer=16, storage=16
        )
        k = TensorMetadata(q.shape, strides=q.stride(), pointer=2064, storage=16)
        return (
            q,
            k,
            TensorMetadata((rows, 1, 32), "fp32"),
            TensorMetadata((rows, 1, 32), "fp32"),
        )

    def test_cuda_eligibility_stays_and_ppu_requires_positive_identification(self):
        for module in (self.rope, self.norm, self.downsample):
            with self.subTest(module=module.__name__):
                self.torch.cuda.get_device_capability.return_value = (10, 3)
                self.quant.is_ppu.reset_mock()
                self.assertTrue(module._supported_device("cuda:0"))
                self.quant.is_ppu.assert_not_called()
                self.torch.cuda.get_device_capability.return_value = (8, 9)
                self.quant.is_ppu.return_value = True
                self.assertTrue(module._supported_device("cuda:0"))
                self.quant.is_ppu.return_value = False
                self.assertFalse(module._supported_device("cuda:0"))

    def test_rope_rejects_wrong_storage_pitch_table_or_geometry_before_launch(self):
        self.assertTrue(self.rope._enabled_or_supported(*self.rope_inputs()))
        for attribute, value in (
            ("storage", 32),
            ("pointer", 2080),
            ("_strides", (2048, 64, 1)),
        ):
            args = self.rope_inputs()
            setattr(args[1], attribute, value)
            self.assertIsNone(self.rope.apply_vision_qk_rope(*args))
        args = self.rope_inputs()
        args[2].dtype = "bf16"
        self.assertIsNone(self.rope.apply_vision_qk_rope(*args))
        args = self.rope_inputs()
        args[0].device, args[0].is_cuda = "cpu", False
        self.assertIsNone(self.rope.apply_vision_qk_rope(*args))
        self.torch.empty.assert_not_called()

    def test_rope_preserves_launch_rounding_and_out_of_place_layout(self):
        launch = LaunchBoundary()
        self.rope._vision_qk_rotary_kernel = launch
        q, k = self.rope.apply_vision_qk_rope(*self.rope_inputs())
        self.assertEqual((q.shape, k.shape), ((7, 16, 64),) * 2)
        self.assertEqual((q.stride(), k.stride()), ((2048, 64, 1),) * 2)
        self.assertNotEqual(q.storage, 16)
        self.assertEqual(launch.call.call_args.kwargs, {"enable_fp_fusion": False})
        self.assertEqual(launch.grid, (16, 1))

    def test_residual_metadata_rejects_before_allocation_and_errors_propagate(self):
        x, residual = TensorMetadata((7, 1024)), TensorMetadata((7, 1024))
        self.assertIsNone(
            self.norm.apply_vision_residual_rmsnorm(
                x, residual, TensorMetadata((1024,), "fp32"), 1e-6
            )
        )
        self.torch.empty_like.assert_not_called()
        launch = LaunchBoundary()
        self.norm._vision_residual_rmsnorm_kernel = launch
        summed, normalized = self.norm.apply_vision_residual_rmsnorm(
            x, residual, TensorMetadata((1024,)), 1e-6
        )
        self.assertEqual(summed.shape, x.shape)
        self.assertNotEqual(summed.pointer, normalized.pointer)
        self.assertEqual(launch.call.call_args.kwargs["enable_fp_fusion"], False)
        launch.call.side_effect = RuntimeError("launch failed")
        with self.assertRaisesRegex(RuntimeError, "launch failed"):
            self.norm.apply_vision_residual_rmsnorm(
                x, residual, TensorMetadata((1024,)), 1e-6
            )

    def test_downsample_admits_only_homogeneous_matching_packed_rows(self):
        value = TensorMetadata((154, 1024))
        self.assertTrue(self.downsample.is_supported(value, [(7, 11)] * 2, 3))
        for grids, ratio in (
            ([(7, 11), (11, 7)], 3),
            ([(7, 11)], 3),
            ([(7, 11)] * 2, 2),
            ([(0, 11)] * 2, 3),
        ):
            self.assertIsNone(
                self.downsample.apply_vision_downsample(value, grids, ratio)
            )
        self.torch.empty.assert_not_called()

    def test_activation_unsupported_metadata_allocates_and_compiles_nothing(self):
        cases = [
            TensorMetadata((7, 5632), device="cpu"),
            TensorMetadata((7, 5632), "fp32"),
            TensorMetadata((7, 2816)),
            TensorMetadata((1, 7, 5632)),
            TensorMetadata((self.activation._MAX_ROWS + 1, 5632)),
            TensorMetadata((7, 5632), pointer=18),
        ]
        for attribute in ("requires_grad", "negative", "conjugate"):
            value = TensorMetadata((7, 5632))
            setattr(value, attribute, True)
            cases.append(value)
        value = TensorMetadata((7, 5632))
        value.contiguous = False
        cases.append(value)
        for value in cases:
            self.assertIsNone(self.activation.apply_vision_silu_mul(value))
        self.quant.is_ppu.return_value = False
        self.assertIsNone(
            self.activation.apply_vision_silu_mul(TensorMetadata((7, 5632)))
        )
        self.torch.empty.assert_not_called()
        self.loader.load_sglang_kernel.assert_not_called()

    def test_activation_empty_is_fresh_and_does_not_compile(self):
        value = TensorMetadata((0, 5632))
        output = self.activation.apply_vision_silu_mul(value)
        self.assertEqual(output.shape, (0, 2816))
        self.assertNotEqual(output.pointer, value.pointer)
        self.loader.load_sglang_kernel.assert_not_called()

    def test_activation_entry_is_rounded_unclipped_non_pdl_and_cached_for_graph(self):
        value = TensorMetadata((7, 5632))
        first = self.activation.apply_vision_silu_mul(value)
        self.loader.load_sglang_kernel.assert_called_once_with(
            "vision_silu_mul_rounded",
            "csrc/elementwise/activation.cuh",
            "sglang::ActivationKernel<bf16_t,false>::run_activation_with_rounding",
            (8, 9),
            ("--fmad=false",),
            bundle="vision_activation",
        )
        native = self.loader.load_sglang_kernel.return_value
        native.forward.assert_called_once_with(value, first, "silu", 0.0)
        self.torch.cuda.is_current_stream_capturing.return_value = True
        second = self.activation.apply_vision_silu_mul(value)
        self.assertNotEqual(first.pointer, second.pointer)
        self.assertEqual(self.loader.load_sglang_kernel.call_count, 1)
        native.forward.side_effect = RuntimeError("native failure")
        with self.assertRaisesRegex(RuntimeError, "native failure"):
            self.activation.apply_vision_silu_mul(value)

    def test_cold_graph_capture_fails_before_jit_or_output_allocation(self):
        self.torch.cuda.is_current_stream_capturing.return_value = True
        with self.assertRaisesRegex(RuntimeError, "Warm PPU vision SwiGLU eagerly"):
            self.activation.apply_vision_silu_mul(TensorMetadata((7, 5632)))
        self.torch.empty.assert_not_called()
        self.loader.load_sglang_kernel.assert_not_called()


class FrozenVisionSourceTest(unittest.TestCase):
    def test_existing_jit_loader_accepts_exact_frozen_bundle_and_include_closure(self):
        loader = load_file("vision_source_loader", JIT / "sglang_jit.py")
        root, digest = loader._source_bundle("vision_activation")
        self.assertEqual(root, BUNDLE)
        self.assertEqual(len(digest), 16)
        manifest = json.loads((BUNDLE / "source-manifest.json").read_text())
        self.assertEqual(
            manifest["upstream"]["commit"], "3bc4f87d5f9bb6f04b724540fb12a3acdc62d2fa"
        )
        self.assertEqual(manifest["license"]["name"], "Apache-2.0")
        self.assertIn("Apache License", (BUNDLE / "LICENSE").read_text())
        for name, expected in manifest["sources"].items():
            data = (BUNDLE / name).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), expected, name)
            if name.endswith((".h", ".cuh")):
                for include in re.findall(
                    r"#include <(sgl_kernel/[^>]+)>", data.decode()
                ):
                    self.assertTrue(
                        any(
                            (BUNDLE / base / include).exists()
                            for base in ("compat", "include")
                        ),
                        include,
                    )
        entry = manifest["entrypoint"]
        self.assertFalse(entry["pdl"])
        self.assertFalse(entry["fast_math"])
        self.assertFalse(entry["input_mutated"])
        self.assertEqual(entry["clamp_limit"], 0.0)

    def test_compatibility_headers_are_identical_to_recorded_rtp_sources(self):
        manifest = json.loads((BUNDLE / "source-manifest.json").read_text())
        support = manifest["support_headers"]
        source_bundle = BUNDLE / support["source_bundle"]
        for name, origin in support["files"].items():
            with self.subTest(name=name):
                data = (BUNDLE / name).read_bytes()
                self.assertEqual(
                    data, (source_bundle / origin["source_path"]).read_bytes()
                )
                self.assertEqual(hashlib.sha256(data).hexdigest(), origin["sha256"])

    def test_upstream_stream_checks_are_provided_by_the_compat_include(self):
        manifest = json.loads((BUNDLE / "source-manifest.json").read_text())
        shim_path = "compat/sgl_kernel/utils.h"
        shim = (BUNDLE / shim_path).read_text()
        marker = "/** \\brief Error class for stream-style error logging. */"
        block = shim[
            shim.index(marker) : shim.index("    host::Error()")
            + len("    host::Error()\n")
        ]
        self.assertEqual(
            hashlib.sha256(block.encode()).hexdigest(),
            manifest["originals"][shim_path]["upstream_block_sha256"],
        )
        self.assertIn('#include "../../include/sgl_kernel/utils.h"', shim)
        self.assertIn("#ifndef CHECK_HOST", shim)
        self.assertIn("namespace host {", shim)
        self.assertIn("throw PanicError", block)
        self.assertIn("CHECK_HOST(", (BUNDLE / "csrc/elementwise/activation.cuh").read_text())


@unittest.skipUnless(
    DEVICE, "explicit --device or RTP_V41_TEST_VISION_PPU_DEVICE=1 required"
)
class VisionPPUDeviceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        if not torch.cuda.is_available() or "ZW" not in torch.cuda.get_device_name():
            raise RuntimeError("explicit vision device tests require a real PPU")
        cls.torch = torch
        cls.rope = load_file("vision_rope_device", MODULES / "_vision_rope_triton.py")
        cls.norm = load_file("vision_norm_device", MODULES / "vision_residual_add.py")
        cls.downsample = load_file(
            "vision_downsample_device", MODULES / "vision_downsample_triton.py"
        )
        cls.activation = load_file(
            "vision_activation_device", MODULES / "vision_ppu.py"
        )

    def setUp(self):
        self.guard = self.torch.inference_mode()
        self.guard.__enter__()
        self.addCleanup(self.guard.__exit__, None, None, None)
        self.torch.manual_seed(20261114)

    def assert_bits(self, actual, expected):
        torch = self.torch
        self.assertEqual((actual.shape, actual.dtype), (expected.shape, expected.dtype))
        self.assertTrue(
            torch.equal(
                actual.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            )
        )

    def rotary_inputs(self, rows):
        torch = self.torch
        qkv = torch.randn((rows, 3072), dtype=torch.bfloat16, device="cuda")
        phase = torch.randn((rows, 1, 32), device="cuda")
        return qkv, phase.cos(), phase.sin()

    @staticmethod
    def qk(qkv):
        return tuple(
            value.view(qkv.shape[0], 16, 64) for value in qkv.chunk(3, dim=-1)
        )[:2]

    def test_split_half_rope_is_bit_exact_and_keeps_qkv_read_only(self):
        torch = self.torch
        for rows in (1, 7, 33, 4097):
            qkv, cos, sin = self.rotary_inputs(rows)
            qkv[0, :4] = torch.tensor([0.0, -0.0, 1.0, -1.0], device="cuda")
            saved = qkv.clone()
            q, k = self.qk(qkv)
            actual = self.rope.apply_vision_qk_rope(q, k, cos, sin)
            self.assertIsNotNone(actual)
            for got, value in zip(actual, (q, k)):
                left, right = value.float().chunk(2, dim=-1)
                expected = torch.cat(
                    (left * cos - right * sin, left * sin + right * cos), dim=-1
                ).to(value.dtype)
                self.assert_bits(got, expected)
                self.assertNotEqual(
                    got.untyped_storage().data_ptr(), qkv.untyped_storage().data_ptr()
                )
            self.assert_bits(qkv, saved)

    def test_residual_rounds_bf16_sum_before_rmsnorm_and_preserves_inputs(self):
        torch = self.torch
        for rows in (1, 7, 4097):
            x = torch.randn((rows, 1024), dtype=torch.bfloat16, device="cuda")
            residual = torch.randn_like(x)
            weight = torch.randn((1024,), dtype=x.dtype, device=x.device)
            saved_x, saved_residual = x.clone(), residual.clone()
            summed, output = self.norm.apply_vision_residual_rmsnorm(
                x, residual, weight, 1e-6
            )
            expected_sum = x + residual
            value = expected_sum.float()
            expected = (
                value
                * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
                * weight.float()
            ).to(x.dtype)
            self.assert_bits(summed, expected_sum)
            relative_l2 = torch.linalg.vector_norm(
                output.float() - expected.float()
            ) / torch.linalg.vector_norm(expected.float())
            self.assertLessEqual(relative_l2.item(), 2e-3)
            self.assertLessEqual(
                (output.float() - expected.float()).abs().max().item(), 0.03125
            )
            self.assertEqual(
                len(
                    {
                        x.data_ptr(),
                        residual.data_ptr(),
                        summed.data_ptr(),
                        output.data_ptr(),
                    }
                ),
                4,
            )
            self.assert_bits(x, saved_x)
            self.assert_bits(residual, saved_residual)

    def test_downsample_preserves_columns_padding_bits_and_consumer_strides(self):
        torch = self.torch
        import torch.nn.functional as functional

        for batch, height, width in ((2, 1, 1), (3, 7, 11), (2, 31, 43)):
            x = (
                torch.randint(
                    0,
                    65536,
                    (batch * height * width, 1024),
                    device="cuda",
                    dtype=torch.int32,
                )
                .to(torch.int16)
                .view(torch.bfloat16)
            )
            saved = x.clone()
            columns = self.downsample.apply_vision_downsample(
                x, [(height, width)] * batch, 3
            )
            self.assertIsNotNone(columns)
            for got, part in zip(columns, x.split(height * width)):
                image = part.view(height, width, -1).permute(2, 0, 1)
                image = functional.pad(image, (0, -width % 3, 0, -height % 3))
                expected = functional.unfold(image.unsqueeze(0), 3, stride=3).squeeze(0)
                self.assert_bits(got, expected)
                self.assertEqual(got.stride(), expected.stride())
                self.assertEqual(got.t().stride(), expected.t().stride())
                self.assertNotEqual(
                    got.untyped_storage().data_ptr(), x.untyped_storage().data_ptr()
                )
            self.assert_bits(x, saved)

    def test_swiglu_preserves_activation_rounding_unclipped_values_and_input(self):
        torch = self.torch
        import torch.nn.functional as functional

        for rows in (1, 7, 33, 4097):
            value = torch.randn((rows, 5632), dtype=torch.bfloat16, device="cuda") * 3
            value[:, 0], value[:, 2816] = 32.0, 17.0
            saved = value.clone()
            output = self.activation.apply_vision_silu_mul(value)
            self.assertIsNotNone(output)
            gate, up = value.chunk(2, dim=-1)
            expected = functional.silu(gate) * up
            torch.testing.assert_close(output, expected, rtol=8e-3, atol=1e-5)
            self.assertTrue(torch.all(output[:, 0] == 544))
            self.assertNotEqual(
                output.untyped_storage().data_ptr(), value.untyped_storage().data_ptr()
            )
            self.assert_bits(value, saved)

    def test_graph_replay_uses_changed_live_inputs_for_all_four_kernels(self):
        torch = self.torch
        rows = 154
        qkv, cos, sin = self.rotary_inputs(rows)
        x = torch.randn((rows, 1024), dtype=torch.bfloat16, device="cuda")
        residual = torch.randn_like(x)
        weight = torch.randn((1024,), dtype=x.dtype, device=x.device)
        gateup = torch.randn((rows, 5632), dtype=x.dtype, device=x.device)

        def forward():
            q, k = self.rope.apply_vision_qk_rope(*self.qk(qkv), cos, sin)
            summed, normalized = self.norm.apply_vision_residual_rmsnorm(
                x, residual, weight, 1e-6
            )
            columns = self.downsample.apply_vision_downsample(
                normalized, [(7, 11)] * 2, 3
            )
            activation = self.activation.apply_vision_silu_mul(gateup)
            return (q, k, summed, normalized, activation, *columns)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            forward()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = forward()
        pointers = [value.data_ptr() for value in actual]
        for _ in range(3):
            for value in (qkv, cos, sin, x, residual, weight, gateup):
                value.normal_()
            saved = qkv.clone()
            expected = forward()
            graph.replay()
            self.assertEqual([value.data_ptr() for value in actual], pointers)
            for got, want in zip(actual, expected):
                self.assert_bits(got, want)
            self.assert_bits(qkv, saved)


if __name__ == "__main__":
    unittest.main()
