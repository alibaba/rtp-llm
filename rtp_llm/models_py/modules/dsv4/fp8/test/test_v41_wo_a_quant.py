"""CPU dispatch contracts and actual SM100 GEMM epilogue regressions."""

import ast
import ctypes
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

_ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


op = _load("v41_wo_a_contract", _ROOT / "_v41_wo_a_quant.py")


def _metadata(m=33):
    device = torch.device("cuda:0")

    def tensor(shape, stride, dtype):
        return SimpleNamespace(
            shape=shape,
            ndim=len(shape),
            device=device,
            dtype=dtype,
            is_cuda=True,
            requires_grad=False,
            stride=lambda: stride,
            is_contiguous=lambda: True,
            data_ptr=lambda: 256,
        )

    aligned = (m + 3) // 4 * 4
    return (
        tensor((m, 8, 4096), (4096, m * 4096, 1), torch.float8_e4m3fn),
        tensor((m, 8, 32), (1, aligned * 32, aligned), torch.int32),
    ), (
        tensor((8, 1024, 4096), (1024 * 4096, 4096, 1), torch.float8_e4m3fn),
        tensor((8, 1024, 32), (32768, 1, 1024), torch.int32),
    )


class WoAQuantCPUContractTest(unittest.TestCase):
    def test_disk_cache_reuses_host_architecture_but_separates_arm_and_x86(self):
        def compile_fixture(command, **kwargs):
            Path(command[command.index("-o") + 1]).write_bytes(b"compiled fixture")
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            include = root / "include"
            include.mkdir()
            dependencies = (include, root, "nvcc", "c++")
            with mock.patch.dict(
                os.environ, {"DG_JIT_CACHE_DIR": directory}
            ), mock.patch.object(
                op.subprocess, "check_output", return_value=b"same compiler version"
            ), mock.patch.object(
                op.subprocess, "run", side_effect=compile_fixture
            ) as compile_, mock.patch.object(
                op.platform, "machine", return_value="aarch64"
            ) as machine:
                arm = op._build_library(152, dependencies)
                self.assertEqual(compile_.call_count, 2)
                self.assertEqual(op._build_library(152, dependencies), arm)
                self.assertEqual(compile_.call_count, 2)
                machine.return_value = "x86_64"
                x86 = op._build_library(152, dependencies)
                self.assertNotEqual(x86, arm)
                self.assertEqual(compile_.call_count, 4)
                machine.return_value = "aarch64"
                self.assertEqual(op._build_library(152, dependencies), arm)
                self.assertEqual(compile_.call_count, 4)

    def test_import_does_not_compile_load_or_initialize_cuda(self):
        with mock.patch.object(op.ctypes, "CDLL") as load, mock.patch.object(
            op.subprocess, "run"
        ) as compile_, mock.patch.object(torch.cuda, "_lazy_init") as init:
            fresh = _load("v41_wo_a_import_only", _ROOT / "_v41_wo_a_quant.py")
        load.assert_not_called()
        compile_.assert_not_called()
        init.assert_not_called()
        self.assertEqual(fresh._READY, {})
        self.assertEqual(fresh._LIBRARIES, {})

    def test_quant_knob_is_live_and_uses_flattened_wo_b_width(self):
        for mode, small, large in (
            ("auto", True, False),
            ("legacy", True, True),
            ("v2", False, False),
        ):
            with self.subTest(mode=mode), mock.patch.dict(
                os.environ, {"DSV4_FP8_QUANT_KERNEL": mode}
            ):
                self.assertEqual(op._legacy(511, None), small)
                self.assertEqual(op._legacy(512, None), large)
        with self.assertRaises(ValueError):
            op._legacy(33, "unknown")

    def test_supported_metadata_and_unprepared_fallback_never_load(self):
        a, w = _metadata()
        prop = SimpleNamespace(major=10, minor=0, multi_processor_count=152)
        with mock.patch.object(
            torch.cuda, "get_device_properties", return_value=prop
        ), mock.patch.dict(op._READY, {}, clear=True), mock.patch.object(
            op, "_load"
        ) as load:
            self.assertTrue(op.is_supported(a, w))
            self.assertIsNone(op.try_grouped_quant(a, w))
        load.assert_not_called()

    def test_incompatible_layout_dtype_device_and_alignment_fall_back(self):
        changes = (
            (0, "shape", (33, 2, 4096)),
            (0, "stride", lambda: (32768, 4096, 1)),
            (0, "dtype", torch.bfloat16),
            (0, "requires_grad", True),
            (1, "stride", lambda: (256, 32, 1)),
            (1, "device", torch.device("cuda:1")),
            (2, "is_contiguous", lambda: False),
            (2, "data_ptr", lambda: 257),
            (3, "dtype", torch.float32),
            (3, "stride", lambda: (32768, 32, 1)),
        )
        with mock.patch.object(op, "_device_supported", return_value=True):
            for index, attr, value in changes:
                with self.subTest(index=index, attr=attr):
                    a, w = _metadata()
                    setattr((*a, *w)[index], attr, value)
                    self.assertFalse(op.is_supported(a, w))

    def test_unverified_architecture_and_sms_fall_back(self):
        for major, minor, sms in (
            (9, 0, 132),
            (10, 3, 148),
            (10, 0, 151),
            (10, 0, 148),
        ):
            with self.subTest(arch=(major, minor, sms)), mock.patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(
                    major=major, minor=minor, multi_processor_count=sms
                ),
            ):
                self.assertFalse(op.is_supported(*_metadata()))

    def test_actual_attention_callsite_consumes_pair_and_retains_fallback(self):
        path = _ROOT / "attention_v41.py"
        cls = next(
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.ClassDef) and node.name == "AttentionV41FP8"
        )
        method = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "_project_output"
        )
        namespace = {}
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
            namespace,
        )

        class V41MXFP8Linear:
            def __init__(self):
                self.forward = mock.Mock()
                self.forward_quantized = mock.Mock()

            def __call__(self, *args, **kwargs):
                return self.forward(*args, **kwargs)

        class OtherLinear(V41MXFP8Linear):
            pass

        packages = {}
        for name in (
            "rtp_llm",
            "rtp_llm.models_py",
            "rtp_llm.models_py.modules",
            "rtp_llm.models_py.modules.dsv4",
            "rtp_llm.models_py.modules.dsv4.fp8",
            "rtp_llm.models_py.modules.dsv4.utils",
        ):
            packages[name] = ModuleType(name)
            packages[name].__path__ = []
        projection = SimpleNamespace(
            is_supported=mock.Mock(return_value=True),
            try_grouped_output_quant=mock.Mock(),
            grouped_output_projection=mock.Mock(),
        )
        packages["rtp_llm.models_py.modules.dsv4.fp8"]._v41_output_projection = (
            projection
        )
        packages["rtp_llm.models_py.modules.dsv4.utils"].V41MXFP8Linear = V41MXFP8Linear
        pair = (object(), object())
        x = torch.empty(2, 64, 512, dtype=torch.bfloat16)
        freqs, output, bf16 = object(), object(), object()
        for linear_cls, result_pair in (
            (V41MXFP8Linear, pair),
            (V41MXFP8Linear, None),
            (OtherLinear, pair),
        ):
            with self.subTest(
                linear=linear_cls.__name__, shared=result_pair is not None
            ), mock.patch.dict(sys.modules, packages):
                for operation in vars(projection).values():
                    operation.reset_mock()
                projection.try_grouped_output_quant.return_value = result_pair
                projection.grouped_output_projection.return_value = bf16
                linear = linear_cls()
                linear.forward.return_value = linear.forward_quantized.return_value = (
                    output
                )
                owner = SimpleNamespace(
                    n_heads=64,
                    head_dim=512,
                    wo_b=linear,
                    _wo_a_stk_w=object(),
                    _wo_a_stk_s=object(),
                )
                actual = namespace["_project_output"](owner, x, freqs, out=output)
                self.assertIs(actual, output)
                if linear_cls is V41MXFP8Linear and result_pair is not None:
                    linear.forward_quantized.assert_called_once_with(*pair, out=output)
                    linear.forward.assert_not_called()
                    projection.grouped_output_projection.assert_not_called()
                else:
                    linear.forward.assert_called_once_with(bf16, out=output)
                    linear.forward_quantized.assert_not_called()
                    projection.grouped_output_projection.assert_called_once()
                if linear_cls is OtherLinear:
                    projection.try_grouped_output_quant.assert_not_called()

    def test_cpu_and_empty_inputs_fall_back(self):
        x = torch.empty(0, 8, 4096, dtype=torch.float8_e4m3fn)
        self.assertFalse(op.is_supported((x, torch.empty(0)), (x, torch.empty(0))))
        self.assertFalse(op._device_supported(torch.device("cpu")))

    def test_output_dtype_shape_stride_and_overlap_are_rejected(self):
        a = (
            torch.empty(8, 4, 4096, dtype=torch.float8_e4m3fn).transpose(0, 1),
            torch.empty(4, 8, 32, dtype=torch.int32),
        )
        w = (torch.empty(1), torch.empty(1))
        good = (
            torch.empty(4, 8192, dtype=torch.float8_e4m3fn),
            torch.empty(64, 4, dtype=torch.int32).T,
        )
        op._validate_output(good, a, w)
        bad = (
            (good[0].float(), good[1]),
            (good[0][:, :4096], good[1]),
            (good[0], good[1].contiguous()),
            (a[0].transpose(0, 1).flatten()[: 4 * 8192].view(4, 8192), good[1]),
        )
        for output in bad:
            with self.subTest(shape=output[0].shape), self.assertRaises(ValueError):
                op._validate_output(output, a, w)

    def test_warmup_cannot_load_inside_capture(self):
        _, w = _metadata()
        with mock.patch.object(
            op, "_weight_supported", return_value=True
        ), mock.patch.object(torch.cuda, "device"), mock.patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=True
        ), mock.patch.object(
            op, "_load"
        ) as load, self.assertRaisesRegex(
            RuntimeError, "precede"
        ):
            op.warmup(*w)
        load.assert_not_called()

    def test_context_mismatch_falls_back_without_load_or_output_allocation(self):
        a, weight = _metadata()
        lib = SimpleNamespace(v41_wo_a_is_current=mock.Mock(return_value=0))
        entry = (lib, object())
        with mock.patch.dict(op._READY, {0: entry}, clear=True), mock.patch.object(
            op, "_device_supported", return_value=True
        ), mock.patch.object(torch.cuda, "device"), mock.patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ), mock.patch.object(
            op, "_load"
        ) as load, mock.patch.object(
            op, "_output"
        ) as output:
            self.assertFalse(op.is_ready(*weight))
            self.assertFalse(op.warmup(*weight))
            self.assertIsNone(op.try_grouped_quant(a, weight))
            self.assertIs(op._READY[0], entry)
            lib.v41_wo_a_is_current.return_value = 1
            self.assertTrue(op.is_ready(*weight))
            self.assertTrue(op.warmup(*weight))
        load.assert_not_called()
        output.assert_not_called()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class WoAQuantCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (10, 0):
            raise unittest.SkipTest("SM100 required")
        import deep_gemm
        from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor

        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8,
        )

        cls.dg = deep_gemm
        cls.pack = staticmethod(get_mn_major_tma_aligned_packed_ue8m0_tensor)
        cls.native_quant = staticmethod(sgl_per_token_group_quant_fp8)
        torch.manual_seed(20260923)
        cls.w = (torch.randn(8, 1024, 4096, device="cuda") * 0.1).to(
            torch.float8_e4m3fn
        )
        cls.w[:, :32].zero_()
        scales = torch.exp2(torch.randint(-3, 4, (8, 32, 128), device="cuda").float())
        cls.sw = cls.pack(scales.repeat_interleave(32, dim=1))
        if not op.warmup(cls.w, cls.sw):
            raise AssertionError("Test runtime must support the pinned CUDA13 epilogue")

    def pair(self, m, exponent=0):
        raw = torch.randn(8, m, 4096, device="cuda", dtype=torch.bfloat16)
        if m > 1:
            raw[:, 0].zero_()
        q = raw.to(torch.float8_e4m3fn).transpose(0, 1)
        raw_s = (
            torch.exp2(torch.randint(-3, 4, (8, m, 128), device="cuda").float())
            if exponent == 0
            else torch.full((8, m, 128), 2.0**exponent, device="cuda")
        )
        return q, self.pack(raw_s).transpose(0, 1)

    def baseline(self, a, mode):
        d = torch.empty(a[0].shape[0], 8, 1024, device="cuda", dtype=torch.bfloat16)
        self.dg.fp8_einsum("bhr,hdr->bhd", a, (self.w, self.sw), d, recipe=(1, 1, 32))
        with mock.patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": mode}):
            return self.native_quant(
                d.flatten(1),
                group_size=32,
                eps=torch.finfo(torch.float32).tiny,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )

    def assert_pair(self, actual, expected):
        self.assertIsNotNone(actual)
        self.assertTrue(
            torch.equal(actual[0].view(torch.uint8), expected[0].view(torch.uint8))
        )
        self.assertTrue(torch.equal(actual[1], expected[1]))

    @torch.inference_mode()
    def test_bitexact_layout_policy_and_ftz_boundaries(self):
        cases = [
            (m, 0)
            for m in (
                1,
                32,
                33,
                128,
                129,
                255,
                257,
                383,
                384,
                385,
                511,
                512,
                513,
                819,
                820,
                1021,
                4096,
                8192,
                32768,
            )
        ]
        cases += [(33, e) for e in (-35, -110, -120, -126)]
        for m, exponent in cases:
            a = self.pair(m, exponent)
            for mode in ("legacy", "v2", "auto"):
                with self.subTest(m=m, exponent=exponent, mode=mode), mock.patch.dict(
                    os.environ, {"DSV4_FP8_QUANT_KERNEL": mode}
                ):
                    self.assert_pair(
                        op.try_grouped_quant(a, (self.w, self.sw)),
                        self.baseline(a, mode),
                    )

    @torch.inference_mode()
    def test_nondefault_stream_and_graph_replay(self):
        a = self.pair(257)
        saved = tuple(t.clone() for t in a)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            eager = op.try_grouped_quant(a, (self.w, self.sw), quant_kernel="v2")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = op.try_grouped_quant(a, (self.w, self.sw), quant_kernel="v2")
            graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        self.assert_pair(eager, self.baseline(a, "v2"))
        self.assert_pair(captured, eager)
        self.assert_pair((a[0].contiguous(), a[1]), (saved[0].contiguous(), saved[1]))

    @torch.inference_mode()
    def test_cold_capture_fallback_does_not_load_or_compile(self):
        a = self.pair(33)
        sentinel = torch.zeros(1, device="cuda")
        with mock.patch.dict(op._READY, {}, clear=True), mock.patch.object(
            op, "_load"
        ) as load:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.assertIsNone(op.try_grouped_quant(a, (self.w, self.sw)))
                sentinel.add_(1)
            graph.replay()
        load.assert_not_called()
        self.assertEqual(sentinel.item(), 1)

    @torch.inference_mode()
    def test_same_device_alternate_context_falls_back_without_loading(self):
        driver = ctypes.CDLL("libcuda.so.1")
        driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        driver.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        driver.cuCtxCreate_v2.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
            ctypes.c_int,
        ]
        driver.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
        original, alternate = ctypes.c_void_p(), ctypes.c_void_p()
        self.assertEqual(driver.cuCtxGetCurrent(ctypes.byref(original)), 0)
        a = self.pair(33)
        torch.cuda.synchronize()
        entry = op._READY[self.w.device.index]
        self.assertTrue(op.is_ready(self.w, self.sw))
        try:
            self.assertEqual(
                driver.cuCtxCreate_v2(ctypes.byref(alternate), 0, self.w.device.index),
                0,
            )
            current = ctypes.c_void_p()
            self.assertEqual(driver.cuCtxGetCurrent(ctypes.byref(current)), 0)
            self.assertEqual(current.value, alternate.value)
            self.assertNotEqual(current.value, original.value)
            # A-side tensors supply metadata only: a cold return must not touch
            # their storage, allocate outputs, query a stream, or launch in B.
            with mock.patch.object(op, "_load") as load, mock.patch.object(
                op, "_output"
            ) as output, mock.patch.object(torch.cuda, "current_stream") as stream:
                self.assertFalse(op.is_ready(self.w, self.sw))
                self.assertFalse(op.warmup(self.w, self.sw))
                self.assertIsNone(op.try_grouped_quant(a, (self.w, self.sw)))
            load.assert_not_called()
            output.assert_not_called()
            stream.assert_not_called()
            self.assertIs(op._READY[self.w.device.index], entry)
            self.assertEqual(driver.cuCtxGetCurrent(ctypes.byref(current)), 0)
            self.assertEqual(current.value, alternate.value)
        finally:
            self.assertEqual(driver.cuCtxSetCurrent(original), 0)
            if alternate.value:
                self.assertEqual(driver.cuCtxDestroy_v2(alternate), 0)
        self.assertTrue(op.is_ready(self.w, self.sw))
        self.assertTrue(op.warmup(self.w, self.sw))
        self.assert_pair(
            op.try_grouped_quant(a, (self.w, self.sw), quant_kernel="v2"),
            self.baseline(a, "v2"),
        )

    @torch.inference_mode()
    def test_public_bf16_api_and_explicit_wo_b_consumer(self):
        from rtp_llm.models_py.modules.dsv4 import fp8
        from rtp_llm.models_py.modules.dsv4.utils import V41MXFP8Linear

        projection = _load("v41_output_product", _ROOT / "_v41_output_projection.py")
        linear = V41MXFP8Linear(
            (torch.randn(5120, 8192, device="cuda") * 0.02).to(torch.float8_e4m3fn),
            torch.ones(160, 256, device="cuda").to(torch.float8_e8m0fnu),
        )
        with mock.patch.object(fp8, "_v41_wo_a_quant", op, create=True):
            for m in (33, 511, 512, 1023, 1024):
                x = torch.randn(m, 64, 512, device="cuda", dtype=torch.bfloat16)
                original = x.clone()
                angles = torch.randn(m, 32, device="cuda")
                freqs = torch.polar(torch.ones_like(angles), angles)
                for mode in ("legacy", "v2", "auto"):
                    with self.subTest(m=m, mode=mode), mock.patch.dict(
                        os.environ, {"DSV4_FP8_QUANT_KERNEL": mode}
                    ):
                        bf16 = projection.grouped_output_projection(
                            x, freqs, self.w, self.sw
                        )
                        self.assertEqual(bf16.dtype, torch.bfloat16)
                        pair = projection.try_grouped_output_quant(
                            x, freqs, self.w, self.sw
                        )
                        expected = linear(bf16)
                        actual = linear.forward_quantized(*pair)
                        self.assertTrue(torch.equal(expected, actual))
                self.assertTrue(torch.equal(x, original))
            with mock.patch.dict(op._READY, {}, clear=True), mock.patch.object(
                projection, "_quantize_projection_input"
            ) as quant:
                self.assertIsNone(
                    projection.try_grouped_output_quant(x, freqs, self.w, self.sw)
                )
                quant.assert_not_called()

    @torch.inference_mode()
    def test_warmed_profile_has_one_gemm_and_no_module_load(self):
        a = self.pair(4096)
        for _ in range(3):
            op.try_grouped_quant(a, (self.w, self.sw))
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            for _ in range(4):
                op.try_grouped_quant(a, (self.w, self.sw))
            torch.cuda.synchronize()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            profile.export_chrome_trace(str(path))
            events = json.loads(path.read_text())["traceEvents"]
        kernels = [e for e in events if e.get("cat") == "kernel"]
        self.assertEqual(len(kernels), 4)
        self.assertTrue(all("v41_wo_a_group32_epilogue" in e["name"] for e in kernels))
        self.assertFalse(any("cuModuleLoad" in e.get("name", "") for e in events))


if __name__ == "__main__":
    unittest.main()
