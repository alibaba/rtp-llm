"""CPU dispatch contracts and qualified Blackwell GEMM epilogue regressions."""

import ast
import ctypes
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
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


def _metadata(m=33, *, row_major=False):
    device = torch.device("cuda:0")

    def tensor(shape, stride, dtype):
        return SimpleNamespace(
            shape=shape,
            ndim=len(shape),
            device=device,
            dtype=dtype,
            is_cuda=True,
            requires_grad=False,
            stride=lambda axis=None: stride if axis is None else stride[axis],
            is_contiguous=lambda: True,
            data_ptr=lambda: 256,
        )

    aligned = (m + 3) // 4 * 4
    return (
        tensor(
            (m, 8, 4096),
            (32768, 4096, 1) if row_major else (4096, m * 4096, 1),
            torch.float8_e4m3fn,
        ),
        tensor((m, 8, 32), (1, aligned * 32, aligned), torch.int32),
    ), (
        tensor((8, 1024, 4096), (1024 * 4096, 4096, 1), torch.float8_e4m3fn),
        tensor((8, 1024, 32), (32768, 1, 1024), torch.int32),
    )


class WoAQuantCPUContractTest(unittest.TestCase):
    def test_candidate_wheel_rejects_wrong_hash(self):
        validator = _load(
            "candidate_validation",
            Path(__file__).with_name("validate_wo_a_quant_wheel.py"),
        )
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "candidate.whl"
            wheel.write_bytes(b"candidate wheel bytes")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                validator.verify_wheel(wheel, "0" * 64)
            expected = hashlib.sha256(wheel.read_bytes()).hexdigest()
            self.assertEqual(validator.verify_wheel(wheel, expected), expected)

    def test_candidate_report_requires_both_gpu_classes_without_skips(self):
        validator = _load(
            "candidate_validation",
            Path(__file__).with_name("validate_wo_a_quant_wheel.py"),
        )
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "junit.xml"
            cpu = '<testcase classname="quant.WoAQuantCPUContractTest" name="cpu"/>'
            first = '<testcase classname="quant.WoAQuantCudaTest" name="gpu"/>'
            second = '<testcase classname="quant.ExternalQuantCudaTest" name="gpu">{}</testcase>'
            for cases, passed in (
                (cpu, False),
                (cpu + first, False),
                (
                    cpu + first + second.format('<skipped message="missing API"/>'),
                    False,
                ),
                (
                    cpu + first + second.format('<failure message="wrong output"/>'),
                    False,
                ),
                (cpu + first + second.format(""), True),
            ):
                with self.subTest(cases=cases):
                    report.write_text(
                        "<testsuites><testsuite>" + cases + "</testsuite></testsuites>"
                    )
                    counts = validator.summarize_results(report)
                    self.assertEqual(counts["accepted"], passed)
                    self.assertEqual(counts["cpu_passed"], 1)
                    if passed:
                        self.assertEqual(counts["gpu_passed"], 2)
                        self.assertEqual(counts["gpu_skipped"], 0)

    def test_missing_external_features_keep_fallback(self):
        for api in (
            SimpleNamespace(epilogue=SimpleNamespace()),
            SimpleNamespace(epilogue=SimpleNamespace(FP8Group32=object)),
        ):
            with mock.patch.dict(sys.modules, {"deep_gemm": api}):
                self.assertIsNone(op._external_api())

    def test_candidate_validation_requires_external_backend(self):
        with mock.patch.object(op, "_external_api", return_value=None):
            with mock.patch.dict(os.environ, {"RTP_V41_REQUIRE_EXTERNAL_QUANT": "0"}):
                with self.assertRaises(unittest.SkipTest):
                    _require_external_backend()
            with mock.patch.dict(os.environ, {"RTP_V41_REQUIRE_EXTERNAL_QUANT": "1"}):
                with self.assertRaises(AssertionError):
                    _require_external_backend()

    def test_prepared_signature_tracks_complete_external_configuration(self):
        a, _ = _metadata()
        api = SimpleNamespace(
            get_num_sms=lambda: 148,
            get_tc_util=lambda: 100,
            get_pdl=lambda: True,
            get_jit_config_fingerprint=mock.Mock(
                return_value=(148, 100, True, False, 1, 1)
            ),
        )
        before = op._signature(a, api)
        for config in ((148, 100, True, True, 1, 1), (148, 100, True, False, 64, 64)):
            api.get_jit_config_fingerprint.return_value = config
            self.assertNotEqual(before, op._signature(a, api))

    def test_ready_and_launch_enter_target_device_before_context_check(self):
        a, w = _metadata()
        for t in (*a, *w):
            t.device = torch.device("cuda:1")
        current = [0]

        @contextmanager
        def guard(device):
            previous = current[0]
            current[0] = device.index
            try:
                yield
            finally:
                current[0] = previous

        entry = dict(context=101, api=object(), prepared=set())
        out = (object(), object())
        with mock.patch.dict(op._READY, {1: entry}, clear=True), mock.patch.object(
            op, "_device_supported", return_value=True
        ), mock.patch.object(
            torch.cuda, "device", side_effect=guard
        ), mock.patch.object(
            op, "_context", side_effect=lambda: 100 + current[0]
        ), mock.patch.object(
            op, "_signature", return_value=(33,)
        ), mock.patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ), mock.patch.object(
            op, "_check_output"
        ), mock.patch.object(
            op, "_launch"
        ) as launch:
            self.assertTrue(op.is_ready(*w))
            self.assertIs(op.try_grouped_quant(a, w, out=out), out)
            launch.assert_called_once()
            self.assertEqual(current[0], 0)

    def setUp(self):
        op._device_geometry.cache_clear()
        self.addCleanup(op._device_geometry.cache_clear)

    def test_inverse_quant_buffers_elide_full_slices_and_cache_explicit_devices(self):
        inverse = _load(
            "v41_inverse_quant_contract",
            _ROOT.parent / "_fused_inv_rope_fp8_quant_triton.py",
        )
        fp8, scales = inverse._alloc_output_buffers(5, 2, 512, 4, torch.device("cpu"))
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as p:
            actual = inverse._alloc_output_buffers(
                5, 2, 512, 4, torch.device("cpu"), fp8, scales
            )
        self.assertIs(actual[0], fp8)
        self.assertIs(actual[1], scales)
        self.assertFalse(any(e.key.startswith("aten::") for e in p.key_averages()))
        short = inverse._alloc_output_buffers(
            3, 2, 512, 4, torch.device("cpu"), fp8, scales
        )
        self.assertEqual(short[0].shape, (2, 3, 512))
        self.assertEqual(short[1].stride(), scales.stride())
        with mock.patch.object(
            torch.cuda, "get_device_capability", side_effect=[(10, 3), (9, 0)]
        ) as get_cap:
            self.assertTrue(inverse._is_blackwell_device(torch.device("cuda:0")))
            self.assertTrue(inverse._is_blackwell_device(0))
            with mock.patch.object(torch.cuda, "current_device", return_value=1):
                self.assertFalse(inverse._is_blackwell_device())
            with mock.patch.object(torch.cuda, "current_device", return_value=0):
                self.assertTrue(inverse._is_blackwell_device(torch.device("cuda")))
            self.assertFalse(inverse._is_blackwell_device(torch.device("cpu")))
            self.assertEqual(get_cap.call_args_list, [mock.call(0), mock.call(1)])

    def test_output_scale_alias_padding_stride_and_full_range_slice_elision(self):
        for m in (0, 1, 3, 4, 5, 512):
            with self.subTest(m=m), torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as profile:
                q, scales = op._output(m, torch.device("cpu"))
            padded = (m + 3) // 4 * 4
            self.assertEqual(q.shape, (m, 8192))
            self.assertEqual(scales.shape, (m, 64))
            self.assertEqual(scales.stride(), (1, padded if padded else 1))
            self.assertEqual(scales.storage_offset(), 0)
            base = scales._base
            self.assertEqual(scales.untyped_storage().data_ptr(), base.data_ptr())
            base.fill_(-1)
            scales.fill_(17)
            self.assertTrue(bool((base[:, :m] == 17).all()))
            self.assertTrue(bool((base[:, m:] == -1).all()))
            slices = sum(
                e.count for e in profile.key_averages() if e.key == "aten::slice"
            )
            self.assertEqual(slices, int(padded != m))

    def test_inverse_quant_forward_config_and_public_environment_dispatch(self):
        inverse = _load(
            "v41_inverse_quant_dispatch",
            _ROOT.parent / "_fused_inv_rope_fp8_quant_triton.py",
        )
        optimized, legacy = mock.MagicMock(), mock.MagicMock()
        o = torch.empty(3, 8, 512, dtype=torch.bfloat16)
        freqs = torch.ones(3, 32, dtype=torch.complex64)
        with mock.patch.object(
            inverse, "_fused_inv_rope_fp8_quant_group_heads", optimized
        ), mock.patch.object(
            inverse, "_fused_inv_rope_fp8_quant_per_head", legacy
        ), mock.patch.dict(
            os.environ, {"DSV4_INV_ROPE_FP8_QUANT_IMPL": "legacy"}
        ):
            inverse.fused_inv_rope_fp8_quant(
                o, freqs, 1, 8, 448, 64, _launch_config=("optimized", "4", "2")
            )
            options = optimized.__getitem__.return_value.call_args.kwargs
            self.assertEqual(options["HEADS_PER_CTA"], 4)
            self.assertEqual(options["num_warps"], 2)
            legacy.__getitem__.assert_not_called()
            inverse.fused_inv_rope_fp8_quant(o, freqs, 1, 8, 448, 64)
            legacy.__getitem__.return_value.assert_called_once()

    def test_quantize_keeps_contiguous_copy_and_restores_leading_dimensions(self):
        linear_cls = self._linear_class()
        linear = linear_cls.__new__(linear_cls)
        torch.nn.Module.__init__(linear)
        linear.K = 8
        captured = []

        def quantize(flat, **kwargs):
            self.assertTrue(flat.is_contiguous())
            result = flat.clone()
            captured.append((flat, result))
            return result, object()

        path = "rtp_llm.models_py.kernels.cuda.fp8_kernel"
        with mock.patch.dict(
            sys.modules, {path: SimpleNamespace(sgl_per_token_group_quant_fp8=quantize)}
        ):
            inputs = (
                torch.arange(32).float().view(4, 8),
                torch.arange(64).float().view(4, 16)[:, ::2],
                torch.arange(48).float().view(2, 3, 8).transpose(0, 1),
                torch.arange(32).float().view(2, 16),
                torch.ones(4, 8, requires_grad=True),
            )
            for x in inputs:
                actual, _ = linear._quantize_input(x)
                flat, quantized = captured[-1]
                torch.testing.assert_close(actual, x)
                self.assertEqual(actual.shape, x.shape)
                self.assertEqual(flat.shape, (x.numel() // 8, 8))
                if (
                    x.ndim == 2
                    and x.shape[1] == 8
                    and x.is_contiguous()
                    and not x.requires_grad
                ):
                    self.assertIs(flat, x)
                    self.assertIs(actual, quantized)
                elif x.requires_grad:
                    self.assertIsNot(flat, x)
                    self.assertIsNot(actual, quantized)

    def test_quantized_exact_shape_keeps_noncontiguous_input_and_out_aliases(self):
        linear_cls = self._linear_class()
        linear = linear_cls.__new__(linear_cls)
        torch.nn.Module.__init__(linear)
        linear.N, linear.K = 3, 8
        linear.weight = torch.ones(3, 8)
        linear.weight_scales = torch.ones(1)
        q = torch.arange(64).float().view(4, 16)[:, ::2]
        storage = torch.full((4, 8), -1.0)
        out = storage[:, 1:7:2]

        def gemm(a, b, output, **kwargs):
            self.assertIs(a[0], q)
            self.assertIs(output, out)
            output.copy_(a[0] @ b[0].T)

        with mock.patch.dict(
            sys.modules, {"deep_gemm": SimpleNamespace(fp8_fp4_gemm_nt=gemm)}
        ):
            self.assertIs(linear.forward_quantized(q, torch.ones(1), out=out), out)
            torch.testing.assert_close(out, q @ linear.weight.T)
            self.assertTrue(bool((storage[:, ::2] == -1).all()))
            with self.assertRaises(RuntimeError):
                linear.forward_quantized(torch.zeros(2, 16), torch.ones(1))

    def test_linear_valid_quantized_calls_keep_output_and_gemm_geometry(self):
        linear_cls = self._linear_class()
        linear = linear_cls.__new__(linear_cls)
        torch.nn.Module.__init__(linear)
        linear.N, linear.K = 8, 128
        linear.weight = torch.ones(8, 128, dtype=torch.float8_e4m3fn)
        linear.weight_scales = torch.ones(8, 1, dtype=torch.int32)
        calls = []

        def gemm(a, b, output, *, recipe):
            calls.append((a, b, output, recipe))
            output.copy_(a[0].float() @ b[0].float().T)

        with mock.patch.dict(
            sys.modules, {"deep_gemm": SimpleNamespace(fp8_fp4_gemm_nt=gemm)}
        ):
            for shape in ((33, 128), (2, 17, 128), (0, 128)):
                for supplied in (False, True):
                    with self.subTest(shape=shape, supplied=supplied):
                        q = torch.ones(shape, dtype=torch.float8_e4m3fn)
                        rows = q.numel() // 128
                        scales = torch.empty(
                            1, max(1, (rows + 3) // 4 * 4), dtype=torch.int32
                        ).T[:rows]
                        out = (
                            torch.empty((*shape[:-1], 8), dtype=torch.bfloat16)
                            if supplied
                            else None
                        )
                        calls.clear()
                        actual = linear.forward_quantized(q, scales, out=out)
                        self.assertEqual(actual.shape, (*shape[:-1], 8))
                        self.assertEqual(actual.dtype, torch.bfloat16)
                        if supplied:
                            self.assertIs(actual, out)
                        self.assertEqual(len(calls), int(rows > 0))
                        if rows:
                            a, b, output, recipe = calls[0]
                            self.assertEqual(a[0].shape, (rows, 128))
                            self.assertIs(a[1], scales)
                            self.assertIs(b[0], linear.weight)
                            self.assertIs(b[1], linear.weight_scales)
                            self.assertEqual(output.shape, (rows, 8))
                            self.assertEqual(output.data_ptr(), actual.data_ptr())
                            self.assertEqual(recipe, (1, 1, 32))
                            torch.testing.assert_close(
                                actual, torch.full_like(actual, 128)
                            )

    def test_linear_forward_preserves_subclass_and_instance_dispatch(self):
        linear_cls = self._linear_class()
        calls = []
        pair, output = (object(), object()), object()

        class Custom(linear_cls):
            def __init__(self):
                torch.nn.Module.__init__(self)

            def _quantize_input(self, x):
                calls.append("quantize")
                return pair

            def forward_quantized(self, q, scales, out=None):
                calls.append(("subclass", q, scales, out))
                return output

        linear = Custom()
        x, out = torch.ones(2, 128), object()
        self.assertIs(linear.forward(x, out=out), output)
        self.assertEqual(calls, ["quantize", ("subclass", *pair, out)])
        linear.forward_quantized = mock.Mock(return_value=output)
        self.assertIs(linear.forward(x, out=out), output)
        linear.forward_quantized.assert_called_once_with(*pair, out=out)

    @staticmethod
    def _linear_class():
        path = _ROOT.parent / "utils.py"
        cls = next(
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.ClassDef) and node.name == "V41MXFP8Linear"
        )
        namespace = {"torch": torch}
        exec(
            compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        return namespace[cls.name]

    def test_import_does_not_compile_load_or_initialize_cuda(self):
        with mock.patch.object(op.ctypes, "CDLL") as load, mock.patch.object(
            subprocess, "run"
        ) as compile_, mock.patch.object(torch.cuda, "_lazy_init") as init:
            fresh = _load("v41_wo_a_import_only", _ROOT / "_v41_wo_a_quant.py")
        load.assert_not_called()
        compile_.assert_not_called()
        init.assert_not_called()
        self.assertEqual(fresh._READY, {})

    def test_fused_launch_uses_fixed_legacy_without_mode_argument(self):
        a, w = _metadata()
        out = (mock.Mock(), mock.Mock())
        api = mock.Mock()
        for tensor in (*a, *w):
            tensor.record_stream = mock.Mock()
        with mock.patch.dict(
            os.environ, {"DSV4_FP8_QUANT_KERNEL": "v2"}
        ), mock.patch.object(torch.cuda, "current_stream", return_value=object()):
            op._launch(dict(api=api), a, w, out)
        api.epilogue.FP8Group32.assert_called_once_with(out[1])

    def test_supported_metadata_and_unprepared_fallback_never_load(self):
        a, w = _metadata()
        prop = SimpleNamespace(major=10, minor=0, multi_processor_count=152)
        with mock.patch.object(
            torch.cuda, "get_device_properties", return_value=prop
        ), mock.patch.dict(op._READY, {}, clear=True), mock.patch.object(
            op, "_external_api"
        ) as load:
            self.assertTrue(op.is_supported(a, w))
            self.assertIsNone(op.try_grouped_quant(a, w))
        load.assert_not_called()

    def test_sm103_148_sms_is_supported_but_cold_calls_cannot_compile(self):
        a, w = _metadata()
        prop = SimpleNamespace(major=10, minor=3, multi_processor_count=148)
        with mock.patch.object(
            torch.cuda, "get_device_properties", return_value=prop
        ), mock.patch.dict(op._READY, {}, clear=True), mock.patch.object(
            op, "_external_api"
        ) as load:
            self.assertTrue(op.is_supported(a, w))
            self.assertIsNone(op.try_grouped_quant(a, w))
        load.assert_not_called()

    def test_incompatible_layout_dtype_device_and_alignment_fall_back(self):
        changes = (
            (0, "shape", (33, 2, 4096)),
            (0, "stride", lambda: (32784, 4096, 1)),
            (0, "stride", lambda: (32768, 4096, 2)),
            (0, "stride", lambda: (0, 4096, 1)),
            (0, "stride", lambda: (4096, 4096, 1)),
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

    def test_blackwell_family_uses_actual_even_sms_and_caches_properties(self):
        for index, (minor, sms) in enumerate(
            ((0, 152), (3, 148), (3, 144), (0, 148), (1, 120))
        ):
            with self.subTest(minor=minor, sms=sms), mock.patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(
                    major=10, minor=minor, multi_processor_count=sms
                ),
            ) as query:
                device = torch.device("cuda", index)
                self.assertTrue(op._device_supported(device))
                self.assertTrue(op._device_supported(device))
                self.assertEqual(op._device_geometry(index), (10, minor, sms))
                query.assert_called_once_with(index)
        with mock.patch.object(
            torch.cuda, "current_device", return_value=0
        ), mock.patch.object(
            torch.cuda,
            "get_device_properties",
            side_effect=AssertionError("cached device"),
        ):
            self.assertTrue(op._device_supported(torch.device("cuda")))

    def test_unsupported_family_or_noncluster_grid_falls_back(self):
        for major, minor, sms in ((9, 0, 132), (12, 0, 148), (10, 0, 151), (10, 3, 0)):
            op._device_geometry.cache_clear()
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
            "rtp_llm.models_py.modules.dsv41",
            "rtp_llm.models_py.modules.dsv41.fp8",
            "rtp_llm.models_py.modules.dsv41.utils",
        ):
            packages[name] = ModuleType(name)
            packages[name].__path__ = []
        projection = SimpleNamespace(
            is_supported=mock.Mock(return_value=True),
            _try_grouped_output_quant_supported=mock.Mock(),
            grouped_output_projection=mock.Mock(),
        )
        packages["rtp_llm.models_py.modules.dsv41.fp8"]._v41_output_projection = (
            projection
        )
        packages["rtp_llm.models_py.modules.dsv41.utils"].V41MXFP8Linear = (
            V41MXFP8Linear
        )
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
                projection._try_grouped_output_quant_supported.return_value = (
                    result_pair
                )
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
                projection.is_supported.assert_called_once_with(
                    x, freqs, owner._wo_a_stk_w, owner._wo_a_stk_s
                )
                if linear_cls is V41MXFP8Linear and result_pair is not None:
                    linear.forward_quantized.assert_called_once_with(*pair, out=output)
                    linear.forward.assert_not_called()
                    projection.grouped_output_projection.assert_not_called()
                else:
                    linear.forward.assert_called_once_with(bf16, out=output)
                    linear.forward_quantized.assert_not_called()
                    projection.grouped_output_projection.assert_called_once()
                if linear_cls is OtherLinear:
                    projection._try_grouped_output_quant_supported.assert_not_called()

    def test_cpu_and_empty_inputs_fall_back(self):
        x = torch.empty(0, 8, 4096, dtype=torch.float8_e4m3fn)
        self.assertFalse(op.is_supported((x, torch.empty(0)), (x, torch.empty(0))))
        self.assertFalse(op._device_supported(torch.device("cpu")))

    def test_warmup_cannot_load_inside_capture(self):
        _, w = _metadata()
        with mock.patch.object(
            op, "_weight_supported", return_value=True
        ), mock.patch.object(torch.cuda, "device"), mock.patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=True
        ), mock.patch.object(
            op, "_external_api"
        ) as load, self.assertRaisesRegex(
            RuntimeError, "precede"
        ):
            op.warmup(*w)
        load.assert_not_called()


def _require_external_backend():
    if op._external_api() is None:
        message = "Install the candidate DeepGEMM FP8Group32 wheel to run external quant GPU tests"
        if os.environ.get("RTP_V41_REQUIRE_EXTERNAL_QUANT") == "1":
            raise AssertionError(message)
        raise unittest.SkipTest(message)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class WoAQuantCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not op._device_supported(torch.device("cuda", torch.cuda.current_device())):
            raise unittest.SkipTest(
                "Blackwell SM10x with two-CTA cluster grid required"
            )
        _require_external_backend()
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
            raise AssertionError(
                "Test runtime must support the external DeepGEMM FP8Group32 epilogue"
            )

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

    def baseline(self, a):
        d = torch.empty(a[0].shape[0], 8, 1024, device="cuda", dtype=torch.bfloat16)
        self.dg.fp8_einsum("bhr,hdr->bhd", a, (self.w, self.sw), d, recipe=(1, 1, 32))
        with mock.patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": "legacy"}):
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
            with self.subTest(m=m, exponent=exponent), mock.patch.dict(
                os.environ, {"DSV4_FP8_QUANT_KERNEL": "legacy"}
            ):
                self.assert_pair(
                    op.try_grouped_quant(a, (self.w, self.sw)),
                    self.baseline(a),
                )

    @torch.inference_mode()
    def test_token_major_mega_layout_is_bitexact_without_repacking(self):
        for m in (4, 24, 257, 4096, 12288):
            original = self.pair(m)
            token_major = (original[0].contiguous(), original[1])
            self.assertEqual(token_major[0].stride(), (32768, 4096, 1))
            self.assertTrue(op.is_supported(token_major, (self.w, self.sw)))
            with self.subTest(m=m):
                actual = op.try_grouped_quant(token_major, (self.w, self.sw))
                self.assert_pair(
                    actual,
                    op.try_grouped_quant(original, (self.w, self.sw)),
                )
                self.assert_pair(actual, self.baseline(original))
            # Neither accepted ordering implies support for an overlapping
            # group layout or incorrectly packed scales.
            bad_layout = token_major[0].as_strided((m, 8, 4096), (32768, 0, 1))
            self.assertIsNone(
                op.try_grouped_quant((bad_layout, token_major[1]), (self.w, self.sw))
            )
            self.assertIsNone(
                op.try_grouped_quant(
                    (token_major[0], token_major[1].contiguous()), (self.w, self.sw)
                )
            )

    @torch.inference_mode()
    def test_token_major_graph_replay_reads_updated_data_and_scales(self):
        for m in (4, 24):
            with self.subTest(m=m):
                pair = self.pair(m)
                pair = (pair[0].contiguous(), pair[1])
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    self.assertIsNotNone(op.try_grouped_quant(pair, (self.w, self.sw)))
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        captured = op.try_grouped_quant(pair, (self.w, self.sw))
                    self.assertIsNotNone(captured)
                    replacement = self.pair(m, exponent=-3)
                    pair[0].copy_(replacement[0])
                    pair[1].copy_(replacement[1])
                    graph.replay()
                torch.cuda.current_stream().wait_stream(stream)
                group_major = (
                    pair[0].transpose(0, 1).contiguous().transpose(0, 1),
                    pair[1],
                )
                self.assert_pair(captured, self.baseline(group_major))

    @torch.inference_mode()
    def test_nondefault_stream_and_graph_replay(self):
        a = self.pair(257)
        saved = tuple(t.clone() for t in a)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            eager = op.try_grouped_quant(a, (self.w, self.sw))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = op.try_grouped_quant(a, (self.w, self.sw))
            graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        self.assert_pair(eager, self.baseline(a))
        self.assert_pair(captured, eager)
        self.assert_pair((a[0].contiguous(), a[1]), (saved[0].contiguous(), saved[1]))

    @torch.inference_mode()
    def test_cold_capture_fallback_does_not_load_or_compile(self):
        a = self.pair(33)
        sentinel = torch.zeros(1, device="cuda")
        with mock.patch.dict(op._READY, {}, clear=True), mock.patch.object(
            op, "_external_api"
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
            with mock.patch.object(op, "_external_api") as load, mock.patch.object(
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
            op.try_grouped_quant(a, (self.w, self.sw)),
            self.baseline(a),
        )

    @torch.inference_mode()
    def test_public_bf16_api_and_explicit_wo_b_consumer(self):
        from rtp_llm.models_py.modules.dsv41 import fp8
        from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear

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
                with self.subTest(m=m), mock.patch.dict(
                    os.environ, {"DSV4_FP8_QUANT_KERNEL": "legacy"}
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
        self.assertTrue(all("sm100_fp8_fp4_gemm" in e["name"] for e in kernels))
        self.assertFalse(any("cuModuleLoad" in e.get("name", "") for e in events))


def reference_quant(x):
    x = x.float().flatten(1).reshape(x.shape[0], -1, 32)
    tiny = torch.finfo(torch.float32).tiny
    raw = x.abs().amax(-1).clamp_min(tiny) * (1.0 / 448.0)
    exponent = torch.ceil(torch.log2(raw.clamp_min(1e-10))).int() + 127
    inv = ((254 - exponent) << 23).view(torch.float32)
    x = torch.where(x.abs() < tiny, torch.copysign(torch.zeros_like(x), x), x)
    y = x * inv.unsqueeze(-1)
    y = torch.where(y.abs() < tiny, torch.copysign(torch.zeros_like(y), y), y)
    q = y.clamp(-448, 448).to(torch.float8_e4m3fn).flatten(1)
    packed = exponent.to(torch.uint8).contiguous().view(torch.int32)
    sf = torch.empty_strided(
        packed.shape, (1, (x.shape[0] + 3) // 4 * 4), device=x.device, dtype=torch.int32
    )
    sf.copy_(packed)
    return q, sf


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ExternalQuantCudaTest(unittest.TestCase):
    def test_output_remains_legacy_with_v2_environment(self):
        a = self.pair(33, exp=92)
        expected = self.baseline(a)
        with mock.patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": "v2"}):
            actual = op.try_grouped_quant(a, (self.w, self.sw))
            self.equal(actual, expected)
            self.assertEqual(int(actual[1][0, 0].item()) & 255, 94)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = op.try_grouped_quant(a, (self.w, self.sw))
            graph.replay()
            self.equal(captured, expected)

    def test_changed_compile_configuration_is_cold_during_capture(self):
        a = self.pair(33)
        original = self.dg.get_jit_config_fingerprint()
        for setter, changed, restored in (
            (self.dg.set_ignore_compile_dims, not original[3], original[3]),
            (self.dg.set_block_size_multiple_of, (64, 64), original[4:6]),
        ):
            self.assertIsNotNone(op.try_grouped_quant(a, (self.w, self.sw)))
            try:
                setter(changed)
                graph = torch.cuda.CUDAGraph()
                sentinel = torch.zeros(1, device=self.w.device)
                with mock.patch.object(
                    op, "_launch", side_effect=AssertionError("cold JIT launch")
                ):
                    with torch.cuda.graph(graph):
                        self.assertIsNone(op.try_grouped_quant(a, (self.w, self.sw)))
                        sentinel.add_(1)
                graph.replay()
                self.assertEqual(sentinel.item(), 1)
                self.equal(
                    op.try_grouped_quant(a, (self.w, self.sw)),
                    self.baseline(a),
                )
                with torch.cuda.graph(torch.cuda.CUDAGraph()):
                    self.assertIsNotNone(op.try_grouped_quant(a, (self.w, self.sw)))
            finally:
                setter(restored)

    @unittest.skipUnless(torch.cuda.device_count() > 1, "Two CUDA devices required")
    def test_warmed_noncurrent_device(self):
        # DeepJIT in release_v2.8.2 binds its cached CUfunctions to the first
        # backend device. Match one GPU per RTP worker, with a fresh process
        # whose backend is initialized on GPU1 before switching current device.
        script = r"""
import importlib.util, sys, torch
torch.cuda.set_device(1)
spec = importlib.util.spec_from_file_location("wo_a_device_test", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cls = module.ExternalQuantCudaTest
cls.setUpClass()
case = cls()
a = case.pair(33)
expected = case.baseline(a)
out = module.op._output(33, cls.w.device)
with torch.cuda.device(0):
    assert module.op.is_ready(cls.w, cls.sw)
    actual = module.op.try_grouped_quant(a, (cls.w, cls.sw), out=out)
    case.equal(actual, expected)
    assert torch.cuda.current_device() == 0
print("Warmed GPU1 called from current GPU0: PASS")
"""
        subprocess.run(
            [sys.executable, "-c", script, str(Path(__file__).resolve())],
            check=True,
            timeout=240,
        )

    @classmethod
    def setUpClass(cls):
        _require_external_backend()
        import deep_gemm as dg

        cls.dg = dg
        if torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest("Blackwell required")
        torch.manual_seed(20261019)
        cls.w = (torch.randn(8, 1024, 4096, device="cuda") / 16).to(torch.float8_e4m3fn)
        cls.w[:, :32] = 0
        cls.sw = torch.full(
            (8, 32, 1024), 0x7F7F7F7F, device="cuda", dtype=torch.int32
        ).transpose(1, 2)
        assert op.warmup(
            cls.w, cls.sw
        ), "Install DeepGEMM with the external FP8Group32 epilogue"

    def pair(self, m, token=False, exp=127):
        q = (
            (torch.randn(8, m, 4096, device="cuda") / 4)
            .to(torch.float8_e4m3fn)
            .transpose(0, 1)
        )
        if token:
            q = q.contiguous()
        word = sum(exp << (8 * i) for i in range(4))
        if word >= 2**31:
            word -= 2**32
        s = (
            torch.full(
                (8, 32, (m + 3) // 4 * 4), word, device="cuda", dtype=torch.int32
            )
            .transpose(1, 2)[:, :m]
            .transpose(0, 1)
        )
        return q, s

    def baseline(self, a):
        z = torch.empty((a[0].shape[0], 8, 1024), device="cuda", dtype=torch.bfloat16)
        self.dg.fp8_einsum("bhr,hdr->bhd", a, (self.w, self.sw), z, recipe=(1, 1, 32))
        return reference_quant(z)

    def equal(self, x, y):
        self.assertIsNotNone(x)
        self.assertTrue(torch.equal(x[0].view(torch.uint8), y[0].view(torch.uint8)))
        self.assertTrue(torch.equal(x[1], y[1]))

    def test_layouts_tails_zero_and_tiny(self):
        for m, exp in [
            (1, 127),
            (5, 127),
            (33, 127),
            (192, 127),
            (385, 127),
            (512, 127),
            (513, 127),
            (33, 92),
            (33, 17),
            (33, 7),
            (33, 1),
        ]:
            for token in (False, True):
                a = self.pair(m, token, exp)
                with self.subTest(m=m, exp=exp, token=token):
                    self.equal(
                        op.try_grouped_quant(a, (self.w, self.sw)),
                        self.baseline(a),
                    )

    def test_reused_output_changed_graph_inputs_and_side_stream(self):
        for token in (False, True):
            a = self.pair(33, token)
            out = op._output(33, self.w.device)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self.assertIs(
                    op.try_grouped_quant(a, (self.w, self.sw), out=out),
                    out,
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    self.assertIs(
                        op.try_grouped_quant(a, (self.w, self.sw), out=out),
                        out,
                    )
                a[0].zero_()
                a[1].fill_(0x01010101)
                graph.replay()
                self.equal(out, self.baseline(a))
            torch.cuda.current_stream().wait_stream(stream)

    def test_cold_capture_context_guard_and_invalid_output(self):
        a = self.pair(517)
        sentinel = torch.zeros(1, device="cuda")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.assertIsNone(op.try_grouped_quant(a, (self.w, self.sw)))
            sentinel.add_(1)
        graph.replay()
        self.assertEqual(sentinel.item(), 1)
        entry = op._READY[self.w.device.index]
        with mock.patch.object(op, "_context", return_value=-1):
            self.assertFalse(op.is_ready(self.w, self.sw))
            self.assertIsNone(op.try_grouped_quant(a, (self.w, self.sw)))
        self.assertIs(op._READY[self.w.device.index], entry)
        with mock.patch.dict(op._READY, {}, clear=True), mock.patch.object(
            op, "_external_api", side_effect=AssertionError("cold load")
        ):
            self.assertIsNone(op.try_grouped_quant(a, (self.w, self.sw)))
        out = op._output(517, self.w.device)
        with self.assertRaises(ValueError):
            op.try_grouped_quant(
                a, (self.w, self.sw), out=(out[0], out[1].contiguous())
            )
        with self.assertRaises(ValueError):
            op.try_grouped_quant(
                a,
                (self.w, self.sw),
                out=(a[0].contiguous().view(517, -1)[:, :8192], out[1]),
            )
        # An eager call prepares the previously cold signature.
        self.equal(op.try_grouped_quant(a, (self.w, self.sw)), self.baseline(a))

    def test_actual_wo_b_forward_quantized_consumes_pair(self):
        path = Path(__file__).resolve().parents[2] / "utils.py"
        node = next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "V41MXFP8Linear"
        )
        from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor

        namespace = dict(
            torch=torch,
            get_mn_major_tma_aligned_packed_ue8m0_tensor=get_mn_major_tma_aligned_packed_ue8m0_tensor,
        )
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        linear = namespace[node.name](
            (torch.randn(5120, 8192, device="cuda") / 64).to(torch.float8_e4m3fn),
            torch.ones(160, 256, device="cuda"),
        )
        a = self.pair(33, True)
        actual_pair = op.try_grouped_quant(a, (self.w, self.sw))
        expected = linear.forward_quantized(*self.baseline(a))
        out = torch.empty_like(expected)
        self.assertIs(linear.forward_quantized(*actual_pair, out=out), out)
        self.assertTrue(torch.equal(out, expected))


if __name__ == "__main__":
    unittest.main()
