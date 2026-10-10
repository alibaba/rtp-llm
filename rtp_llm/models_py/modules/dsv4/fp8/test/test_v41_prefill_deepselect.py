"""FP32 token top-k contract, allowing legal cutoff ties and output order.

GPU execution is explicit; candidate-block selection is outside this adapter.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_deepselect as selector
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_topk as topk


def reference(logits, visible):
    columns = torch.arange(logits.shape[1], device=logits.device)
    masked = logits.masked_fill(
        columns[None] >= visible[:, None].clamp(0, logits.shape[1]), -torch.inf
    )
    values, indices = masked.topk(512, dim=-1)
    return torch.where(values.isfinite(), indices, -1).int()


def assert_equivalent(case, actual, expected, logits, visible):
    """Require unique in-range indices and exact selected FP32 value multisets."""
    case.assertIsNotNone(actual)
    case.assertEqual(actual.dtype, torch.int32)
    case.assertEqual(actual.shape, (logits.shape[0], 512))
    ends = visible.clamp(0, logits.shape[1])
    valid = actual >= 0
    case.assertTrue(((actual == -1) | (valid & (actual < ends[:, None]))).all().item())
    ordered = actual.masked_fill(~valid, logits.shape[1]).sort(-1).values
    duplicate = (ordered[:, 1:] == ordered[:, :-1]) & (
        ordered[:, 1:] != logits.shape[1]
    )
    case.assertFalse(duplicate.any().item())
    actual_values = logits.gather(1, actual.long().clamp_min(0)).masked_fill(
        ~valid, -torch.inf
    )
    expected_values = logits.gather(1, expected.long().clamp_min(0)).masked_fill(
        expected < 0, -torch.inf
    )
    torch.testing.assert_close(
        actual_values.sort(-1).values, expected_values.sort(-1).values, rtol=0, atol=0
    )
    torch.testing.assert_close(valid.sum(-1), (expected >= 0).sum(-1), rtol=0, atol=0)


def metadata_fixture():
    device = torch.device("cuda:1")

    def storage(address):
        return SimpleNamespace(data_ptr=lambda: address, nbytes=lambda: 7 * 1280 * 4)

    logits = SimpleNamespace(
        is_cuda=True,
        dtype=torch.float32,
        ndim=2,
        shape=(7, 1024),
        device=device,
        stride=lambda axis: (1280, 1)[axis],
        data_ptr=lambda: 4096,
        storage_offset=lambda: 0,
        element_size=lambda: 4,
        untyped_storage=lambda: storage(4096),
    )
    ends = SimpleNamespace(
        dtype=torch.int32,
        ndim=1,
        device=device,
        numel=lambda: 7,
        stride=lambda axis: 1,
        is_contiguous=lambda: True,
        untyped_storage=lambda: storage(8192),
    )
    out = SimpleNamespace(
        device=device,
        dtype=torch.int32,
        shape=(7, 512),
        stride=lambda axis: (520, 1)[axis],
        data_ptr=lambda: 12288,
        untyped_storage=lambda: storage(12288),
    )
    return logits, ends, out


class PrefillDeepSelectCPU(unittest.TestCase):
    def setUp(self):
        feature = patch.dict(os.environ, {"DSV41_PREFILL_DEEPSELECT": "1"})
        feature.start()
        self.addCleanup(feature.stop)

    def test_cpu_and_disabled_fallback_do_not_query_cuda(self):
        logits = torch.zeros((2, 1024))
        ends = torch.zeros(2, dtype=torch.int32)
        module = SimpleNamespace(
            deepselect_fp32_available=Mock(return_value=True), deepselect_fp32=Mock()
        )
        with patch.object(
            selector, "_device_supported", return_value=True
        ) as query, patch.object(selector, "rtp_llm_ops", module):
            self.assertIsNone(selector.try_select_tokens(logits, ends))
            fake, bounds, _ = metadata_fixture()
            with patch.dict(os.environ, {"DSV41_PREFILL_DEEPSELECT": "0"}):
                self.assertFalse(selector.is_available(fake.device))
                self.assertIsNone(selector.try_select_tokens(fake, bounds))
            with patch.dict(os.environ):
                os.environ.pop("DSV41_PREFILL_DEEPSELECT", None)
                self.assertFalse(selector.is_available(fake.device))
                self.assertIsNone(selector.try_select_tokens(fake, bounds))
        query.assert_not_called()
        module.deepselect_fp32_available.assert_not_called()
        module.deepselect_fp32.assert_not_called()

    def test_supported_architectures_use_torch_capability(self):
        selector._device_supported.cache_clear()
        self.addCleanup(selector._device_supported.cache_clear)
        for capability in ((8, 9), (9, 0), (10, 0), (10, 3), (12, 0)):
            selector._device_supported.cache_clear()
            with patch.object(
                torch.cuda, "get_device_capability", return_value=capability
            ):
                self.assertEqual(
                    selector._device_supported(torch.device("cuda:0")),
                    capability in ((10, 0), (10, 3)),
                )

    def test_missing_or_disabled_native_binding_falls_back(self):
        modules = (
            SimpleNamespace(),
            SimpleNamespace(
                deepselect_fp32_available=lambda: False, deepselect_fp32=Mock()
            ),
        )
        with patch.object(selector, "_device_supported", return_value=True):
            for module in modules:
                with patch.object(selector, "rtp_llm_ops", module):
                    self.assertFalse(selector.is_available(torch.device("cuda:0")))
            module = SimpleNamespace(
                deepselect_fp32_available=lambda: True, deepselect_fp32=Mock()
            )
            with patch.object(selector, "rtp_llm_ops", module):
                self.assertTrue(selector.is_available(torch.device("cuda:0")))
        logits, ends, _ = metadata_fixture()
        with patch.object(selector, "is_available", return_value=False):
            self.assertIsNone(selector.try_select_tokens(logits, ends))

    def test_native_availability_errors_propagate(self):
        module = SimpleNamespace(
            deepselect_fp32_available=Mock(side_effect=RuntimeError("binding failure")),
            deepselect_fp32=Mock(),
        )
        with patch.object(
            selector, "_device_supported", return_value=True
        ), patch.object(selector, "rtp_llm_ops", module):
            with self.assertRaisesRegex(RuntimeError, "binding failure"):
                selector.is_available(torch.device("cuda:0"))

    def test_metadata_admission_and_rejected_layouts(self):
        with patch.dict(os.environ, {"DSV41_PREFILL_DEEPSELECT": "1"}), patch.object(
            selector, "is_available", return_value=True
        ):
            logits, ends, out = metadata_fixture()
            self.assertTrue(selector.is_supported(logits, ends, out))
            self.assertFalse(selector.is_supported(logits, ends, filter_finite=False))
            self.assertTrue(
                selector.is_supported(logits, ends, out, filter_finite=False)
            )
            cases = (
                (0, {"is_cuda": False}),
                (0, {"dtype": torch.bfloat16}),
                (0, {"shape": (0, 1024)}),
                (0, {"shape": (7, 511)}),
                (0, {"shape": (7, 2**23)}),
                (0, {"stride": lambda axis: (2048, 2)[axis]}),
                (0, {"stride": lambda axis: (512, 1)[axis]}),
                (0, {"stride": lambda axis: (1281, 1)[axis]}),
                (0, {"data_ptr": lambda: 4100}),
                (1, {"device": torch.device("cuda:0")}),
                (1, {"dtype": torch.int64}),
                (1, {"numel": lambda: 8}),
                (1, {"stride": lambda axis: 2, "is_contiguous": lambda: False}),
                (2, {"device": torch.device("cuda:0")}),
                (2, {"dtype": torch.int64}),
                (2, {"shape": (7, 511)}),
                (2, {"stride": lambda axis: (513, 1)[axis]}),
                (2, {"data_ptr": lambda: 12292}),
            )
            for tensor_index, changes in cases:
                tensors = metadata_fixture()
                vars(tensors[tensor_index]).update(changes)
                with self.subTest(tensor=tensor_index, fields=tuple(changes)):
                    self.assertFalse(selector.is_supported(*tensors))
            for index in (0, 1):
                tensors = metadata_fixture()
                tensors[2].untyped_storage = tensors[index].untyped_storage
                self.assertFalse(selector.is_supported(*tensors))

    def test_non32_width_requires_real_last_row_padding(self):
        with patch.object(selector, "is_available", return_value=True):
            for width, stride in (
                (545, 768),
                (801, 1024),
                (1837, 2048),
                (10004, 10240),
            ):
                for offset in (0, 16):
                    with self.subTest(width=width, stride=stride, offset=offset):
                        logits, ends, out = metadata_fixture()
                        logits.shape = (7, width)
                        logits.stride = lambda axis: (stride, 1)[axis]
                        logits.storage_offset = lambda: offset
                        logits.data_ptr = lambda: 4096 + 4 * offset
                        required = (offset + 6 * stride + (width + 31) // 32 * 32) * 4
                        storage = SimpleNamespace(
                            data_ptr=lambda: 4096, nbytes=lambda: required
                        )
                        logits.untyped_storage = lambda: storage
                        self.assertTrue(selector.is_supported(logits, ends, out))
                        storage.nbytes = lambda: required - 4
                        self.assertFalse(selector.is_supported(logits, ends, out))
                        storage.nbytes = lambda: (offset + 6 * stride + width) * 4
                        self.assertFalse(selector.is_supported(logits, ends, out))

    def test_native_arguments_and_execution_errors_propagate(self):
        logits, ends, out = metadata_fixture()
        module = SimpleNamespace(deepselect_fp32=Mock())
        with patch.object(selector, "is_supported", return_value=True), patch.object(
            selector, "rtp_llm_ops", module
        ):
            self.assertIs(selector.try_select_tokens(logits, ends, out=out), out)
            module.deepselect_fp32.assert_called_once_with(logits, ends, out, True)
            module.deepselect_fp32.reset_mock()
            self.assertIs(
                selector.try_select_tokens(logits, ends, out=out, filter_finite=False),
                out,
            )
            module.deepselect_fp32.assert_called_once_with(logits, ends, out, False)
            module.deepselect_fp32.side_effect = RuntimeError("backend launch failure")
            with self.assertRaisesRegex(RuntimeError, "backend launch failure"):
                selector.try_select_tokens(logits, ends, out=out)

    def test_public_raw_contract_still_requires_bounds_and_out(self):
        with patch.object(topk, "is_supported", return_value=True), patch.object(
            topk, "_select_tokens", side_effect=AssertionError("must not launch")
        ):
            self.assertIsNone(topk.try_select_tokens(None, None, filter_finite=False))


class PrefillDeepSelectCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("RTP_V41_PREFILL_DEEPSELECT_GPU_TEST") != "1":
            raise unittest.SkipTest("set RTP_V41_PREFILL_DEEPSELECT_GPU_TEST=1")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            raise unittest.SkipTest("SM100/SM103 required")
        feature = patch.dict(os.environ, {"DSV41_PREFILL_DEEPSELECT": "1"})
        feature.start()
        cls.addClassCleanup(feature.stop)
        if not selector.is_available(torch.device("cuda", torch.cuda.current_device())):
            raise RuntimeError(
                "explicit SM100/SM103 GPU test requires native FP32 DeepSelect binding/build"
            )

    def setUp(self):
        torch.manual_seed(719)

    def check(self, logits, visible, **kwargs):
        actual = selector.try_select_tokens(logits, visible, **kwargs)
        assert_equivalent(self, actual, reference(logits, visible), logits, visible)
        return actual

    def test_strided_scores_and_noncontiguous_int64_bounds(self):
        for width in (512, 544, 768, 4128, 16384, 32768):
            with self.subTest(width=width):
                backing = torch.randn(
                    (13, ((width + 255) // 256 + 1) * 256), device="cuda"
                )
                logits = backing[:, :width]
                original = backing.clone()
                visible = torch.tensor(
                    [
                        -(2**40),
                        -1,
                        0,
                        1,
                        17,
                        511,
                        512,
                        513,
                        769,
                        width - 1,
                        width,
                        width + 99,
                        2**40,
                    ],
                    device="cuda",
                    dtype=torch.int64,
                )
                visible = torch.stack((visible, visible), 1)[:, 0]
                with patch.object(
                    selector.rtp_llm_ops,
                    "deepselect_fp32",
                    wraps=selector.rtp_llm_ops.deepselect_fp32,
                ) as launch:
                    actual = topk.try_select_tokens(logits, visible)
                launch.assert_called_once()
                assert_equivalent(
                    self, actual, reference(logits, visible), logits, visible
                )
                torch.testing.assert_close(backing, original, rtol=0, atol=0)

    def test_nan_inf_short_long_and_outside_visible_range(self):
        width = 4128
        logits = torch.randn((17, 4352), device="cuda")[:, :width]
        visible = torch.tensor(
            [0, 1, 17, 511, 512, 513, 1024] + [width] * 10,
            dtype=torch.int32,
            device="cuda",
        )
        logits[:, 0] = float("nan")
        logits[:, 1] = -float("nan")
        logits[:, 2] = torch.inf
        logits[:, 3] = -torch.inf
        logits[7, 600:] = float("nan")
        visible[7] = 600
        logits[8].fill_(0)
        logits[9].fill_(-torch.inf)
        logits[10].fill_(torch.inf)
        logits[11].fill_(-float("nan"))
        logits[12, ::3] = -torch.inf
        for row, count in enumerate((513, 1025, 2049, 3073), start=13):
            logits[row, :count] = -float("nan")
        self.check(logits, visible)

    def test_fp32_adjacent_values_subnormals_extrema_and_ties(self):
        width = 8192
        logits = torch.zeros((7, width), device="cuda")
        bits = torch.arange(width, dtype=torch.int32) + 0x3F800000
        logits[0].copy_(bits.view(torch.float32).to("cuda"))
        logits[1].copy_(-bits.view(torch.float32).to("cuda"))
        tiny = torch.arange(1, width + 1, dtype=torch.int32).view(torch.float32)
        logits[2].copy_(tiny.to("cuda"))
        logits[3].copy_(-tiny.to("cuda"))
        logits[4, ::2] = -0.0
        logits[5, :700] = torch.finfo(torch.float32).min
        logits[5, 700:1400] = torch.finfo(torch.float32).max
        logits[6, :700] = 7
        self.check(logits, torch.full((7,), width, device="cuda", dtype=torch.int32))

    def test_out_storage_identity_padding_and_canaries(self):
        logits = torch.randn((7, 1024), device="cuda")[:, :768]
        visible = torch.tensor(
            [0, 1, 17, 511, 512, 513, 768], device="cuda", dtype=torch.int32
        )
        storage = torch.full((7, 528), 1234567, device="cuda", dtype=torch.int32)
        out = storage[:, 8:520]
        self.assertIs(self.check(logits, visible, out=out), out)
        self.assertTrue((storage[:, :8] == 1234567).all().item())
        self.assertTrue((storage[:, 520:] == 1234567).all().item())

    def test_unaligned_layouts_use_existing_fallback(self):
        for width, offset, stride in ((1024, 1, 1280), (1024, 0, 1025)):
            with self.subTest(width=width, offset=offset, stride=stride):
                logits = torch.randn((3, stride), device="cuda")[
                    :, offset : offset + width
                ]
                ends = torch.tensor([width, 17, 0], device="cuda", dtype=torch.int32)
                self.assertIsNone(selector.try_select_tokens(logits, ends))
                with patch.object(
                    selector.rtp_llm_ops,
                    "deepselect_fp32",
                    wraps=selector.rtp_llm_ops.deepselect_fp32,
                ) as launch:
                    actual = topk.try_select_tokens(logits, ends)
                launch.assert_not_called()
                assert_equivalent(self, actual, reference(logits, ends), logits, ends)

    def test_native_rejects_strided_dtype_view_aliases(self):
        storage = torch.randn((3, 1536), device="cuda")
        logits = storage[:, :1024]
        ends = torch.full((3,), 1024, device="cuda", dtype=torch.int32)
        # Both the overlapping view and the disjoint row-gap view share storage.
        for start in (512, 1024):
            with self.subTest(output_start=start):
                out = storage.view(torch.int32)[:, start : start + 512]
                self.assertIsNone(selector.try_select_tokens(logits, ends, out=out))
                with self.assertRaises(RuntimeError):
                    selector.rtp_llm_ops.deepselect_fp32(logits, ends, out, True)
        shared_ints = torch.empty((3 * 512,), device="cuda", dtype=torch.int32)
        alias_ends = shared_ints[:3]
        alias_ends.fill_(1024)
        out = shared_ints.view(3, 512)
        self.assertIsNone(selector.try_select_tokens(logits, alias_ends, out=out))
        with self.assertRaises(RuntimeError):
            selector.rtp_llm_ops.deepselect_fp32(logits, alias_ends, out, True)

    def test_large_rows_nonfinite_in_all_selection_rounds(self):
        for width in (32768, 65536):
            with self.subTest(width=width):
                logits = torch.randn((6, width), device="cuda")
                logits[:, ::13] = -torch.inf
                positions = torch.randperm(width, device="cuda")
                for row, count in enumerate((1, 511, 512, 513, 2000)):
                    picked = positions[:count]
                    logits[row, picked[::3]] = float("nan")
                    logits[row, picked[1::3]] = -float("nan")
                    logits[row, picked[2::3]] = torch.inf
                logits[5].fill_(-torch.inf)
                logits[:5, -1] = -float("nan")
                ends = torch.full((6,), width, device="cuda", dtype=torch.int32)
                self.check(logits, ends)
                out = torch.empty((6, 512), device="cuda", dtype=torch.int32)
                selector.try_select_tokens(logits, ends, out=out, filter_finite=False)
                topk.finish_tokens(logits, ends, out)
                assert_equivalent(self, out, reference(logits, ends), logits, ends)

    def test_tma_tail_with_exact_last_row_storage_and_canaries(self):
        for width in (544, 800):
            with self.subTest(width=width):
                stride = ((width + 255) // 256) * 256
                elements = 2 * stride + width
                storage = torch.full((elements + 32,), 123456.0, device="cuda")
                logits = storage[16:-16].as_strided((3, width), (stride, 1))
                logits.normal_()
                before = storage.clone()
                ends = torch.tensor(
                    [width, 513, width], device="cuda", dtype=torch.int32
                )
                self.check(logits, ends)
                torch.testing.assert_close(storage, before, rtol=0, atol=0)

    def test_non32_last_row_padding_is_checked_before_native_launch(self):
        for width in (545, 801):
            with self.subTest(width=width):
                stride = (width + 255) // 256 * 256
                ends = torch.full((3,), width, device="cuda", dtype=torch.int32)
                storage = torch.randn((16 + 2 * stride + width,), device="cuda")
                logits = storage[16:].as_strided((3, width), (stride, 1))
                self.assertIsNone(selector.try_select_tokens(logits, ends))
                out = torch.empty((3, 512), device="cuda", dtype=torch.int32)
                with self.assertRaises(RuntimeError):
                    selector.rtp_llm_ops.deepselect_fp32(logits, ends, out, True)
                with patch.object(
                    selector.rtp_llm_ops,
                    "deepselect_fp32",
                    wraps=selector.rtp_llm_ops.deepselect_fp32,
                ) as launch:
                    actual = topk.try_select_tokens(logits, ends)
                launch.assert_not_called()
                assert_equivalent(self, actual, reference(logits, ends), logits, ends)
                rounded = (width + 31) // 32 * 32
                padded_storage = torch.full(
                    (16 + 2 * stride + rounded,), torch.inf, device="cuda"
                )
                padded = padded_storage[16:].as_strided((3, width), (stride, 1))
                padded.normal_()
                before = padded_storage.clone()
                self.check(padded, ends)
                torch.testing.assert_close(padded_storage, before, rtol=0, atol=0)

    def test_raw_selection_then_existing_finite_epilogue(self):
        logits = torch.randn((7, 2048), device="cuda")
        ends = torch.tensor(
            [0, 17, 511, 512, 513, 1024, 2048], device="cuda", dtype=torch.int32
        )
        starts = torch.zeros_like(ends)
        logits[:, 0] = float("nan")
        logits[:, 1] = torch.inf
        logits[:, 3] = -torch.inf
        out = torch.empty((7, 512), device="cuda", dtype=torch.int32)
        backend = selector.rtp_llm_ops
        with patch.object(
            backend, "deepselect_fp32", wraps=backend.deepselect_fp32
        ) as select:
            actual = topk.try_select_tokens(
                logits, ends, bounds=(starts, ends), out=out, filter_finite=False
            )
        select.assert_called_once()
        self.assertIs(actual, out)
        torch.testing.assert_close(
            out[1, :17], torch.arange(17, device="cuda", dtype=torch.int32)
        )
        self.assertTrue((out[1, 17:] == -1).all().item())
        topk.finish_tokens(logits, ends, out)
        assert_equivalent(self, out, reference(logits, ends), logits, ends)

    def test_adaptive_dispatch_boundary_preserves_nonfinite_contract(self):
        sm_count = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        width = 8449
        stride = (width + 255) // 256 * 256
        lengths = [-(2**31), -1, 0, 1, 511, 512, 513, 8192, width - 1, width, 2**31 - 1]
        for rows in (sm_count, sm_count + 1):
            with self.subTest(rows=rows, sm_count=sm_count):
                backing = torch.full((rows, stride), torch.inf, device="cuda")
                logits = backing[:, :width]
                logits.normal_()
                logits[:, 0] = float("nan")
                logits[:, 1] = torch.inf
                logits[:, 2] = -torch.inf
                logits[:, 4096] = -float("nan")
                logits[:, 8192] = float("nan")
                logits[:, 8193] = torch.inf
                logits[:, -1] = -torch.inf
                ends = torch.tensor(
                    [lengths[row % len(lengths)] for row in range(rows)],
                    device="cuda",
                    dtype=torch.int32,
                )
                expected = reference(logits, ends)
                out = torch.full(
                    (rows, 512), -(2**31), device="cuda", dtype=torch.int32
                )
                self.assertIs(selector.try_select_tokens(logits, ends, out=out), out)
                assert_equivalent(self, out, expected, logits, ends)
                out.fill_(-(2**31))
                backend = selector.rtp_llm_ops
                with patch.object(
                    backend, "deepselect_fp32", wraps=backend.deepselect_fp32
                ) as launch:
                    selected = topk.try_select_tokens(
                        logits,
                        ends,
                        bounds=(torch.zeros_like(ends), ends),
                        out=out,
                        filter_finite=False,
                    )
                launch.assert_called_once()
                self.assertIs(selected, out)
                topk.finish_tokens(logits, ends, out)
                assert_equivalent(self, out, expected, logits, ends)

    def test_nondefault_stream_preserves_dependency_order(self):
        logits = torch.empty((7, 4352), device="cuda")[:, :4128]
        visible = torch.empty(7, device="cuda", dtype=torch.int32)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            logits.normal_()
            logits[:, 0] = torch.inf
            visible.fill_(4128)
            actual = selector.try_select_tokens(logits, visible)
            snapshot = actual.clone()
        torch.cuda.current_stream().wait_stream(stream)
        assert_equivalent(self, snapshot, reference(logits, visible), logits, visible)

    def test_cuda_graph_replay_uses_new_values_and_bounds(self):
        logits = torch.randn((7, 4352), device="cuda")[:, :4128]
        visible = torch.tensor(
            [0, 17, 511, 512, 513, 1024, 4128], device="cuda", dtype=torch.int32
        )
        out = torch.empty((7, 512), device="cuda", dtype=torch.int32)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                selector.try_select_tokens(logits, visible, out=out)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            selector.try_select_tokens(logits, visible, out=out)
        for lengths in ([2**31 - 1, 1024, 513, 512, 511, 17, -(2**31)], [17] * 7):
            logits.normal_()
            logits[:, 0] = -float("nan")
            logits[:, 1] = torch.inf
            visible.copy_(torch.tensor(lengths, device="cuda"))
            graph.replay()
            assert_equivalent(self, out, reference(logits, visible), logits, visible)

    def test_real_grouped_prefill_shapes(self):
        for rows, width in (
            (4096, 16384),
            (2048, 32768),
            (1024, 65536),
            (1478, 1837),
            (4096, 10004),
        ):
            with self.subTest(rows=rows, width=width):
                stride = (width + 255) // 256 * 256
                logits = torch.randn((rows, stride), device="cuda")[:, :width]
                visible = torch.arange(
                    width - rows + 1, width + 1, device="cuda", dtype=torch.int32
                )
                actual = selector.try_select_tokens(logits, visible)
                sample = torch.linspace(0, rows - 1, 37, device="cuda").long()
                selected_logits, selected_visible = logits[sample], visible[sample]
                assert_equivalent(
                    self,
                    actual[sample],
                    reference(selected_logits, selected_visible),
                    selected_logits,
                    selected_visible,
                )


if __name__ == "__main__":
    unittest.main()
