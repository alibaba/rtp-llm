"""Host dispatch contracts; CUDA checks require explicit separate authorization."""

import contextlib
import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock

import torch


def load():
    path = Path(__file__).resolve().parents[1] / "_v41_candidate_topk.py"
    spec = importlib.util.spec_from_file_location("candidate_topk_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


op = load()


def metadata(shape, dtype, storage, stride=None):
    tensor = mock.Mock(spec=torch.Tensor)
    tensor.layout = torch.strided
    tensor.is_cuda = True
    tensor.dtype = dtype
    tensor.device = torch.device("cuda:0")
    tensor.requires_grad = False
    tensor.shape = shape
    tensor.ndim = len(shape)
    strides = stride or ((shape[1], 1) if len(shape) == 2 else (1,))
    tensor.stride.side_effect = lambda dim: strides[dim]
    tensor.is_contiguous.return_value = True
    tensor.untyped_storage.return_value.data_ptr.return_value = storage
    tensor.data_ptr.return_value = storage
    return tensor


class HostContractTest(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"DSV41_CANDIDATE_NATIVE_TOPK": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.scores = metadata((386, 3945), torch.float32, 4096)
        self.out = metadata((386, 2048), torch.int32, 8192)
        self.scratch = metadata((386,), torch.int32, 12288)

    def test_import_cpu_and_cold_paths_have_no_cuda_compile_or_output_effect(self):
        with mock.patch.object(torch.cuda, "_lazy_init") as init, mock.patch.object(
            op.subprocess, "run"
        ) as compiler, mock.patch.object(op.ctypes, "CDLL") as cdll:
            fresh = load()
            out = torch.full((256, 2048), 123, dtype=torch.int32)
            self.assertIsNone(fresh.try_select(torch.empty(256, 4099), out))
            self.assertTrue(bool((out == 123).all()))
            self.assertFalse(fresh.warmup("cpu"))
            self.assertIsNone(fresh.try_select(self.scores, self.out))
        init.assert_not_called()
        compiler.assert_not_called()
        cdll.assert_not_called()

    def test_gate_covers_mixed_domain_without_shape_whitelist(self):
        shapes = [
            (386, 3945),
            (1664, 4168),
            (914, 6601),
            (1186, 6601),
            (1824, 4168),
            (2538, 2249),
            (932, 8457),
            (482, 6280),
            (1824, 4168),
            (1168, 2632),
            (544, 12424),
            (272, 12424),
            (656, 12745),
            (258, 12745),
            (882, 8457),
            (656, 12745),
            (258, 12745),
            (256, 2049),
            (4096, 16384),
            (8192, 4099),
            (8192, 8192),
            (601, 8193),
            (1, 4099),
            (3, 4099),
            (17, 4099),
            (127, 4099),
            (255, 4099),
        ]
        for m, n in shapes:
            with self.subTest(m=m, n=n):
                self.assertTrue(
                    op.is_supported(
                        metadata((m, n), torch.float32, 1),
                        metadata((m, 2048), torch.int32, 2),
                    )
                )
        for m, n in (
            (1, 257),
            (0, 4099),
            (8193, 4099),
            (257, 32771),
            (64, 65537),
            (256, 2048),
            (256, 16385),
        ):
            self.assertFalse(op._shape_supported(m, n, 2048))
        for k in (512, 1024, 2047, 2049, True):
            self.assertFalse(op._shape_supported(256, 4099, k))

    def test_strides_padding_and_storage_offsets_are_supported(self):
        scores = metadata((386, 3945), torch.float32, 4096, (3951, 1))
        out = metadata((386, 2065), torch.int32, 8192, (2099, 1))
        self.assertTrue(op.is_supported(scores, out, scratch=self.scratch))

    def test_bad_dtype_layout_stride_grad_and_device_are_rejected(self):
        changes = [
            dict(dtype=torch.float16),
            dict(requires_grad=True),
            dict(device=torch.device("cuda:1")),
            dict(shape=(386, 2047)),
            dict(layout=torch.sparse_coo),
        ]
        for change in changes:
            bad = metadata((386, 2048), torch.int32, 8192)
            for name, value in change.items():
                setattr(bad, name, value)
            self.assertFalse(op.is_supported(self.scores, bad))
        for stride in ((2047, 1), (2048, 2), (-2048, 1), (2**62, 1)):
            self.assertFalse(
                op.is_supported(
                    self.scores, metadata((386, 2048), torch.int32, 8192, stride)
                )
            )

    def test_aliases_and_bad_scratch_are_rejected_before_launch(self):
        for bad in (
            metadata((386,), torch.int32, 4096),
            metadata((386,), torch.int32, 8192),
            metadata((385,), torch.int32, 12288),
            metadata((386,), torch.int64, 12288),
        ):
            self.assertFalse(op.is_supported(self.scores, self.out, scratch=bad))
        alias = metadata((386, 2048), torch.int32, 4096)
        self.assertFalse(op.is_supported(self.scores, alias))
        self.scratch.is_contiguous.return_value = False
        self.assertFalse(op.is_supported(self.scores, self.out, scratch=self.scratch))

    def test_max_rows_stride_byte_offset_bound(self):
        rows = 8192
        limit = (2**63 - 1) // rows // 4
        scores = metadata((rows, 8192), torch.float32, 4096, (limit, 1))
        out = metadata((rows, 2048), torch.int32, 8192, (limit, 1))
        scratch = metadata((rows,), torch.int32, 12288)
        self.assertTrue(op.is_supported(scores, out, scratch=scratch))
        scores.stride.side_effect = lambda dim: (limit + 1, 1)[dim]
        self.assertFalse(op.is_supported(scores, out, scratch=scratch))
        scores.stride.side_effect = lambda dim: (8192, 1)[dim]
        out.stride.side_effect = lambda dim: (limit + 1, 1)[dim]
        self.assertFalse(op.is_supported(scores, out, scratch=scratch))

    def test_cold_and_disabled_do_not_allocate_or_compile(self):
        with mock.patch.dict(op._READY, {}, clear=True), mock.patch.object(
            op, "_load"
        ) as build, mock.patch.object(torch, "empty") as allocate, mock.patch.object(
            op, "_execute"
        ) as launch:
            self.assertIsNone(op.try_select(self.scores, self.out))
            with mock.patch.dict(os.environ, {"DSV41_CANDIDATE_NATIVE_TOPK": "0"}):
                self.assertIsNone(op.try_select(self.scores, self.out))
            build.assert_not_called()
            allocate.assert_not_called()
            launch.assert_not_called()

    def test_context_mismatch_does_not_mutate_or_allocate(self):
        lib = mock.Mock()
        lib.v41_candidate_topk_is_current.return_value = 0
        with mock.patch.dict(op._READY, {0: (lib, 123)}, clear=True), mock.patch.object(
            torch.cuda, "device", return_value=contextlib.nullcontext()
        ), mock.patch.object(torch, "empty") as allocate, mock.patch.object(
            op, "_execute"
        ) as launch:
            self.assertIsNone(op.try_select(self.scores, self.out))
            allocate.assert_not_called()
            launch.assert_not_called()

    def test_optional_scratch_is_per_call_and_supplied_scratch_avoids_allocation(self):
        lib = mock.Mock()
        lib.v41_candidate_topk_is_current.return_value = 1
        with mock.patch.dict(op._READY, {0: (lib, 123)}, clear=True), mock.patch.object(
            torch.cuda, "device", return_value=contextlib.nullcontext()
        ), mock.patch.object(
            torch, "empty", side_effect=[self.scratch, object()]
        ) as allocate, mock.patch.object(
            op, "_execute", return_value=self.out
        ) as launch:
            for _ in range(2):
                self.assertIs(op.try_select(self.scores, self.out), self.out)
            self.assertEqual(allocate.call_count, 2)
            self.assertIsNot(
                launch.call_args_list[0].args[3], launch.call_args_list[1].args[3]
            )
            self.assertIs(
                op.try_select(self.scores, self.out, scratch=self.scratch), self.out
            )
            self.assertEqual(allocate.call_count, 2)
        self.assertFalse(any(isinstance(v, torch.Tensor) for v in op._READY.values()))

    def test_launch_uses_current_stream_and_tracks_external_lifetime_on_error(self):
        lib = mock.Mock()
        lib.v41_candidate_topk_launch.return_value = 0
        stream = mock.Mock(cuda_stream=123456)
        with mock.patch.object(torch.cuda, "current_stream", return_value=stream):
            self.assertIs(
                op._execute(lib, self.scores, self.out, self.scratch), self.out
            )
            self.assertEqual(lib.v41_candidate_topk_launch.call_args.args[-1], 123456)
            for tensor in (self.scores, self.out, self.scratch):
                tensor.record_stream.assert_called_with(stream)
            lib.v41_candidate_topk_launch.return_value = 719
            with self.assertRaises(RuntimeError):
                op._execute(lib, self.scores, self.out, self.scratch)
            for tensor in (self.scores, self.out, self.scratch):
                self.assertEqual(tensor.record_stream.call_count, 2)

    def test_warmup_rejects_capture_before_loading(self):
        with mock.patch.object(torch, "__version__", "2.11.0+cu130"), mock.patch.object(
            torch.cuda, "device", return_value=contextlib.nullcontext()
        ), mock.patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=True
        ), mock.patch.object(
            op, "_load"
        ) as build:
            with self.assertRaises(RuntimeError):
                op.warmup("cuda:0")
            build.assert_not_called()

    def test_warmup_primes_four_variants_once_and_publishes_after_success(self):
        lib = mock.Mock()
        lib.v41_candidate_topk_is_current.return_value = 1

        def prepare(pointer):
            pointer._obj.value = 123
            return 0

        lib.v41_candidate_topk_prepare.side_effect = prepare
        prop = mock.Mock(major=10, minor=0, multi_processor_count=80)
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(op._READY, {}, clear=True))
            stack.enter_context(mock.patch.object(torch, "__version__", "2.11.0+cu130"))
            stack.enter_context(
                mock.patch.object(
                    torch.cuda, "device", return_value=contextlib.nullcontext()
                )
            )
            stack.enter_context(
                mock.patch.object(
                    torch.cuda, "is_current_stream_capturing", return_value=False
                )
            )
            stack.enter_context(
                mock.patch.object(torch.cuda, "current_device", return_value=0)
            )
            stack.enter_context(
                mock.patch.object(
                    torch.cuda, "get_device_properties", return_value=prop
                )
            )
            stack.enter_context(mock.patch.object(torch.cuda, "current_stream"))
            build = stack.enter_context(
                mock.patch.object(op, "_load", return_value=lib)
            )
            allocate = stack.enter_context(
                mock.patch.object(torch, "full", return_value=self.scores)
            )
            stack.enter_context(
                mock.patch.object(torch, "empty", return_value=self.out)
            )
            execute = stack.enter_context(
                mock.patch.object(op, "_execute", side_effect=RuntimeError("failed"))
            )
            with self.assertRaises(RuntimeError):
                op.warmup("cuda:0")
            self.assertEqual(op._READY, {})
            allocate.reset_mock()
            execute.side_effect = None
            self.assertTrue(op.warmup("cuda:0"))
            self.assertEqual(
                [call.args[0][1] for call in allocate.call_args_list],
                [4096, 4099, 8196, 8199],
            )
            build.reset_mock()
            self.assertTrue(op.warmup("cuda:0"))
            build.assert_not_called()


@unittest.skipUnless(
    os.environ.get("RTP_CANDIDATE_TOPK_GPU_TEST") == "1",
    "GPU execution requires explicit authorization",
)
class CUDAContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = mock.patch.dict(os.environ, {"DSV41_CANDIDATE_NATIVE_TOPK": "1"})
        cls.env.start()
        if not op.warmup("cuda:0"):
            cls.env.stop()
            raise AssertionError("Candidate startup warmup unavailable")

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()

    def check(self, scores, out=None, scratch=None):
        if out is None:
            out = torch.empty(
                (scores.shape[0], 2065), device=scores.device, dtype=torch.int32
            )
        values, ids = torch.topk(scores, 2048, dim=-1, largest=True, sorted=False)
        expected = torch.nn.functional.pad(
            torch.where(values > -float("inf"), ids, -1).int(),
            (0, out.shape[1] - 2048),
            value=-1,
        )
        self.assertIs(op.try_select(scores, out, scratch=scratch), out)
        self.assertTrue(torch.equal(out.sort(-1).values, expected.sort(-1).values))
        return out

    def test_exact_exception_boundaries_and_saturated_negative(self):
        for n in (4099, 8193):
            out = torch.empty((256, 2065), dtype=torch.int32, device="cuda")
            status = torch.empty((256,), dtype=torch.int32, device="cuda")
            for finite, nans in (
                (2047, 0),
                (2048, 0),
                (2049, 0),
                (2047, 1),
                (2048, 1),
                (1, 2047),
                (1, 2048),
                (1, 2049),
                (0, 0),
                (0, 2049),
            ):
                for value in (
                    -torch.finfo(torch.float32).max,
                    -1e30,
                    -65536.0,
                    0.0,
                    -0.0,
                    1.0,
                    float("inf"),
                ):
                    scores = torch.full((256, n), -float("inf"), device="cuda")
                    scores[:, :finite] = value
                    scores[:, finite : finite + nans] = float("nan")
                    self.check(scores, out, status)
                    self.assertFalse(bool((status == 0).any()))
                    if finite + nans > 2048:
                        self.assertFalse(bool((status == 7).any()))

    def test_reuse_odd_strides_midpoints_and_current_stream(self):
        stream = torch.cuda.Stream()
        for n in (4096, 4099, 8192, 8193, 16384):
            storage = torch.randn((256, n + 3), device="cuda")
            scores = storage[:, 1 : n + 1]
            out = torch.empty((256, 2070), device="cuda", dtype=torch.int32)[:, 1:2066]
            status = torch.empty((256,), device="cuda", dtype=torch.int32)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                actual = op.try_select(scores, out, scratch=status)
            torch.cuda.current_stream().wait_stream(stream)
            self.assertIs(actual, out)
            self.check(scores, out, status)
            scores.fill_(-0.500244140625)
            self.check(scores, out, status)

    def test_graph_capture_replay_with_internal_and_external_scratch(self):
        for supplied in (False, True):
            scores = torch.randn((256, 8193), device="cuda")
            out = torch.empty((256, 2048), device="cuda", dtype=torch.int32)
            scratch = (
                torch.empty((256,), device="cuda", dtype=torch.int32)
                if supplied
                else None
            )
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                op.try_select(scores, out, scratch=scratch)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                self.assertIs(op.try_select(scores, out, scratch=scratch), out)
            for _ in range(3):
                scores.normal_()
                graph.replay()
                values, ids = torch.topk(scores, 2048, dim=-1, sorted=False)
                self.assertTrue(
                    torch.equal(out.sort(-1).values, ids.int().sort(-1).values)
                )


if __name__ == "__main__":
    unittest.main()
