import logging
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    _is_fmha_impl_disabled,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl import py_flash_attn_v4
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flash_attn_v4 import (
    FlashAttn4TargetVerifyOp,
    FlashAttn4TargetVerifyParams,
    get_fa4_target_verify_num_splits,
)
from rtp_llm.ops import KvCacheDataType


class _CudaTensorStub:
    is_cuda = True
    device = torch.device("cuda", 0)


def _make_config(**overrides):
    values = dict(
        dtype=torch.bfloat16,
        kv_cache_dtype=KvCacheDataType.BASE,
        size_per_head=256,
        kernel_tokens_per_block=64,
        is_causal=True,
        need_rope_kv_cache=True,
        head_num=12,
        kv_head_num=2,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _make_inputs(**overrides):
    values = dict(
        is_target_verify=True,
        is_prefill=True,
        is_cuda_graph=True,
        input_lengths=torch.tensor([5, 5], dtype=torch.int32),
        input_lengths_device=_CudaTensorStub(),
        cu_kv_seqlens_device=_CudaTensorStub(),
        kv_cache_kernel_block_id_device=_CudaTensorStub(),
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


class TestFlashAttn4TargetVerify(unittest.TestCase):
    def test_split_count_tracks_sm_count_and_batch(self):
        h20 = [
            get_fa4_target_verify_num_splits(
                sm_count=78,
                batch_size=batch,
                query_len=5,
                num_q_heads=12,
                num_kv_heads=2,
                max_kv_len=23700,
            )
            for batch in range(1, 5)
        ]
        h200 = [
            get_fa4_target_verify_num_splits(
                sm_count=132,
                batch_size=batch,
                query_len=5,
                num_q_heads=12,
                num_kv_heads=2,
                max_kv_len=23700,
            )
            for batch in range(1, 5)
        ]
        self.assertEqual(h20, [39, 19, 13, 9])
        self.assertEqual(h200, [66, 33, 22, 16])

    def test_split_count_is_bounded_by_available_kv_tiles(self):
        self.assertEqual(
            get_fa4_target_verify_num_splits(
                sm_count=132,
                batch_size=1,
                query_len=5,
                num_q_heads=12,
                num_kv_heads=2,
                max_kv_len=64,
            ),
            2,
        )

    def test_split_count_respects_fa4_limit(self):
        self.assertEqual(
            get_fa4_target_verify_num_splits(
                sm_count=1024,
                batch_size=1,
                query_len=1,
                num_q_heads=1,
                num_kv_heads=1,
                max_kv_len=32768,
            ),
            128,
        )

    def test_split_count_rejects_invalid_inputs(self):
        valid_inputs = {
            "sm_count": 132,
            "batch_size": 2,
            "query_len": 5,
            "num_q_heads": 12,
            "num_kv_heads": 2,
            "max_kv_len": 320,
        }
        for name in valid_inputs:
            invalid_inputs = dict(valid_inputs)
            invalid_inputs[name] = 0
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "must all be positive"
            ):
                get_fa4_target_verify_num_splits(**invalid_inputs)
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            get_fa4_target_verify_num_splits(**{**valid_inputs, "num_q_heads": 11})

    def test_support_is_limited_to_tuned_graph_shape(self):
        with mock.patch.object(
            py_flash_attn_v4, "is_sm90", return_value=True
        ), mock.patch.object(py_flash_attn_v4, "_fa4_is_available", return_value=True):
            self.assertTrue(
                FlashAttn4TargetVerifyOp.support(_make_config(), _make_inputs())
            )
            self.assertFalse(
                FlashAttn4TargetVerifyOp.support(
                    _make_config(kv_cache_dtype=KvCacheDataType.FP8),
                    _make_inputs(),
                )
            )
            self.assertFalse(
                FlashAttn4TargetVerifyOp.support(
                    _make_config(), _make_inputs(is_cuda_graph=False)
                )
            )
            self.assertFalse(
                FlashAttn4TargetVerifyOp.support(
                    _make_config(need_rope_kv_cache=False), _make_inputs()
                )
            )
            self.assertFalse(
                FlashAttn4TargetVerifyOp.support(
                    _make_config(),
                    _make_inputs(input_lengths=torch.tensor([5, 4], dtype=torch.int32)),
                )
            )

    def test_support_falls_back_when_fa4_is_unavailable(self):
        with mock.patch.object(
            py_flash_attn_v4, "is_sm90", return_value=True
        ), mock.patch.object(py_flash_attn_v4, "_fa4_is_available", return_value=False):
            self.assertFalse(
                FlashAttn4TargetVerifyOp.support(_make_config(), _make_inputs())
            )

    def test_dependency_check_rejects_version_above_supported_range(self):
        installed_versions = {
            "nvidia-cutlass-dsl": "4.6.0",
            "apache-tvm-ffi": "0.1.13",
            "quack-kernels": "0.5.0",
            "torch-c-dlpack-ext": "0.1.5",
        }
        with mock.patch.object(
            py_flash_attn_v4,
            "version",
            side_effect=installed_versions.__getitem__,
        ):
            with self.assertRaisesRegex(RuntimeError, "nvidia-cutlass-dsl"):
                py_flash_attn_v4._check_fa4_dependencies()

    def test_dependency_check_rejects_missing_dependency(self):
        with mock.patch.object(
            py_flash_attn_v4,
            "version",
            side_effect=py_flash_attn_v4.PackageNotFoundError("nvidia-cutlass-dsl"),
        ):
            with self.assertRaisesRegex(RuntimeError, "nvidia-cutlass-dsl"):
                py_flash_attn_v4._check_fa4_dependencies()

    def test_dependency_check_rejects_invalid_version(self):
        installed_versions = {
            "nvidia-cutlass-dsl": "not-a-version",
            "apache-tvm-ffi": "0.1.13",
            "quack-kernels": "0.5.0",
            "torch-c-dlpack-ext": "0.1.5",
        }
        with mock.patch.object(
            py_flash_attn_v4,
            "version",
            side_effect=installed_versions.__getitem__,
        ):
            with self.assertRaisesRegex(RuntimeError, "invalid nvidia-cutlass-dsl"):
                py_flash_attn_v4._check_fa4_dependencies()

    def test_dependency_check_accepts_supported_versions(self):
        installed_versions = {
            "nvidia-cutlass-dsl": "4.5.3",
            "apache-tvm-ffi": "0.1.13",
            "quack-kernels": "0.5.1",
            "torch-c-dlpack-ext": "0.1.5",
        }
        self.assertEqual(
            set(installed_versions), set(py_flash_attn_v4._FA4_DEPENDENCY_SPECS)
        )
        with mock.patch.object(
            py_flash_attn_v4,
            "version",
            side_effect=installed_versions.__getitem__,
        ):
            py_flash_attn_v4._check_fa4_dependencies()

    def test_dependency_specs_match_cuda12_9_requirements(self):
        requirements_path = (
            Path(os.environ["TEST_SRCDIR"])
            / "rtp_deps"
            / "requirements_torch_gpu_cuda12_9.txt"
        )
        production_specs = {
            canonicalize_name(name): specifier
            for name, specifier in py_flash_attn_v4._FA4_DEPENDENCY_SPECS.items()
        }
        declared_specs = {}
        for raw_line in requirements_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith(("#", "-", "http://", "https://")):
                continue
            try:
                requirement = Requirement(line)
            except InvalidRequirement:
                continue
            package_name = canonicalize_name(requirement.name)
            if package_name in production_specs:
                declared_specs[package_name] = requirement.specifier

        self.assertEqual(set(declared_specs), set(production_specs))
        for package_name, production_spec in production_specs.items():
            with self.subTest(package=package_name):
                self.assertEqual(declared_specs[package_name], production_spec)

    def test_invalid_log_level_is_reported_and_vendor_handler_is_removed(self):
        default_handler = logging.StreamHandler()
        vendor_logger = logging.Logger("flash_attn")
        vendor_logger.addHandler(default_handler)
        fa_logging = types.ModuleType("test_vendor.cute.fa_logging")

        def set_fa_log_level(level):
            if level == 0 and default_handler in vendor_logger.handlers:
                vendor_logger.removeHandler(default_handler)

        fa_logging.set_fa_log_level = mock.Mock(side_effect=set_fa_log_level)
        fa_logging.get_fa_log_level = mock.Mock(return_value=0)
        with mock.patch.object(
            py_flash_attn_v4.logging,
            "getLogger",
            return_value=vendor_logger,
        ), mock.patch.dict(os.environ, {"FA_LOG_LEVEL": "verbose"}), self.assertLogs(
            py_flash_attn_v4.logger, level="WARNING"
        ) as captured:
            py_flash_attn_v4._configure_fa4_logging(fa_logging)

        self.assertEqual(
            fa_logging.set_fa_log_level.call_args_list,
            [mock.call(0), mock.call(0)],
        )
        self.assertNotIn(default_handler, vendor_logger.handlers)
        self.assertEqual(vendor_logger.level, logging.NOTSET)
        self.assertTrue(vendor_logger.propagate)
        self.assertIn("invalid FA_LOG_LEVEL", captured.output[0])

    def test_numeric_log_level_is_clamped(self):
        for raw_level, expected in (("-1", 0), ("4", 3)):
            with self.subTest(raw_level=raw_level), mock.patch.dict(
                os.environ, {"FA_LOG_LEVEL": raw_level}
            ), self.assertLogs(py_flash_attn_v4.logger, level="WARNING"):
                self.assertEqual(py_flash_attn_v4._get_fa4_log_level(), expected)

    def test_vendor_host_log_levels_map_to_rtp_logging_levels(self):
        app_handler = logging.StreamHandler()
        initial_vendor_handler = logging.StreamHandler()
        replacement_vendor_handler = logging.StreamHandler()
        vendor_logger = logging.Logger("flash_attn")
        vendor_logger.addHandler(app_handler)
        vendor_logger.addHandler(initial_vendor_handler)
        vendor_logger.info = mock.Mock()
        vendor_logger.debug = mock.Mock()

        def set_fa_log_level(level):
            if level == 0:
                vendor_logger.removeHandler(initial_vendor_handler)
            else:
                vendor_logger.addHandler(replacement_vendor_handler)

        fa_logging = types.ModuleType("test_vendor.cute.fa_logging")
        fa_logging.set_fa_log_level = mock.Mock(side_effect=set_fa_log_level)
        fa_logging.get_fa_log_level = mock.Mock(return_value=3)
        interface = types.ModuleType("test_vendor.cute.interface")
        with mock.patch.object(
            py_flash_attn_v4.logging,
            "getLogger",
            return_value=vendor_logger,
        ), mock.patch.dict(os.environ, {"FA_LOG_LEVEL": "max"}), mock.patch.dict(
            sys.modules, {interface.__name__: interface}
        ):
            py_flash_attn_v4._configure_fa4_logging(fa_logging)

        self.assertEqual(
            fa_logging.set_fa_log_level.call_args_list,
            [mock.call(0), mock.call(3)],
        )
        self.assertIn(app_handler, vendor_logger.handlers)
        self.assertNotIn(initial_vendor_handler, vendor_logger.handlers)
        self.assertNotIn(replacement_vendor_handler, vendor_logger.handlers)
        self.assertEqual(vendor_logger.level, logging.NOTSET)
        self.assertTrue(vendor_logger.propagate)
        fa_logging.fa_log(1, "host summary")
        fa_logging.fa_log(2, "kernel detail")
        fa_logging.fa_log(3, "maximum detail")
        vendor_logger.info.assert_called_once_with("host summary")
        vendor_logger.debug.assert_has_calls(
            [mock.call("kernel detail"), mock.call("maximum detail")]
        )
        self.assertIs(interface.fa_log, fa_logging.fa_log)

    def test_fallback_logs_loader_failure(self):
        py_flash_attn_v4._fa4_is_available.cache_clear()
        try:
            with mock.patch.object(
                py_flash_attn_v4,
                "_load_fa4_forward",
                side_effect=RuntimeError("missing FA4 dependency"),
            ), self.assertLogs(py_flash_attn_v4.logger, level="ERROR") as captured:
                self.assertFalse(py_flash_attn_v4._fa4_is_available())
            self.assertIn("missing FA4 dependency", captured.output[0])
        finally:
            py_flash_attn_v4._fa4_is_available.cache_clear()

    def test_rollback_flags_control_backend(self):
        config = types.SimpleNamespace(
            enable_paged_open_source_fmha=True,
            enable_fa4_target_verify=True,
            enable_flashinfer_fa2_target_verify=True,
            disable_flashinfer_native=False,
        )
        self.assertFalse(_is_fmha_impl_disabled("FlashAttn4TargetVerifyImpl", config))
        config.enable_fa4_target_verify = False
        self.assertTrue(_is_fmha_impl_disabled("FlashAttn4TargetVerifyImpl", config))
        config.enable_fa4_target_verify = True
        config.enable_paged_open_source_fmha = False
        self.assertTrue(_is_fmha_impl_disabled("FlashAttn4TargetVerifyImpl", config))

        for implementation_name in (
            "PyFlashinferFa2TargetVerifyImpl",
            "PyFlashinferMropeTargetVerifyImpl",
        ):
            config.enable_flashinfer_fa2_target_verify = True
            config.disable_flashinfer_native = False
            self.assertFalse(_is_fmha_impl_disabled(implementation_name, config))
            config.enable_flashinfer_fa2_target_verify = False
            self.assertTrue(_is_fmha_impl_disabled(implementation_name, config))
            config.enable_flashinfer_fa2_target_verify = True
            config.disable_flashinfer_native = True
            self.assertTrue(_is_fmha_impl_disabled(implementation_name, config))

    def test_backend_precedes_flashinfer_fa2_fallbacks(self):
        from rtp_llm.models_py.modules.factory.attention import PREFILL_MHA_IMPS

        implementation_names = [
            implementation.__name__ for implementation in PREFILL_MHA_IMPS
        ]
        fa4_index = implementation_names.index("FlashAttn4TargetVerifyImpl")
        self.assertLess(
            fa4_index,
            implementation_names.index("PyFlashinferFa2TargetVerifyImpl"),
        )
        self.assertLess(
            fa4_index,
            implementation_names.index("PyFlashinferMropeTargetVerifyImpl"),
        )

    def test_prepare_cuda_graph_refreshes_kv_lengths_in_place(self):
        cu_kv = torch.tensor([0, 12, 31], dtype=torch.int32)
        kv_lengths = torch.empty(2, dtype=torch.int32)
        query_lengths = torch.tensor([5, 5], dtype=torch.int32)
        page_table = torch.zeros(2, 1, dtype=torch.int32)
        params = FlashAttn4TargetVerifyParams(
            batch_size=2,
            query_len=5,
            max_kv_len=20,
            num_splits=1,
            query_lengths=query_lengths,
            kv_lengths=kv_lengths,
            cu_kv_seqlens=cu_kv,
            page_table=page_table,
        )
        inputs = types.SimpleNamespace(
            input_lengths=query_lengths,
            prefix_lengths=torch.tensor([7, 14], dtype=torch.int32),
            input_lengths_device=query_lengths,
            cu_kv_seqlens_device=cu_kv,
            kv_cache_kernel_block_id_device=page_table,
        )
        op = object.__new__(FlashAttn4TargetVerifyOp)
        op.page_size = 64
        op.prepare_cuda_graph(params, inputs)
        self.assertEqual(kv_lengths.tolist(), [12, 19])

    def test_prepare_cuda_graph_allows_shorter_kv_and_padded_batch(self):
        cu_kv = torch.tensor([0, 12, 12], dtype=torch.int32)
        query_lengths = torch.tensor([5, 0], dtype=torch.int32)
        page_table = torch.zeros(2, 1, dtype=torch.int32)
        params = FlashAttn4TargetVerifyParams(
            batch_size=2,
            query_len=5,
            max_kv_len=64,
            num_splits=1,
            query_lengths=query_lengths,
            kv_lengths=torch.empty(2, dtype=torch.int32),
            cu_kv_seqlens=cu_kv,
            page_table=page_table,
        )
        inputs = types.SimpleNamespace(
            input_lengths=query_lengths,
            prefix_lengths=torch.tensor([7, 0], dtype=torch.int32),
            input_lengths_device=query_lengths,
            cu_kv_seqlens_device=cu_kv,
            kv_cache_kernel_block_id_device=page_table,
        )
        op = object.__new__(FlashAttn4TargetVerifyOp)
        op.page_size = 64
        op.prepare_cuda_graph(params, inputs)
        self.assertEqual(params.kv_lengths.tolist(), [12, 0])

    def test_prepare_cuda_graph_rejects_kv_above_capture_bound(self):
        cu_kv = torch.tensor([0, 21], dtype=torch.int32)
        query_lengths = torch.tensor([5], dtype=torch.int32)
        page_table = torch.zeros(1, 1, dtype=torch.int32)
        params = FlashAttn4TargetVerifyParams(
            batch_size=1,
            query_len=5,
            max_kv_len=20,
            num_splits=1,
            query_lengths=query_lengths,
            kv_lengths=torch.empty(1, dtype=torch.int32),
            cu_kv_seqlens=cu_kv,
            page_table=page_table,
        )
        inputs = types.SimpleNamespace(
            input_lengths=query_lengths,
            prefix_lengths=torch.tensor([16], dtype=torch.int32),
            input_lengths_device=query_lengths,
            cu_kv_seqlens_device=cu_kv,
            kv_cache_kernel_block_id_device=page_table,
        )
        op = object.__new__(FlashAttn4TargetVerifyOp)
        op.page_size = 64
        with self.assertRaisesRegex(
            RuntimeError,
            "capture max_kv_len=20, replay max_kv_len=21",
        ):
            op.prepare_cuda_graph(params, inputs)

    def test_prepare_cuda_graph_rejects_num_splits_above_capture_bound(self):
        cu_kv = torch.tensor([0, 64], dtype=torch.int32)
        query_lengths = torch.tensor([5], dtype=torch.int32)
        page_table = torch.zeros(1, 1, dtype=torch.int32)
        params = FlashAttn4TargetVerifyParams(
            batch_size=1,
            query_len=5,
            max_kv_len=64,
            num_splits=3,
            query_lengths=query_lengths,
            kv_lengths=torch.empty(1, dtype=torch.int32),
            cu_kv_seqlens=cu_kv,
            page_table=page_table,
        )
        inputs = types.SimpleNamespace(
            input_lengths=query_lengths,
            prefix_lengths=torch.tensor([59], dtype=torch.int32),
            input_lengths_device=query_lengths,
            cu_kv_seqlens_device=cu_kv,
            kv_cache_kernel_block_id_device=page_table,
        )
        op = object.__new__(FlashAttn4TargetVerifyOp)
        op.page_size = 64
        with self.assertRaisesRegex(
            RuntimeError,
            "capture num_splits=3, capture max_kv_len=64",
        ):
            op.prepare_cuda_graph(params, inputs)

    def test_prepare_cuda_graph_rejects_page_table_overflow(self):
        cu_kv = torch.tensor([0, 65], dtype=torch.int32)
        query_lengths = torch.tensor([5], dtype=torch.int32)
        page_table = torch.zeros(1, 1, dtype=torch.int32)
        params = FlashAttn4TargetVerifyParams(
            batch_size=1,
            query_len=5,
            max_kv_len=128,
            num_splits=1,
            query_lengths=query_lengths,
            kv_lengths=torch.empty(1, dtype=torch.int32),
            cu_kv_seqlens=cu_kv,
            page_table=page_table,
        )
        inputs = types.SimpleNamespace(
            input_lengths=query_lengths,
            prefix_lengths=torch.tensor([60], dtype=torch.int32),
            input_lengths_device=query_lengths,
            cu_kv_seqlens_device=cu_kv,
            kv_cache_kernel_block_id_device=page_table,
        )
        op = object.__new__(FlashAttn4TargetVerifyOp)
        op.page_size = 64
        with self.assertRaisesRegex(
            RuntimeError,
            "replay max_kv_len=65, page_table_kv_capacity=64",
        ):
            op.prepare_cuda_graph(params, inputs)

    def test_prepare_cuda_graph_rejects_changed_query_length(self):
        cu_kv = torch.tensor([0, 12, 31], dtype=torch.int32)
        query_lengths = torch.tensor([5, 4], dtype=torch.int32)
        page_table = torch.zeros(2, 1, dtype=torch.int32)
        params = FlashAttn4TargetVerifyParams(
            batch_size=2,
            query_len=5,
            max_kv_len=32,
            num_splits=1,
            query_lengths=query_lengths,
            kv_lengths=torch.empty(2, dtype=torch.int32),
            cu_kv_seqlens=cu_kv,
            page_table=page_table,
        )
        inputs = types.SimpleNamespace(
            input_lengths=query_lengths,
            prefix_lengths=torch.tensor([7, 15], dtype=torch.int32),
            input_lengths_device=query_lengths,
            cu_kv_seqlens_device=cu_kv,
            kv_cache_kernel_block_id_device=page_table,
        )
        op = object.__new__(FlashAttn4TargetVerifyOp)
        op.page_size = 64
        with self.assertRaisesRegex(
            RuntimeError,
            r"capture query_len=5, replay query_lengths=\[5, 4\]",
        ):
            op.prepare_cuda_graph(params, inputs)

    def test_prepare_cuda_graph_rejects_changed_batch_size(self):
        cu_kv = torch.tensor([0, 12], dtype=torch.int32)
        query_lengths = torch.tensor([5], dtype=torch.int32)
        page_table = torch.zeros(1, 1, dtype=torch.int32)
        params = FlashAttn4TargetVerifyParams(
            batch_size=2,
            query_len=5,
            max_kv_len=32,
            num_splits=1,
            query_lengths=torch.tensor([5, 5], dtype=torch.int32),
            kv_lengths=torch.empty(2, dtype=torch.int32),
            cu_kv_seqlens=torch.tensor([0, 12, 24], dtype=torch.int32),
            page_table=torch.zeros(2, 1, dtype=torch.int32),
        )
        inputs = types.SimpleNamespace(
            input_lengths=query_lengths,
            prefix_lengths=torch.tensor([7], dtype=torch.int32),
            input_lengths_device=query_lengths,
            cu_kv_seqlens_device=cu_kv,
            kv_cache_kernel_block_id_device=page_table,
        )
        op = object.__new__(FlashAttn4TargetVerifyOp)
        op.page_size = 64
        with self.assertRaisesRegex(
            RuntimeError,
            "capture batch_size=2, replay batch_size=1",
        ):
            op.prepare_cuda_graph(params, inputs)

    def test_prepare_cuda_graph_rejects_replaced_metadata_buffer(self):
        cu_kv = torch.tensor([0, 12], dtype=torch.int32)
        query_lengths = torch.tensor([5], dtype=torch.int32)
        page_table = torch.zeros(1, 1, dtype=torch.int32)
        params = FlashAttn4TargetVerifyParams(
            batch_size=1,
            query_len=5,
            max_kv_len=32,
            num_splits=1,
            query_lengths=query_lengths,
            kv_lengths=torch.empty(1, dtype=torch.int32),
            cu_kv_seqlens=cu_kv,
            page_table=page_table,
        )
        inputs = types.SimpleNamespace(
            input_lengths=query_lengths,
            prefix_lengths=torch.tensor([7], dtype=torch.int32),
            input_lengths_device=query_lengths.clone(),
            cu_kv_seqlens_device=cu_kv,
            kv_cache_kernel_block_id_device=page_table,
        )
        op = object.__new__(FlashAttn4TargetVerifyOp)
        op.page_size = 64
        with self.assertRaisesRegex(RuntimeError, "buffer=query_lengths"):
            op.prepare_cuda_graph(params, inputs)

    def test_forward_uses_zero_copy_rtp_kv_views_and_tuned_config(self):
        captured = {}

        def fake_forward(query, key, value, **kwargs):
            captured.update(query=query, key=key, value=value, kwargs=kwargs)
            return torch.zeros_like(query), None, None, None

        op = object.__new__(FlashAttn4TargetVerifyOp)
        op.head_dim = 256
        op.head_num = 12
        op.kv_head_num = 2
        op.page_size = 64
        op.softmax_scale = 256**-0.5
        op._forward = fake_forward

        query = torch.zeros(5, 12, 256, dtype=torch.bfloat16)
        combined_cache = torch.zeros(3, 2, 2, 64, 256, dtype=torch.bfloat16)
        params = FlashAttn4TargetVerifyParams(
            batch_size=1,
            query_len=5,
            max_kv_len=192,
            num_splits=39,
            query_lengths=torch.tensor([5], dtype=torch.int32),
            kv_lengths=torch.tensor([192], dtype=torch.int32),
            cu_kv_seqlens=torch.tensor([0, 192], dtype=torch.int32),
            page_table=torch.arange(3, dtype=torch.int32).view(1, 3),
        )
        output = op.forward(
            query,
            types.SimpleNamespace(kv_cache_base=combined_cache),
            params,
        )

        self.assertEqual(output.shape, query.shape)
        self.assertEqual(captured["query"].shape, (1, 5, 12, 256))
        self.assertEqual(captured["key"].shape, (3, 64, 2, 256))
        self.assertEqual(
            captured["key"].untyped_storage().data_ptr(),
            combined_cache.untyped_storage().data_ptr(),
        )
        self.assertEqual(captured["kwargs"]["tile_mn"], (64, 32))
        self.assertEqual(captured["kwargs"]["num_splits"], 39)
        self.assertTrue(captured["kwargs"]["pack_gqa"])
        self.assertTrue(captured["kwargs"]["mma_pv_is_rs"])


class TestFlashAttn4CudaGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            raise unittest.SkipTest("FA4 target-verify integration test requires SM9x")
        try:
            cls.fa4_forward = py_flash_attn_v4._load_fa4_forward()
        except Exception as error:
            raise AssertionError(
                "FA4 dependencies and backend must load on an SM9x test worker"
            ) from error

    def test_loader_smoke(self):
        self.assertTrue(callable(self.fa4_forward))

    def test_real_vendor_logging_is_host_integrated(self):
        from rtp_llm.third_party.vllm_flash_attention.cute import fa_logging, interface

        vendor_logger = logging.getLogger("flash_attn")
        self.assertTrue(
            all(
                isinstance(handler, logging.NullHandler)
                for handler in vendor_logger.handlers
            )
        )
        self.assertEqual(vendor_logger.level, logging.NOTSET)
        self.assertTrue(vendor_logger.propagate)
        self.assertIs(interface.fa_log, fa_logging.fa_log)

    @staticmethod
    def _reference(
        query,
        combined_cache,
        page_table,
        kv_lengths,
        query_len=5,
        num_q_heads=12,
        num_kv_heads=2,
        page_size=64,
    ):
        if isinstance(kv_lengths, int):
            kv_lengths = [kv_lengths]
        dense_query = query.reshape(-1, query_len, num_q_heads, 256)
        outputs = []
        for batch_index, kv_len in enumerate(kv_lengths):
            num_pages = (kv_len + page_size - 1) // page_size
            page_ids = page_table[batch_index, :num_pages].to(torch.long)
            key = (
                combined_cache[page_ids, 0]
                .permute(0, 2, 1, 3)
                .reshape(-1, num_kv_heads, 256)[:kv_len]
            )
            value = (
                combined_cache[page_ids, 1]
                .permute(0, 2, 1, 3)
                .reshape(-1, num_kv_heads, 256)[:kv_len]
            )
            group_size = num_q_heads // num_kv_heads
            key = key.repeat_interleave(group_size, dim=1)
            value = value.repeat_interleave(group_size, dim=1)

            scores = torch.einsum(
                "qhd,khd->hqk", dense_query[batch_index].float(), key.float()
            ) * (256**-0.5)
            query_positions = torch.arange(query_len, device=query.device)
            key_positions = torch.arange(kv_len, device=query.device)
            causal_mask = key_positions[None, :] <= (
                kv_len - query_len + query_positions[:, None]
            )
            scores.masked_fill_(~causal_mask[None, :, :], float("-inf"))
            probabilities = torch.softmax(scores, dim=-1)
            outputs.append(torch.einsum("hqk,khd->qhd", probabilities, value.float()))
        return torch.stack(outputs).reshape_as(query)

    def test_capture_batches_use_runtime_sm_count(self):
        device = torch.device("cuda")
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        query_len = 5
        kv_len = 23700
        pages_per_sequence = (kv_len + 63) // 64
        config = _make_config(softmax_extra_scale=1.0, q_scaling=1.0)
        observed_splits = []

        for batch_size in range(1, 5):
            input_lengths = torch.full((batch_size,), query_len, dtype=torch.int32)
            prefix_lengths = torch.full(
                (batch_size,), kv_len - query_len, dtype=torch.int32
            )
            inputs = types.SimpleNamespace(
                input_lengths=input_lengths,
                prefix_lengths=prefix_lengths,
                input_lengths_device=input_lengths.to(device),
                cu_kv_seqlens_device=(
                    torch.arange(batch_size + 1, dtype=torch.int32, device=device)
                    * kv_len
                ),
                kv_cache_kernel_block_id_device=torch.zeros(
                    batch_size,
                    pages_per_sequence,
                    dtype=torch.int32,
                    device=device,
                ),
            )
            params = FlashAttn4TargetVerifyOp(config).prepare(inputs)
            expected = get_fa4_target_verify_num_splits(
                sm_count=sm_count,
                batch_size=batch_size,
                query_len=query_len,
                num_q_heads=12,
                num_kv_heads=2,
                max_kv_len=kv_len,
            )
            self.assertEqual(params.num_splits, expected)
            observed_splits.append(params.num_splits)

        self.assertEqual(observed_splits, sorted(observed_splits, reverse=True))

    def test_paged_bf16_cuda_graph_replay_updates_kv_length(self):
        torch.manual_seed(20260731)
        device = torch.device("cuda")
        batch_size = 2
        query_len = 5
        capture_max_kv_len = 320
        initial_kv_lengths = [257, 193]
        replay_kv_lengths = [301, 225]
        page_table = torch.tensor(
            [[4, 1, 7, 0, 5], [2, 8, 6, 9, 3]],
            dtype=torch.int32,
            device=device,
        )
        combined_cache = torch.randn(
            10, 2, 2, 64, 256, dtype=torch.bfloat16, device=device
        )
        query = torch.randn(
            batch_size * query_len,
            12,
            256,
            dtype=torch.bfloat16,
            device=device,
        )
        cu_kv_seqlens = torch.tensor(
            [0, initial_kv_lengths[0], sum(initial_kv_lengths)],
            dtype=torch.int32,
            device=device,
        )
        input_lengths = torch.full((batch_size,), query_len, dtype=torch.int32)
        inputs = types.SimpleNamespace(
            input_lengths=input_lengths,
            prefix_lengths=torch.full(
                (batch_size,), capture_max_kv_len - query_len, dtype=torch.int32
            ),
            input_lengths_device=input_lengths.to(device),
            cu_kv_seqlens_device=cu_kv_seqlens,
            kv_cache_kernel_block_id_device=page_table,
        )
        config = _make_config(softmax_extra_scale=1.0, q_scaling=1.0)
        op = FlashAttn4TargetVerifyOp(config)
        params = op.prepare(inputs)
        op.compile_probe(params)
        kv_cache = types.SimpleNamespace(kv_cache_base=combined_cache)
        self.assertEqual(params.max_kv_len, capture_max_kv_len)
        self.assertGreater(params.num_splits, 1)

        eager_output = op.forward(query, kv_cache, params)
        eager_reference = self._reference(
            query, combined_cache, page_table, initial_kv_lengths
        )
        torch.testing.assert_close(
            eager_output.float(), eager_reference, atol=2e-2, rtol=2e-2
        )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = op.forward(query, kv_cache, params)

        metadata_pointers = {
            "query_lengths": inputs.input_lengths_device.data_ptr(),
            "cu_kv_seqlens": cu_kv_seqlens.data_ptr(),
            "page_table": page_table.data_ptr(),
        }
        cu_kv_seqlens.copy_(
            torch.tensor(
                [0, replay_kv_lengths[0], sum(replay_kv_lengths)],
                dtype=torch.int32,
                device=device,
            )
        )
        page_table.copy_(
            torch.tensor(
                [[5, 0, 7, 1, 4], [3, 9, 6, 8, 2]],
                dtype=torch.int32,
                device=device,
            )
        )
        query.copy_(torch.randn_like(query))
        inputs.prefix_lengths.copy_(
            torch.tensor(
                [kv_len - query_len for kv_len in replay_kv_lengths],
                dtype=torch.int32,
            )
        )
        op.prepare_cuda_graph(params, inputs)
        self.assertEqual(params.kv_lengths.tolist(), replay_kv_lengths)
        self.assertEqual(
            inputs.input_lengths_device.data_ptr(),
            metadata_pointers["query_lengths"],
        )
        self.assertEqual(cu_kv_seqlens.data_ptr(), metadata_pointers["cu_kv_seqlens"])
        self.assertEqual(page_table.data_ptr(), metadata_pointers["page_table"])
        graph.replay()
        replay_reference = self._reference(
            query, combined_cache, page_table, replay_kv_lengths
        )
        torch.testing.assert_close(
            graph_output.float(), replay_reference, atol=2e-2, rtol=2e-2
        )


if __name__ == "__main__":
    unittest.main()
