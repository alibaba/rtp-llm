import logging
import os
import types
import unittest
from unittest import mock

import torch

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

    def test_invalid_log_level_is_reported_and_vendor_handler_is_removed(self):
        default_handler = logging.StreamHandler()
        vendor_logger = mock.Mock()
        fa_logging = types.SimpleNamespace(
            set_fa_log_level=mock.Mock(),
            _default_handler=default_handler,
            _logger=vendor_logger,
        )
        with mock.patch.dict(os.environ, {"FA_LOG_LEVEL": "verbose"}), self.assertLogs(
            py_flash_attn_v4.logger, level="WARNING"
        ) as captured:
            py_flash_attn_v4._configure_fa4_logging(fa_logging)

        fa_logging.set_fa_log_level.assert_called_once_with(0)
        vendor_logger.removeHandler.assert_called_once_with(default_handler)
        self.assertIsNone(fa_logging._default_handler)
        self.assertTrue(vendor_logger.propagate)
        self.assertIn("invalid FA_LOG_LEVEL", captured.output[0])

    def test_vendor_host_log_levels_map_to_rtp_logging_levels(self):
        vendor_logger = mock.Mock()
        fa_logging = types.SimpleNamespace(
            set_fa_log_level=mock.Mock(),
            get_fa_log_level=mock.Mock(return_value=3),
            _default_handler=None,
            _logger=vendor_logger,
        )
        with mock.patch.dict(os.environ, {"FA_LOG_LEVEL": "max"}):
            py_flash_attn_v4._configure_fa4_logging(fa_logging)

        fa_logging.fa_log(1, "host summary")
        fa_logging.fa_log(2, "kernel detail")
        fa_logging.fa_log(3, "maximum detail")
        vendor_logger.info.assert_called_once_with("host summary")
        vendor_logger.debug.assert_has_calls(
            [mock.call("kernel detail"), mock.call("maximum detail")]
        )

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
        params = FlashAttn4TargetVerifyParams(
            batch_size=2,
            query_len=5,
            max_kv_len=20,
            num_splits=1,
            query_lengths=torch.tensor([5, 5], dtype=torch.int32),
            kv_lengths=kv_lengths,
            cu_kv_seqlens=cu_kv,
            page_table=torch.zeros(2, 1, dtype=torch.int32),
        )
        FlashAttn4TargetVerifyOp.prepare_cuda_graph(params)
        self.assertEqual(kv_lengths.tolist(), [12, 19])

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

    @staticmethod
    def _reference(
        query,
        combined_cache,
        page_table,
        kv_len,
        query_len=5,
        num_q_heads=12,
        num_kv_heads=2,
        page_size=64,
    ):
        num_pages = (kv_len + page_size - 1) // page_size
        page_ids = page_table[0, :num_pages].to(torch.long)
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

        scores = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * (256**-0.5)
        query_positions = torch.arange(query_len, device=query.device)
        key_positions = torch.arange(kv_len, device=query.device)
        causal_mask = key_positions[None, :] <= (
            kv_len - query_len + query_positions[:, None]
        )
        scores.masked_fill_(~causal_mask[None, :, :], float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        return torch.einsum("hqk,khd->qhd", probabilities, value.float())

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
        query_len = 5
        initial_kv_len = 257
        replay_kv_len = 193
        page_table = torch.tensor([[4, 1, 7, 0, 5]], dtype=torch.int32, device=device)
        combined_cache = torch.randn(
            8, 2, 2, 64, 256, dtype=torch.bfloat16, device=device
        )
        query = torch.randn(query_len, 12, 256, dtype=torch.bfloat16, device=device)
        cu_kv_seqlens = torch.tensor(
            [0, initial_kv_len], dtype=torch.int32, device=device
        )
        inputs = types.SimpleNamespace(
            input_lengths=torch.tensor([query_len], dtype=torch.int32),
            prefix_lengths=torch.tensor(
                [initial_kv_len - query_len], dtype=torch.int32
            ),
            input_lengths_device=torch.tensor(
                [query_len], dtype=torch.int32, device=device
            ),
            cu_kv_seqlens_device=cu_kv_seqlens,
            kv_cache_kernel_block_id_device=page_table,
        )
        config = _make_config(softmax_extra_scale=1.0, q_scaling=1.0)
        op = FlashAttn4TargetVerifyOp(config)
        params = op.prepare(inputs)
        op.compile_probe(params)
        kv_cache = types.SimpleNamespace(kv_cache_base=combined_cache)

        eager_output = op.forward(query, kv_cache, params)
        eager_reference = self._reference(
            query, combined_cache, page_table, initial_kv_len
        )
        torch.testing.assert_close(
            eager_output.float(), eager_reference, atol=2e-2, rtol=2e-2
        )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = op.forward(query, kv_cache, params)

        cu_kv_seqlens.copy_(
            torch.tensor([0, replay_kv_len], dtype=torch.int32, device=device)
        )
        page_table.copy_(
            torch.tensor([[5, 0, 7, 1, 4]], dtype=torch.int32, device=device)
        )
        query.copy_(torch.randn_like(query))
        FlashAttn4TargetVerifyOp.prepare_cuda_graph(params)
        graph.replay()
        replay_reference = self._reference(
            query, combined_cache, page_table, replay_kv_len
        )
        torch.testing.assert_close(
            graph_output.float(), replay_reference, atol=2e-2, rtol=2e-2
        )


if __name__ == "__main__":
    unittest.main()
