import types
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl import py_flashinfer_mha
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferFa2TargetVerifyImpl,
    PyFlashinferMropeTargetVerifyImpl,
    PyFlashinferPrefillPagedAttnOp,
    _get_py_flashinfer_prefill_plan_workspace_size_bytes,
    _validate_py_flashinfer_prefill_wrapper,
)


class FakePrefillWrapper:
    def __init__(
        self,
        backend="fa2",
        plan_info=None,
        plan_failures=None,
    ):
        self._backend = backend
        self._fixed_batch_size = 0
        self._int_workspace_buffer = torch.empty(1, dtype=torch.uint8)
        self._paged_kv_indices_buf = None
        self._paged_kv_indptr_buf = None
        self._paged_kv_last_page_len_buf = None
        self._plan_info = plan_info
        self._qo_indptr_buf = None
        self._use_cuda_graph = False
        self.plan_calls = 0
        self.plan_failures = list(plan_failures or [])
        self.reset_workspace_buffer_args = None

    def reset_workspace_buffer(self, float_workspace_buffer, int_workspace_buffer):
        self.reset_workspace_buffer_args = (
            float_workspace_buffer,
            int_workspace_buffer,
        )

    def plan(self, *args, **kwargs):
        self.plan_calls += 1
        if self.plan_failures:
            raise RuntimeError(self.plan_failures.pop(0))


def make_plan_info(
    padded_batch_size,
    cta_tile_q,
    v_offset,
    s_offset,
    split_kv=True,
):
    values = [0] * 15
    values[0] = padded_batch_size
    values[3] = cta_tile_q
    values[10] = v_offset
    values[11] = s_offset
    values[14] = int(split_kv)
    return values


def make_fake_op(
    enable_cuda_graph=False,
    workspace_bytes=1024,
    plan_info=None,
    plan_failures=None,
    backend="fa2",
    stub_resize=True,
):
    op = object.__new__(PyFlashinferPrefillPagedAttnOp)
    op.backend = backend
    op.datatype = torch.float16
    op.kv_datatype = torch.float16
    op.enable_cuda_graph = enable_cuda_graph
    op.g_workspace_buffer = torch.empty(workspace_bytes, dtype=torch.uint8)
    op.head_dim_qk = 4
    op.head_dim_vo = 4
    op.local_head_num = 2
    op.local_kv_head_num = 1
    op.page_size = 16
    op.prefill_wrapper = FakePrefillWrapper(
        backend=backend,
        plan_info=plan_info,
        plan_failures=plan_failures,
    )
    op._plan_shape = (0, 0, 0)
    op._cuda_graph_workspace_size_upper_bound_bytes = 0
    op._owns_workspace_buffer = not enable_cuda_graph

    if stub_resize:
        def resize_workspace(self, required_bytes):
            self.g_workspace_buffer = torch.empty(required_bytes, dtype=torch.uint8)
            self.prefill_wrapper.reset_workspace_buffer(
                self.g_workspace_buffer,
                self.prefill_wrapper._int_workspace_buffer,
            )

        op._resize_workspace_buffer = types.MethodType(resize_workspace, op)
    return op


class TestPyFlashinferWorkspaceSizing(unittest.TestCase):
    def setUp(self):
        # Never mutate the process-global pools used by other attention tests.
        self.workspace_pool = []
        self.cuda_graph_workspaces = {}
        self._pool_patch = mock.patch.object(
            py_flashinfer_mha,
            "_g_py_flashinfer_workspace_pool",
            self.workspace_pool,
        )
        self._cuda_graph_pool_patch = mock.patch.object(
            py_flashinfer_mha,
            "_g_py_flashinfer_cuda_graph_workspace_buffers",
            self.cuda_graph_workspaces,
        )
        self._pool_patch.start()
        self._cuda_graph_pool_patch.start()
        self.addCleanup(self._cuda_graph_pool_patch.stop)
        self.addCleanup(self._pool_patch.stop)

    def test_mrope_target_verify_impl_supports_only_sm9x_target_verify(self):
        attn_configs = types.SimpleNamespace(
            rope_config=types.SimpleNamespace(style=py_flashinfer_mha.RopeStyle.Mrope),
            need_rope_kv_cache=True,
            dtype=torch.bfloat16,
            kv_cache_dtype=py_flashinfer_mha.KvCacheDataType.BASE,
            head_num=8,
            kv_head_num=1,
            size_per_head=128,
            kernel_tokens_per_block=64,
            is_causal=True,
        )
        attn_inputs = types.SimpleNamespace(
            is_target_verify=True,
            is_prefill=True,
        )

        with mock.patch.object(py_flashinfer_mha, "is_sm90", return_value=True):
            self.assertTrue(
                PyFlashinferMropeTargetVerifyImpl.support(
                    attn_configs,
                    attn_inputs,
                )
            )
            attn_inputs.is_target_verify = False
            self.assertFalse(
                PyFlashinferMropeTargetVerifyImpl.support(
                    attn_configs,
                    attn_inputs,
                )
            )
            attn_inputs.is_target_verify = True
            attn_configs.need_rope_kv_cache = False
            self.assertFalse(
                PyFlashinferMropeTargetVerifyImpl.support(
                    attn_configs,
                    attn_inputs,
                )
            )

        attn_configs.need_rope_kv_cache = True
        with mock.patch.object(py_flashinfer_mha, "is_sm90", return_value=False):
            self.assertFalse(
                PyFlashinferMropeTargetVerifyImpl.support(
                    attn_configs,
                    attn_inputs,
                )
            )

    def test_fa2_target_verify_impl_supports_non_mrope_sm9x_inputs(self):
        attn_configs = types.SimpleNamespace(
            rope_config=types.SimpleNamespace(style=py_flashinfer_mha.RopeStyle.Base),
            need_rope_kv_cache=True,
            dtype=torch.bfloat16,
            kv_cache_dtype=py_flashinfer_mha.KvCacheDataType.FP8,
            head_num=8,
            kv_head_num=1,
            size_per_head=256,
            kernel_tokens_per_block=64,
            is_causal=True,
        )
        attn_inputs = types.SimpleNamespace(
            is_target_verify=True,
            is_prefill=True,
        )

        with mock.patch.object(py_flashinfer_mha, "is_sm90", return_value=True):
            self.assertTrue(
                PyFlashinferFa2TargetVerifyImpl.support(attn_configs, attn_inputs)
            )
            attn_configs.rope_config.style = py_flashinfer_mha.RopeStyle.Mrope
            self.assertFalse(
                PyFlashinferFa2TargetVerifyImpl.support(attn_configs, attn_inputs)
            )
            attn_configs.rope_config.style = py_flashinfer_mha.RopeStyle.Base
            attn_configs.kernel_tokens_per_block = 48
            self.assertFalse(
                PyFlashinferFa2TargetVerifyImpl.support(attn_configs, attn_inputs)
            )

    def test_cuda_graph_workspace_is_shared_per_device(self):
        with mock.patch.object(
            py_flashinfer_mha,
            "DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_BYTES",
            16,
        ):
            first = py_flashinfer_mha.get_py_flashinfer_cuda_graph_workspace_buffer(
                "cpu", min_size_bytes=1
            )
            second = py_flashinfer_mha.get_py_flashinfer_cuda_graph_workspace_buffer(
                "cpu", min_size_bytes=1
            )
        self.assertIs(first, second)

    def test_cuda_graph_workspace_is_isolated_by_device_key(self):
        cuda_0_buffer = torch.empty(16, dtype=torch.uint8)
        cuda_1_buffer = torch.empty(16, dtype=torch.uint8)
        with (
            mock.patch.object(
                py_flashinfer_mha,
                "DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_BYTES",
                16,
            ),
            mock.patch.object(
                torch,
                "zeros",
                side_effect=(cuda_0_buffer, cuda_1_buffer),
            ) as allocate,
        ):
            first_cuda_0 = (
                py_flashinfer_mha.get_py_flashinfer_cuda_graph_workspace_buffer(
                    "cuda:0", min_size_bytes=1
                )
            )
            first_cuda_1 = (
                py_flashinfer_mha.get_py_flashinfer_cuda_graph_workspace_buffer(
                    "cuda:1", min_size_bytes=1
                )
            )
            second_cuda_0 = (
                py_flashinfer_mha.get_py_flashinfer_cuda_graph_workspace_buffer(
                    "cuda:0", min_size_bytes=1
                )
            )

        self.assertIs(first_cuda_0, cuda_0_buffer)
        self.assertIs(first_cuda_1, cuda_1_buffer)
        self.assertIs(second_cuda_0, cuda_0_buffer)
        self.assertEqual(allocate.call_count, 2)
        self.assertEqual(
            set(self.cuda_graph_workspaces),
            {("cuda", 0), ("cuda", 1)},
        )

    def test_workspace_pool_uses_smallest_sufficient_buffer(self):
        best_fit = torch.empty(64, dtype=torch.uint8)
        oversized = torch.empty(128, dtype=torch.uint8)
        self.workspace_pool.extend((oversized, best_fit))
        with mock.patch.object(
            py_flashinfer_mha,
            "DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_BYTES",
            16,
        ):
            selected = py_flashinfer_mha.get_py_flashinfer_workspace_buffer(
                "cpu", min_size_bytes=40
            )
        self.assertIs(selected, best_fit)
        self.assertEqual(self.workspace_pool, [oversized])

    def test_empty_workspace_pool_allocates_next_power_of_two(self):
        with mock.patch.object(
            py_flashinfer_mha,
            "DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_BYTES",
            16,
        ):
            selected = py_flashinfer_mha.get_py_flashinfer_workspace_buffer(
                "cpu", min_size_bytes=33
            )

        self.assertEqual(selected.numel(), 64)
        self.assertEqual(self.workspace_pool, [])

    def test_workspace_pool_evicts_largest_buffer_above_device_limit(self):
        small = torch.empty(32, dtype=torch.uint8)
        medium = torch.empty(64, dtype=torch.uint8)
        large = torch.empty(128, dtype=torch.uint8)
        self.workspace_pool.extend((small, medium))

        with mock.patch.object(
            py_flashinfer_mha,
            "MAX_PY_FLASHINFER_POOL_BUFFERS_PER_DEVICE",
            2,
        ):
            py_flashinfer_mha.release_py_flashinfer_workspace_buffer(large)

        self.assertEqual(self.workspace_pool, [small, medium])

    def test_workspace_request_rejects_allocation_over_safety_limit(self):
        with (
            mock.patch.object(
                py_flashinfer_mha,
                "DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_BYTES",
                16,
            ),
            mock.patch.object(
                py_flashinfer_mha,
                "MAX_PY_FLASHINFER_WORKSPACE_SIZE_BYTES",
                64,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "safety limit"):
                py_flashinfer_mha.get_py_flashinfer_workspace_buffer(
                    "cpu",
                    min_size_bytes=65,
                )

    def test_real_resize_resets_wrapper_and_releases_owned_buffer(self):
        op = make_fake_op(workspace_bytes=64, stub_resize=False)
        old_workspace = op.g_workspace_buffer
        new_workspace = torch.empty(128, dtype=torch.uint8)
        int_workspace = op.prefill_wrapper._int_workspace_buffer

        self.assertTrue(op._owns_workspace_buffer)
        with (
            mock.patch.object(
                py_flashinfer_mha,
                "get_py_flashinfer_workspace_buffer",
                return_value=new_workspace,
            ) as allocate,
            mock.patch.object(
                py_flashinfer_mha,
                "release_py_flashinfer_workspace_buffer",
            ) as release,
        ):
            op._resize_workspace_buffer(96)

        allocate.assert_called_once_with(old_workspace.device, 96)
        release.assert_called_once_with(old_workspace)
        self.assertIs(op.g_workspace_buffer, new_workspace)
        self.assertEqual(
            op.prefill_wrapper.reset_workspace_buffer_args,
            (new_workspace, int_workspace),
        )

    def test_real_cuda_graph_resize_resets_without_releasing_shared_buffer(self):
        op = make_fake_op(
            enable_cuda_graph=True,
            workspace_bytes=64,
            stub_resize=False,
        )
        old_workspace = op.g_workspace_buffer
        new_workspace = torch.empty(128, dtype=torch.uint8)
        int_workspace = op.prefill_wrapper._int_workspace_buffer

        self.assertFalse(op._owns_workspace_buffer)
        with (
            mock.patch.object(
                py_flashinfer_mha,
                "get_py_flashinfer_cuda_graph_workspace_buffer",
                return_value=new_workspace,
            ) as allocate,
            mock.patch.object(
                py_flashinfer_mha,
                "release_py_flashinfer_workspace_buffer",
            ) as release,
        ):
            op._resize_workspace_buffer(96)

        allocate.assert_called_once_with(old_workspace.device, 96)
        release.assert_not_called()
        self.assertIs(op.g_workspace_buffer, new_workspace)
        self.assertEqual(
            op.prefill_wrapper.reset_workspace_buffer_args,
            (new_workspace, int_workspace),
        )

    def test_plan_info_workspace_size_accepts_tensor_and_sequence(self):
        values = make_plan_info(
            padded_batch_size=3,
            cta_tile_q=8,
            v_offset=0,
            s_offset=2 * 3 * 8 * 4 * 4,
        )
        expected = 2 * 3 * 8 * 4 * 4 + 2 * 3 * 8 * 4
        self.assertEqual(
            _get_py_flashinfer_prefill_plan_workspace_size_bytes(values, 2, 4),
            expected,
        )
        self.assertEqual(
            _get_py_flashinfer_prefill_plan_workspace_size_bytes(
                torch.tensor(values, dtype=torch.int64), 2, 4
            ),
            expected,
        )

    def test_unknown_plan_info_layout_keeps_current_workspace_bound(self):
        with self.assertLogs(py_flashinfer_mha.logger, level="WARNING"):
            required_bytes = _get_py_flashinfer_prefill_plan_workspace_size_bytes(
                [0, 1],
                2,
                4,
            )
        self.assertEqual(required_bytes, 0)

    def test_plan_retry_expands_float_workspace(self):
        plan_info = make_plan_info(1, 8, 0, 256)
        op = make_fake_op(
            plan_info=plan_info,
            plan_failures=["Failed to allocate memory for batch_prefill_tmp_v"],
        )
        with mock.patch.object(
            py_flashinfer_mha,
            "DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_BYTES",
            1024,
        ):
            op._plan_prefill_with_workspace_retry(False)

        self.assertEqual(op.prefill_wrapper.plan_calls, 2)
        self.assertEqual(op.g_workspace_buffer.numel(), 2048)
        self.assertIsNotNone(op.prefill_wrapper.reset_workspace_buffer_args)

    def test_plan_retry_does_not_mask_int_workspace_error(self):
        op = make_fake_op(
            plan_info=make_plan_info(1, 8, 0, 256),
            plan_failures=["Failed to allocate memory for int_workspace"],
        )
        with self.assertRaisesRegex(RuntimeError, "int_workspace"):
            op._plan_prefill_with_workspace_retry(False)
        self.assertEqual(op.prefill_wrapper.plan_calls, 1)

    def test_plan_info_overflow_resizes_and_replans(self):
        plan_info = make_plan_info(
            padded_batch_size=2,
            cta_tile_q=8,
            v_offset=0,
            s_offset=2 * 2 * 8 * 4 * 4,
        )
        op = make_fake_op(workspace_bytes=128, plan_info=plan_info)
        with mock.patch.object(
            py_flashinfer_mha,
            "DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_BYTES",
            128,
        ):
            op._plan_prefill_with_workspace_retry(False)

        self.assertEqual(op.prefill_wrapper.plan_calls, 2)
        self.assertGreater(op.g_workspace_buffer.numel(), 128)

    def test_cuda_graph_capture_caches_plan_workspace_bound(self):
        op = make_fake_op(
            enable_cuda_graph=True,
            workspace_bytes=1024,
            plan_info=make_plan_info(1, 8, 0, 256),
        )
        with mock.patch.object(
            py_flashinfer_mha,
            "DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_BYTES",
            1024,
        ):
            op._record_workspace_size_after_plan(forbid_realloc=False)
        self.assertEqual(op._cuda_graph_workspace_size_upper_bound_bytes, 1024)

    def test_cuda_graph_replay_uses_cached_bound(self):
        op = make_fake_op(enable_cuda_graph=True, workspace_bytes=1024)
        op._cuda_graph_workspace_size_upper_bound_bytes = 1024
        op._check_cuda_graph_replay_workspace_size(forbid_realloc=True)

    def test_cuda_graph_replay_rejects_workspace_growth(self):
        op = make_fake_op(enable_cuda_graph=True, workspace_bytes=512)
        op._cuda_graph_workspace_size_upper_bound_bytes = 1024
        with self.assertRaisesRegex(RuntimeError, "too small during CUDA graph replay"):
            op._check_cuda_graph_replay_workspace_size(forbid_realloc=True)

    def test_non_fa2_capture_uses_current_workspace_bound(self):
        op = make_fake_op(
            enable_cuda_graph=True,
            workspace_bytes=1024,
            backend="fa3",
        )
        op._record_workspace_size_after_plan(forbid_realloc=False)
        self.assertEqual(op._cuda_graph_workspace_size_upper_bound_bytes, 1024)

    def test_flashinfer_compat_layer_fails_fast_on_missing_attrs(self):
        with self.assertRaisesRegex(RuntimeError, "Unsupported FlashInfer wrapper"):
            _validate_py_flashinfer_prefill_wrapper(types.SimpleNamespace())

    def test_real_sm9x_explicit_fa2_wrapper_uses_fa2(self):
        if not py_flashinfer_mha.is_sm90():
            self.skipTest("requires SM9x")

        attn_configs = types.SimpleNamespace(
            head_num=8,
            kv_head_num=1,
            size_per_head=256,
            kernel_tokens_per_block=64,
            dtype=torch.bfloat16,
            kv_cache_dtype=py_flashinfer_mha.KvCacheDataType.BASE,
            max_seq_len=32000,
            is_causal=True,
        )
        attn_inputs = types.SimpleNamespace(
            is_cuda_graph=False,
            is_target_verify=True,
        )
        op = PyFlashinferPrefillPagedAttnOp(
            attn_configs,
            attn_inputs,
            backend="fa2",
        )
        self.assertEqual(op.backend, "fa2")
        self.assertEqual(op.prefill_wrapper._backend, "fa2")

    def test_real_fa2_plan_info_workspace_size(self):
        if not py_flashinfer_mha.is_sm90():
            self.skipTest("requires SM9x")

        page_size = 64
        kv_len = 12000
        page_count = (kv_len + page_size - 1) // page_size
        workspace = torch.empty(
            py_flashinfer_mha.DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_BYTES,
            dtype=torch.uint8,
            device="cuda",
        )
        wrapper = py_flashinfer_mha.BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            "HND",
            backend="fa2",
        )
        wrapper.plan(
            torch.tensor([0, 5], dtype=torch.int32, device="cuda"),
            torch.tensor([0, page_count], dtype=torch.int32, device="cuda"),
            torch.arange(page_count, dtype=torch.int32, device="cuda"),
            torch.tensor(
                [kv_len - (page_count - 1) * page_size],
                dtype=torch.int32,
                device="cuda",
            ),
            8,
            1,
            256,
            page_size,
            causal=True,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        required_bytes = _get_py_flashinfer_prefill_plan_workspace_size_bytes(
            wrapper._plan_info,
            8,
            256,
        )
        self.assertGreater(required_bytes, 0)
        self.assertLessEqual(required_bytes, workspace.numel())


if __name__ == "__main__":
    unittest.main()
