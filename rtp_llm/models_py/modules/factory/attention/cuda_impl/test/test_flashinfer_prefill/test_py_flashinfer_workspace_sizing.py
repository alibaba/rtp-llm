import types
import unittest

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_BYTES,
    PyFlashinferPrefillPagedAttnOp,
    _get_py_flashinfer_prefill_plan_workspace_size_bytes,
    _validate_py_flashinfer_prefill_wrapper,
)


class FakePrefillWrapper:
    def __init__(
        self, backend="fa2", max_total_num_rows=None, plan_info=None, plan_failures=0
    ):
        self._backend = backend
        self._fixed_batch_size = 0
        self._int_workspace_buffer = torch.empty(1, dtype=torch.uint8)
        self._max_total_num_rows = max_total_num_rows
        self._paged_kv_indices_buf = None
        self._paged_kv_indptr_buf = None
        self._paged_kv_last_page_len_buf = None
        self._plan_info = plan_info
        self._qo_indptr_buf = None
        self._use_cuda_graph = False
        self.plan_calls = 0
        self.plan_failures = plan_failures
        self.reset_workspace_buffer_args = None

    def reset_workspace_buffer(self, float_workspace_buffer, int_workspace_buffer):
        self.reset_workspace_buffer_args = (
            float_workspace_buffer,
            int_workspace_buffer,
        )

    def plan(self, *args, **kwargs):
        self.plan_calls += 1
        if self.plan_failures > 0:
            self.plan_failures -= 1
            raise RuntimeError("Failed to allocate memory for batch_prefill_tmp_v")


def make_plan_info(padded_batch_size, cta_tile_q, v_offset, s_offset, split_kv=True):
    values = [0] * 15
    values[0] = padded_batch_size
    values[3] = cta_tile_q
    values[10] = v_offset
    values[11] = s_offset
    values[14] = int(split_kv)
    return torch.tensor(values, dtype=torch.int64)


def make_fake_op(
    enable_cuda_graph=False, workspace_bytes=None, plan_info=None, plan_failures=0
):
    op = object.__new__(PyFlashinferPrefillPagedAttnOp)
    op.backend = "fa2"
    op.datatype = torch.float16
    op.enable_cuda_graph = enable_cuda_graph
    op.g_workspace_buffer = torch.empty(
        workspace_bytes or DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_BYTES,
        dtype=torch.uint8,
    )
    op.head_dim_vo = 128
    op.local_head_num = 40
    op.local_kv_head_num = 8
    op.page_size = 16
    op.prefill_wrapper = FakePrefillWrapper(
        plan_info=plan_info,
        plan_failures=plan_failures,
    )
    op._cuda_graph_workspace_size_upper_bound_bytes = 0
    op._last_estimated_workspace_size_bytes = 0
    op._last_plan_workspace_size_bytes = 0
    op._owns_workspace_buffer = False
    return op


class TestPyFlashinferWorkspaceSizing(unittest.TestCase):
    def test_plan_info_workspace_size_uses_flashinfer_offsets(self):
        num_qo_heads = 2
        padded_batch_size = 3
        cta_tile_q = 8
        head_dim_vo = 4
        tmp_v_bytes = num_qo_heads * padded_batch_size * cta_tile_q * head_dim_vo * 4
        tmp_s_bytes = num_qo_heads * padded_batch_size * cta_tile_q * 4
        plan_info = make_plan_info(
            padded_batch_size,
            cta_tile_q,
            v_offset=0,
            s_offset=tmp_v_bytes,
        )

        self.assertEqual(
            _get_py_flashinfer_prefill_plan_workspace_size_bytes(
                plan_info,
                num_qo_heads,
                head_dim_vo,
            ),
            tmp_v_bytes + tmp_s_bytes,
        )

    def test_normal_prepare_uses_current_workspace_without_preplan_resize(self):
        op = make_fake_op(workspace_bytes=1024)
        required = op._check_cuda_graph_replay_workspace_size(forbid_realloc=False)

        self.assertEqual(required, 1024)
        self.assertEqual(op._last_estimated_workspace_size_bytes, 1024)
        self.assertIsNone(op.prefill_wrapper.reset_workspace_buffer_args)

    def test_plan_info_check_caches_cuda_graph_workspace_upper_bound(self):
        plan_info = make_plan_info(
            padded_batch_size=1,
            cta_tile_q=8,
            v_offset=0,
            s_offset=40 * 1 * 8 * 128 * 4,
        )
        op = make_fake_op(enable_cuda_graph=True)
        op.prefill_wrapper._plan_info = plan_info

        op._check_workspace_size_after_plan(forbid_realloc=False)

        self.assertGreaterEqual(
            op._cuda_graph_workspace_size_upper_bound_bytes,
            DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_BYTES,
        )

    def test_non_fa2_cuda_graph_capture_caches_current_workspace_bound(self):
        op = make_fake_op(enable_cuda_graph=True, workspace_bytes=1024)
        op.backend = "fa3"
        op.prefill_wrapper._backend = "fa3"
        op.prefill_wrapper._plan_info = None

        op._check_workspace_size_after_plan(forbid_realloc=False)

        self.assertEqual(op._cuda_graph_workspace_size_upper_bound_bytes, 1024)

    def test_cuda_graph_replay_uses_cached_upper_bound_without_tensor_read(self):
        op = make_fake_op(enable_cuda_graph=True)
        op._cuda_graph_workspace_size_upper_bound_bytes = (
            DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_BYTES
        )

        required = op._check_cuda_graph_replay_workspace_size(forbid_realloc=True)

        self.assertEqual(required, DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_BYTES)

    def test_cuda_graph_replay_forbid_realloc_reports_cached_bound_overflow(self):
        op = make_fake_op(enable_cuda_graph=True, workspace_bytes=1)
        op._cuda_graph_workspace_size_upper_bound_bytes = (
            DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_BYTES
        )

        with self.assertRaisesRegex(RuntimeError, "too small during CUDA graph replay"):
            op._check_cuda_graph_replay_workspace_size(forbid_realloc=True)

    def test_plan_retry_expands_workspace_when_cta_estimate_is_low(self):
        op = make_fake_op(workspace_bytes=1024, plan_failures=1)

        workspace_bytes = op._plan_prefill_with_workspace_retry(
            workspace_bytes=1024,
            forbid_realloc=False,
        )

        self.assertEqual(op.prefill_wrapper.plan_calls, 2)
        self.assertGreater(workspace_bytes, 1024)
        self.assertIsNotNone(op.prefill_wrapper.reset_workspace_buffer_args)

    def test_plan_info_check_uses_actual_plan_as_cuda_graph_upper_bound(self):
        plan_info = make_plan_info(
            padded_batch_size=1,
            cta_tile_q=8,
            v_offset=0,
            s_offset=40 * 1 * 8 * 128 * 4,
        )
        op = make_fake_op(
            enable_cuda_graph=True,
            workspace_bytes=1024 * 1024,
            plan_info=plan_info,
        )

        op._check_workspace_size_after_plan(forbid_realloc=False)

        self.assertGreater(op._last_plan_workspace_size_bytes, 1)
        self.assertGreaterEqual(
            op._cuda_graph_workspace_size_upper_bound_bytes,
            DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_BYTES,
        )

    def test_plan_info_check_reports_actual_workspace_overflow(self):
        plan_info = make_plan_info(
            padded_batch_size=2,
            cta_tile_q=8,
            v_offset=0,
            s_offset=40 * 2 * 8 * 128 * 4,
        )
        op = make_fake_op(workspace_bytes=1024, plan_info=plan_info)

        with self.assertRaisesRegex(RuntimeError, "exceeds workspace buffer"):
            op._check_workspace_size_after_plan(forbid_realloc=False)

    def test_flashinfer_compat_layer_fails_fast_on_missing_attrs(self):
        with self.assertRaisesRegex(RuntimeError, "Unsupported FlashInfer wrapper"):
            _validate_py_flashinfer_prefill_wrapper(types.SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
