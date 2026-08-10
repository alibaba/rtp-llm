import types
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl import py_flashinfer_mha
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferPrefillPagedAttnOp,
)


class _FailingPrefillWrapper:
    def __init__(self, error_message):
        self.error_message = error_message
        self.plan_calls = 0
        self._int_workspace_buffer = torch.empty(1, dtype=torch.uint8)
        self.reset_workspace_buffer_args = None

    def plan(self, *args, **kwargs):
        self.plan_calls += 1
        raise RuntimeError(self.error_message)

    def reset_workspace_buffer(self, float_workspace_buffer, int_workspace_buffer):
        self.reset_workspace_buffer_args = (
            float_workspace_buffer,
            int_workspace_buffer,
        )


def _make_retry_op(workspace_bytes, error_message):
    op = object.__new__(PyFlashinferPrefillPagedAttnOp)
    op.g_workspace_buffer = torch.empty(workspace_bytes, dtype=torch.uint8)
    op.prefill_wrapper = _FailingPrefillWrapper(error_message)
    op._plan_shape = (2, 5, 1024)

    def resize_workspace(self, required_bytes):
        allocation_bytes = py_flashinfer_mha._workspace_allocation_size(required_bytes)
        self.g_workspace_buffer = torch.empty(allocation_bytes, dtype=torch.uint8)
        self.prefill_wrapper.reset_workspace_buffer(
            self.g_workspace_buffer,
            self.prefill_wrapper._int_workspace_buffer,
        )

    op._resize_workspace_buffer = types.MethodType(resize_workspace, op)
    return op


class TestPyFlashinferWorkspaceHelpersCpu(unittest.TestCase):
    def test_workspace_allocation_rounds_up_and_enforces_limit(self):
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
            self.assertEqual(py_flashinfer_mha._workspace_allocation_size(33), 64)
            with self.assertRaisesRegex(RuntimeError, "safety limit"):
                py_flashinfer_mha._workspace_allocation_size(65)

    def test_plan_retry_reports_non_stabilizing_workspace(self):
        op = _make_retry_op(
            1,
            "Failed to allocate memory for batch_prefill_tmp_v",
        )
        with (
            mock.patch.object(
                py_flashinfer_mha,
                "DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_BYTES",
                1,
            ),
            mock.patch.object(
                py_flashinfer_mha,
                "MAX_PY_FLASHINFER_WORKSPACE_SIZE_BYTES",
                1024,
            ),
            mock.patch.object(
                py_flashinfer_mha,
                "MAX_PY_FLASHINFER_WORKSPACE_RETRIES",
                3,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "did not stabilize"):
                op._plan_prefill_with_workspace_retry(False)

        self.assertEqual(op.prefill_wrapper.plan_calls, 3)
        self.assertEqual(op.g_workspace_buffer.numel(), 8)

    def test_plan_retry_reports_workspace_safety_limit(self):
        op = _make_retry_op(
            64,
            "Failed to allocate memory for batch_prefill_tmp_s",
        )
        with (
            mock.patch.object(
                py_flashinfer_mha,
                "DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_BYTES",
                1,
            ),
            mock.patch.object(
                py_flashinfer_mha,
                "MAX_PY_FLASHINFER_WORKSPACE_SIZE_BYTES",
                64,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "reached its safety limit"):
                op._plan_prefill_with_workspace_retry(False)

        self.assertEqual(op.prefill_wrapper.plan_calls, 1)
        self.assertIsNone(op.prefill_wrapper.reset_workspace_buffer_args)


if __name__ == "__main__":
    unittest.main()
