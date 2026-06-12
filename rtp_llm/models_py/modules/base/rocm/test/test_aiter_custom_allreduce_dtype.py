# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``_AiterARManager.should_use`` dtype gating.

The manager's ``allreduce`` kernel only handles BF16 and FP8 dtypes;
unsupported dtypes (FP16/FP32) must fall through to the next tier
rather than hitting the kernel and producing garbage. This test
verifies that gating without requiring multiple GPUs by stubbing the
manager state.
"""

import unittest

import torch

from rtp_llm.models_py.modules.base.rocm.aiter_custom_allreduce import (
    _SUPPORTED_DTYPES,
    _AiterARManager,
)


class AiterCustomARDtypeTest(unittest.TestCase):
    def _make_initialized_manager(self) -> _AiterARManager:
        m = _AiterARManager()
        # Stub the post-init invariants the dtype check runs after.
        m.initialized = True
        m.disabled = False
        m.fa = 1  # any non-zero value
        m.group = object()
        m.device_id = 0
        m.max_size = 128 * 1024 * 1024
        return m

    def _eligible_tensor(self, dtype: torch.dtype) -> torch.Tensor:
        # Use CPU tensors and only check the dtype gate; numel * element_size
        # is divisible by 16 so the size-alignment check still passes.
        return torch.empty(1024, dtype=dtype)

    def test_supported_dtypes_pass_gate(self):
        m = self._make_initialized_manager()
        for dtype in _SUPPORTED_DTYPES:
            with self.subTest(dtype=dtype):
                t = self._eligible_tensor(dtype)
                self.assertTrue(m.should_use(t, m.group, m.device_id))

    def test_unsupported_dtypes_fall_through(self):
        m = self._make_initialized_manager()
        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                t = self._eligible_tensor(dtype)
                self.assertFalse(m.should_use(t, m.group, m.device_id))


if __name__ == "__main__":
    unittest.main()
