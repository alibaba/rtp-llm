"""Buffer ABI isolation in the public runtime; no optional backend is loaded."""
import sys
import types
import unittest
from unittest import mock

from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe import buffer


class MegaMoeBufferBackendTest(unittest.TestCase):
    def test_buffers_are_shared_only_within_same_backend(self):
        group = object()
        old = types.ModuleType("deep_gemm")
        native = types.ModuleType("independent_backend_fixture")
        old.get_symm_buffer_for_mega_moe_k3 = mock.Mock(side_effect=lambda **kw: object())
        native.get_symm_buffer_for_mega_moe_k3 = mock.Mock(side_effect=lambda **kw: object())
        for backend in (old, native):
            backend.fp8_fp4_mega_moe_k3 = mock.Mock()
            backend.transform_weights_for_mega_moe_k3 = mock.Mock()
        args = (group, 896, 629, 16, 3584, 3072, True, "situ")
        with mock.patch.dict(buffer._MEGA_BUF_CACHE, {}, clear=True), mock.patch.dict(
            sys.modules, {"deep_gemm": old}
        ):
            a = buffer._get_or_create_mega_buf(*args)
            b = buffer._get_or_create_mega_buf(*args, backend=native)
            self.assertIsNot(a, b)
            self.assertIs(buffer._get_or_create_mega_buf(*args), a)
            self.assertIs(buffer._get_or_create_mega_buf(*args, backend=native), b)
            old.get_symm_buffer_for_mega_moe_k3.assert_called_once()
            native.get_symm_buffer_for_mega_moe_k3.assert_called_once()
            self.assertIn((id(group), *args[1:], "mega_moe_k3"), buffer._MEGA_BUF_CACHE)


if __name__ == "__main__":
    unittest.main()
