import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.triton_kernels.moe.test.small_moe_fixture import (
    SmallMoeGemvTestFixture,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SmallMoeGemvTest(SmallMoeGemvTestFixture, unittest.TestCase):

    def test_executor_dispatch_and_nondefault_semantics(self):
        p = self.payload(1)
        expected = self.ref(p)
        with patch.object(self.module, "_small_moe_gemv", True):
            from rtp_llm.models_py.triton_kernels.moe.small_moe_gemv import (
                small_moe_gemv as candidate,
            )

            for activation in ("silu", "SiGLU"):
                with patch(
                    "rtp_llm.models_py.triton_kernels.moe.small_moe_gemv.small_moe_gemv",
                    wraps=candidate,
                ) as mock:
                    self.module.execute(p, activation, None, None, False, None)
                    mock.assert_called_once()
            actual = self.ref(p)
            torch.testing.assert_close(actual, expected, atol=0.0078125, rtol=0.008)
            for activation, expert_map, scale, apply, extra in (
                ("gelu", None, None, False, None),
                ("silu", p.expert_topk_ids, None, False, None),
                ("silu", None, p.expert_topk_weights, False, None),
                ("silu", None, None, True, None),
                ("silu", None, None, False, {"custom": True}),
            ):
                with patch(
                    "rtp_llm.models_py.triton_kernels.moe.small_moe_gemv.small_moe_gemv"
                ) as mock:
                    self.module.execute(p, activation, expert_map, scale, apply, extra)
                    mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
