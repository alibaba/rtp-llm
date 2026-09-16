import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.shared_expert import (
    combine_routed_and_shared,
)


class SharedExpertCombineTest(unittest.TestCase):
    def setUp(self):
        self.routed = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        self.shared = torch.tensor([[0.5, -0.5]], dtype=torch.float32)
        self.out = torch.empty((1, 2), dtype=torch.bfloat16)

    def test_bf16_add_writes_supplied_output(self):
        with patch.dict(
            os.environ,
            {"MOE_SHARED_EXPERT_BF16_ADD": "1", "MOE_STRICT_FUSED": "0"},
        ):
            result = combine_routed_and_shared(
                self.routed, self.shared, torch.bfloat16, out=self.out
            )

        self.assertIs(result, self.out)
        torch.testing.assert_close(
            result, (self.routed + self.shared).to(torch.bfloat16)
        )

    def test_fallback_writes_supplied_output(self):
        triton_module = "rtp_llm.models_py.triton_kernels.moe.shared_expert"
        fake_module = SimpleNamespace(
            fused_moe_epilogue=lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("unavailable")
            )
        )
        with (
            patch.dict(
                os.environ,
                {"MOE_SHARED_EXPERT_BF16_ADD": "0", "MOE_STRICT_FUSED": "0"},
            ),
            patch.dict(sys.modules, {triton_module: fake_module}),
        ):
            result = combine_routed_and_shared(
                self.routed, self.shared, torch.bfloat16, out=self.out
            )

        self.assertIs(result, self.out)
        torch.testing.assert_close(
            result, (self.routed + self.shared).to(torch.bfloat16)
        )


if __name__ == "__main__":
    unittest.main()
