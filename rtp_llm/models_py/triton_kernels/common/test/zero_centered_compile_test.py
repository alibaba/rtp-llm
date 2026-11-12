"""Qwen4 fused Q/K norm parity and CUDA Graph replay tests."""

import unittest

import torch
import triton


class Qwen4FusedQKNormTest(unittest.TestCase):
    def test_prefill_new_row_counts_reuse_compiled_kernel(self):
        from rtp_llm.models_py.triton_kernels.common.zero_centered_rms_norm import (
            try_prefill_qk_rmsnorm_,
        )

        gamma = torch.zeros(256, device="cuda", dtype=torch.bfloat16)

        def norm(rows):
            qkv = torch.ones(rows, 5 * 256, device="cuda", dtype=torch.bfloat16)
            output = try_prefill_qk_rmsnorm_(qkv, gamma, gamma, 3, 1, 256, 1e-6)
            self.assertIsNotNone(output)
            torch.testing.assert_close(output, torch.ones_like(output), rtol=0, atol=0)

        norm(267)
        compiles = []
        with triton.knobs.compilation.scope():
            triton.knobs.compilation.listener = (
                lambda *, cache_hit, **_: compiles.append(cache_hit)
            )
            norm(269)
            norm(271)
        self.assertNotIn(False, compiles)


if __name__ == "__main__":
    unittest.main()
