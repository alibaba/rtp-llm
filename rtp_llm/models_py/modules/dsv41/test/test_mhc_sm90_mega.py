"""Host dispatch contracts for the SM90 vLLM two-kernel MegaMHC path."""

import unittest
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv41.hc import v41_mega_mhc as mega


def _cpu_args():
    return dict(
        attn_out=torch.zeros((1, 1, 8), dtype=torch.bfloat16),
        residual=torch.zeros((1, 1, 4, 8), dtype=torch.bfloat16),
        post=torch.zeros((1, 1, 4, 1), dtype=torch.float32),
        comb=torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        previous=object(),
        next_hc=object(),
        norm=object(),
    )


class MhcSm90MegaHostTest(unittest.TestCase):
    def setUp(self):
        mega._get_sm90_mega_mhc.cache_clear()
        self.addCleanup(mega._get_sm90_mega_mhc.cache_clear)

    def test_cpu_shape_does_not_import_tilelang_backend(self):
        args = _cpu_args()
        with patch.object(mega, "_get_sm90_mega_mhc") as backend:
            self.assertIsNone(mega.try_fused_post_pre(**args))
        backend.assert_not_called()

    def test_low_token_sm90_dispatches_to_single_backend(self):
        args = _cpu_args()
        sentinel = object()
        backend = Mock()
        backend.try_vllm_two_kernel_post_pre.return_value = sentinel
        with (
            patch.object(mega, "_can_try_sm90_mega_mhc", return_value=True),
            patch.object(mega, "_get_sm90_mega_mhc", return_value=backend),
        ):
            self.assertIs(mega.try_fused_post_pre(**args), sentinel)
        backend.try_vllm_two_kernel_post_pre.assert_called_once()

    def test_extended_modes_skip_sm90_backend(self):
        args = _cpu_args()
        for kwargs in (
            {"original_tokens": 16},
            {"reuse_residual": True},
            {"output_fp8": True},
        ):
            with self.subTest(kwargs=kwargs), patch.object(
                mega, "_get_sm90_mega_mhc"
            ) as backend:
                self.assertIsNone(mega.try_fused_post_pre(**args, **kwargs))
            backend.assert_not_called()


if __name__ == "__main__":
    unittest.main()
