"""Guard the EP8 dispatcher while exposing large-EP tiles as an explicit profile."""
import os
import unittest
from unittest.mock import patch

from rtp_llm.models_py.kernels.ppu.attention.ppu_attention_gemm import decode_attention_gemm_config


class AttentionGemmConfigTest(unittest.TestCase):
    @patch("deep_gemm.jit_kernels.utils.get_num_sms", return_value=39)
    def test_d8_keeps_small_shapes_on_the_library_dispatcher(self, _):
        for ep in (1, 8):
            for m in (2, 4, 6, 12):
                for n, k in ((32768, 1024), (4096, 8192)):
                    self.assertIsNone(
                        decode_attention_gemm_config(
                            m, n, k, ep_size=ep, tp_size=1,
                            profile="large_ep_full",
                        )
                    )
            self.assertIsNotNone(
                decode_attention_gemm_config(24, 1024, 4096, groups=8, ep_size=ep)
            )

    @patch("deep_gemm.jit_kernels.utils.get_num_sms", return_value=39)
    def test_large_ep_tiles_need_no_opt_in(self, _):
        for ep in (16, 32):
            for m in (2, 4, 6, 12):
                for n, k in ((32768, 1024), (4096, 8192)):
                    self.assertIsNotNone(
                        decode_attention_gemm_config(m, n, k, ep_size=ep, tp_size=1)
                    )
            self.assertIsNone(decode_attention_gemm_config(3, 4096, 8192, ep_size=ep))
            self.assertIsNone(decode_attention_gemm_config(4, 4096, 2048, ep_size=ep))
            self.assertIsNone(
                decode_attention_gemm_config(4, 4096, 8192, groups=8, ep_size=ep)
            )

    @patch("deep_gemm.jit_kernels.utils.get_num_sms", return_value=39)
    def test_wo_a_small_tile_requires_profile_and_exact_topology(self, _):
        # m=4 is only qualified for EP16; m=2/6/12 for both large-EP sizes.
        for ep in (16, 32):
            for m in (2, 4, 6, 12):
                got = decode_attention_gemm_config(
                    m, 1024, 4096, groups=8, ep_size=ep, tp_size=1,
                    profile="large_ep_full",
                )
                if m == 4 and ep == 32:
                    self.assertIsNone(got)
                else:
                    self.assertEqual(got[1:7], (16, 64, 128, 16, 16, 4))
        # Any of: small EP, TP>1, missing profile, wrong shape or groups must
        # fall back instead of widening the qualified set.
        for kwargs in (
            {"ep_size": 8, "tp_size": 1, "profile": "large_ep_full"},
            {"ep_size": 16, "tp_size": 2, "profile": "large_ep_full"},
            {"ep_size": 16, "tp_size": 1, "profile": "auto"},
        ):
            self.assertIsNone(
                decode_attention_gemm_config(2, 1024, 4096, groups=8, **kwargs)
            )
        self.assertIsNone(
            decode_attention_gemm_config(
                2, 1024, 4096, groups=1, ep_size=16, tp_size=1,
                profile="large_ep_full",
            )
        )
        self.assertIsNone(
            decode_attention_gemm_config(
                2, 2048, 4096, groups=8, ep_size=16, tp_size=1,
                profile="large_ep_full",
            )
        )

    def test_kernel_ignores_process_environment(self):
        # Topology and the profile are kernel arguments now; a stale shell
        # export must not widen the qualified set.
        with patch.dict(os.environ, {
            "EP_SIZE": "32", "TP_SIZE": "1", "DSV4_PPU_WOA_SMALL_TILE": "1",
        }):
            with patch("deep_gemm.jit_kernels.utils.get_num_sms", return_value=39):
                self.assertIsNone(
                    decode_attention_gemm_config(2, 1024, 4096, groups=8, ep_size=8)
                )
                self.assertIsNotNone(
                    decode_attention_gemm_config(
                        2, 1024, 4096, groups=8, ep_size=16, tp_size=1,
                        profile="large_ep_full",
                    )
                )

    @patch("deep_gemm.jit_kernels.utils.get_num_sms", return_value=39)
    def test_old_qualified_shapes_are_identical_for_all_ep_sizes(self, _):
        for m, n, k in ((8, 32768, 1024), (24, 32768, 1024), (8, 4096, 8192)):
            original = decode_attention_gemm_config(m, n, k, ep_size=8)
            self.assertIsNotNone(original)
            for ep in (16, 32):
                self.assertEqual(
                    decode_attention_gemm_config(m, n, k, ep_size=ep), original
                )

    @patch("deep_gemm.jit_kernels.utils.get_num_sms", return_value=20)
    def test_different_sm_allocation_keeps_the_library_dispatcher(self, _):
        self.assertIsNone(decode_attention_gemm_config(4, 4096, 8192, ep_size=32))


if __name__ == "__main__":
    unittest.main()
