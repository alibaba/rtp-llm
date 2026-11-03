"""Guard the existing EP8 dispatcher when adding large-EP small-M tiles."""
import os
import unittest
from unittest.mock import patch

from rtp_llm.platforms.ppu.kernels.ppu_attention_gemm import decode_attention_gemm_config


class AttentionGemmConfigTest(unittest.TestCase):
    @patch("deep_gemm.jit_kernels.utils.get_num_sms", return_value=39)
    def test_large_ep_wo_a_requires_opt_in_and_exact_topology(self, _):
        for ep in ("8", "16", "32"):
            for tp in ("1", "8"):
                for enabled in ("0", "1"):
                    with patch.dict(os.environ, {
                        "EP_SIZE": ep, "TP_SIZE": tp,
                        "DSV4_PPU_WOA_SMALL_TILE": enabled,
                    }):
                        for m in (2, 4, 6, 12):
                            config = decode_attention_gemm_config(m, 1024, 4096, groups=8)
                            if ep in ("16", "32") and tp == "1" and enabled == "1" and (m != 4 or ep == "16"):
                                self.assertEqual(config[1:7], (16, 64, 128, 16, 16, 4))
                            else:
                                self.assertIsNone(config)
                        self.assertIsNone(decode_attention_gemm_config(3, 1024, 4096, groups=8))
                        self.assertIsNone(decode_attention_gemm_config(6, 1024, 4096, groups=1))

    @patch("deep_gemm.jit_kernels.utils.get_num_sms", return_value=39)
    def test_wo_a_opt_in_preserves_existing_d8_tile(self, _):
        with patch.dict(os.environ, {
            "EP_SIZE": "8", "TP_SIZE": "1", "DSV4_PPU_WOA_SMALL_TILE": "0",
        }):
            old = decode_attention_gemm_config(24, 1024, 4096, groups=8)
        with patch.dict(os.environ, {
            "EP_SIZE": "8", "TP_SIZE": "1", "DSV4_PPU_WOA_SMALL_TILE": "1",
        }):
            self.assertEqual(old, decode_attention_gemm_config(24, 1024, 4096, groups=8))

    def test_ep8_keeps_small_shapes_on_original_dispatcher(self):
        for ep in ("1", "8"):
            with patch.dict(os.environ, {"EP_SIZE": ep}):
                for m in (2, 4, 6, 12):
                    for n, k in ((32768, 1024), (4096, 8192)):
                        self.assertIsNone(decode_attention_gemm_config(m, n, k))

    @patch("deep_gemm.jit_kernels.utils.get_num_sms", return_value=39)
    def test_old_qualified_shapes_are_identical_for_all_ep_sizes(self, _):
        for m, n, k in ((8, 32768, 1024), (24, 32768, 1024), (8, 4096, 8192)):
            with patch.dict(os.environ, {"EP_SIZE": "8"}):
                original = decode_attention_gemm_config(m, n, k)
            self.assertIsNotNone(original)
            for ep in ("16", "32"):
                with patch.dict(os.environ, {"EP_SIZE": ep}):
                    self.assertEqual(decode_attention_gemm_config(m, n, k), original)

    @patch("deep_gemm.jit_kernels.utils.get_num_sms", return_value=39)
    def test_only_qualified_large_ep_shapes_select_tiles(self, _):
        for ep in ("16", "32"):
            with patch.dict(os.environ, {"EP_SIZE": ep}):
                for m in (2, 4, 6, 12):
                    for n, k in ((32768, 1024), (4096, 8192)):
                        self.assertIsNotNone(decode_attention_gemm_config(m, n, k))
                self.assertIsNone(decode_attention_gemm_config(3, 4096, 8192))
                self.assertIsNone(decode_attention_gemm_config(4, 4096, 2048))
                self.assertIsNone(decode_attention_gemm_config(4, 4096, 8192, groups=8))

    @patch("deep_gemm.jit_kernels.utils.get_num_sms", return_value=20)
    def test_different_sm_allocation_keeps_original_dispatcher(self, _):
        with patch.dict(os.environ, {"EP_SIZE": "32"}):
            self.assertIsNone(decode_attention_gemm_config(4, 4096, 8192))
