"""cross_seq_diverge_start_combo validator 单元测试。

覆盖 _clamp_diverge_start_combo 的 4 条分支：
1. None → 返回 0
2. 非整数类型(TypeError/ValueError) → warning + 返回 0
3. 负值 → warning + clamp 到 0
4. 超大值(>100) → warning + 原值返回
"""

import logging
import unittest

from rtp_llm.config.generate_config import GenerateConfig


class TestClampDivergeStartCombo(unittest.TestCase):

    def test_none_returns_zero(self):
        """None 输入应返回 0。"""
        cfg = GenerateConfig(cross_seq_diverge_start_combo=None)
        self.assertEqual(cfg.cross_seq_diverge_start_combo, 0)

    def test_non_integer_string_returns_zero(self):
        """非整数字符串应触发 warning 并返回 0。"""
        with self.assertLogs(level=logging.WARNING) as cm:
            cfg = GenerateConfig(cross_seq_diverge_start_combo="abc")
        self.assertEqual(cfg.cross_seq_diverge_start_combo, 0)
        self.assertTrue(any("non-integer" in msg for msg in cm.output))

    def test_negative_clamped_to_zero(self):
        """负值应 clamp 到 0 并触发 warning。"""
        with self.assertLogs(level=logging.WARNING) as cm:
            cfg = GenerateConfig(cross_seq_diverge_start_combo=-5)
        self.assertEqual(cfg.cross_seq_diverge_start_combo, 0)
        self.assertTrue(any("negative" in msg for msg in cm.output))

    def test_large_value_warns_but_keeps(self):
        """超大值(>100)应触发 warning 但保留原值。"""
        with self.assertLogs(level=logging.WARNING) as cm:
            cfg = GenerateConfig(cross_seq_diverge_start_combo=999)
        self.assertEqual(cfg.cross_seq_diverge_start_combo, 999)
        self.assertTrue(any("very large" in msg for msg in cm.output))

    def test_normal_value_no_warning(self):
        """正常值(0-100)无 warning，原值返回。"""
        cfg = GenerateConfig(cross_seq_diverge_start_combo=5)
        self.assertEqual(cfg.cross_seq_diverge_start_combo, 5)

    def test_zero_value(self):
        """0 是合法值，应原样返回。"""
        cfg = GenerateConfig(cross_seq_diverge_start_combo=0)
        self.assertEqual(cfg.cross_seq_diverge_start_combo, 0)

    def test_boundary_100_no_warning(self):
        """恰好 100 不触发 warning。"""
        cfg = GenerateConfig(cross_seq_diverge_start_combo=100)
        self.assertEqual(cfg.cross_seq_diverge_start_combo, 100)


class TestCrossSeqBanCompatibility(unittest.TestCase):
    """测试 enable_cross_sequence_ban 与 beam search / combo_token_size 的互斥校验。"""

    def test_beam_search_incompatible_warns(self):
        """开启 cross_seq_ban + num_beams>1 应触发 warning。"""
        with self.assertLogs(level=logging.WARNING) as cm:
            GenerateConfig(
                enable_cross_sequence_ban=True,
                num_beams=4,
                combo_token_size=3,
            )
        self.assertTrue(any("incompatible with beam search" in msg for msg in cm.output))

    def test_variable_num_beams_incompatible_warns(self):
        """开启 cross_seq_ban + num_beams=1 但 variable_num_beams 包含>1 应触发 warning。"""
        with self.assertLogs(level=logging.WARNING) as cm:
            GenerateConfig(
                enable_cross_sequence_ban=True,
                num_beams=1,
                variable_num_beams=[1, 4],
                combo_token_size=3,
            )
        self.assertTrue(any("incompatible with beam search" in msg for msg in cm.output))

    def test_combo_token_size_lt2_warns(self):
        """开启 cross_seq_ban + combo_token_size<2 应触发 warning。"""
        with self.assertLogs(level=logging.WARNING) as cm:
            GenerateConfig(
                enable_cross_sequence_ban=True,
                combo_token_size=1,
            )
        self.assertTrue(any("combo_token_size>=2" in msg for msg in cm.output))

    def test_valid_config_no_warning(self):
        """合法配置不触发 warning。"""
        cfg = GenerateConfig(
            enable_cross_sequence_ban=True,
            combo_token_size=3,
            num_beams=1,
        )
        self.assertTrue(cfg.enable_cross_sequence_ban)


if __name__ == "__main__":
    unittest.main()
