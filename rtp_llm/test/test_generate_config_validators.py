"""cross_seq_diverge_start_combo validator 单元测试。

覆盖 _clamp_diverge_start_combo 的 4 条分支：
1. None → 返回 0
2. 非整数类型(TypeError/ValueError) → warning + 返回 0
3. 负值 → warning + clamp 到 0
4. 超大值(>100) → 保留原值（告警仅在特性启用时触发，与 C++ 侧一致）

覆盖 _check_cross_seq_ban_compatibility 的分支：
1. 多重不兼容 → 一次性报告所有原因 + 禁用
2. 合法配置 → 无 warning，保持启用
3. num_return_sequences 超 depth → 采样质量告警
4. update() 路径也触发校验
5. fail-safe 降级后 update 补齐条件 → WARNING 诊断日志

跨语言常量同步测试：
确保 Python _DIVERGE_START_COMBO_WARN_THRESHOLD / _MAX_DIVERGE_DEPTH
与 C++ kDivergeStartComboWarnThreshold / kMaxDivergeDepth 一致。
策略：Python 侧硬编码期望值 + C++ 侧 static_assert，无需解析源码。
"""

import logging
import unittest

from rtp_llm.config.generate_config import (
    GenerateConfig,
    _DIVERGE_START_COMBO_WARN_THRESHOLD,  # pyright: ignore[reportPrivateUsage]
    _MAX_DIVERGE_DEPTH,  # pyright: ignore[reportPrivateUsage]
    _reset_sanitize_warn_state,  # pyright: ignore[reportPrivateUsage]
)


class TestClampDivergeStartCombo(unittest.TestCase):

    def setUp(self):
        # 每个用例前复位限流状态，确保用例间独立不受全局时间戳影响。
        _reset_sanitize_warn_state()

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

    def test_exceeds_int32_clamped(self):
        """超过 INT32_MAX 的值应 clamp 到 2**31-1 并触发 warning。"""
        huge = 2**31 + 100
        with self.assertLogs(level=logging.WARNING) as cm:
            cfg = GenerateConfig(cross_seq_diverge_start_combo=huge)
        self.assertEqual(cfg.cross_seq_diverge_start_combo, 2**31 - 1)
        self.assertTrue(any("int32" in msg for msg in cm.output))

    def test_large_value_warns_but_keeps(self):
        """超大值(>100)且特性启用时应触发 warning 但保留原值。

        NOTE: 「过大」告警仅在 enable_cross_sequence_ban=True 且特性未被降级时触发，
        与 C++ 侧 (enable_cross_seq_ban && diverge_start_combo > threshold) 行为一致。
        """
        with self.assertLogs(level=logging.WARNING) as cm:
            cfg = GenerateConfig(
                cross_seq_diverge_start_combo=999,
                enable_cross_sequence_ban=True,
                combo_token_size=3,
                num_beams=1,
                num_return_sequences=2,
            )
        self.assertEqual(cfg.cross_seq_diverge_start_combo, 999)
        self.assertTrue(any("very large" in msg for msg in cm.output))

    def test_large_value_no_warn_when_disabled(self):
        """超大值但特性未启用时不产生告警噪声。"""
        with self.assertNoLogs(level=logging.WARNING):
            cfg = GenerateConfig(cross_seq_diverge_start_combo=999)
        self.assertEqual(cfg.cross_seq_diverge_start_combo, 999)

    def test_normal_value_no_warning(self):
        """正常值(0-100)无 warning，原值返回。"""
        with self.assertNoLogs(level=logging.WARNING):
            cfg = GenerateConfig(cross_seq_diverge_start_combo=5)
        self.assertEqual(cfg.cross_seq_diverge_start_combo, 5)

    def test_zero_value(self):
        """​0 是合法值，应原样返回。"""
        with self.assertNoLogs(level=logging.WARNING):
            cfg = GenerateConfig(cross_seq_diverge_start_combo=0)
        self.assertEqual(cfg.cross_seq_diverge_start_combo, 0)

    def test_boundary_100_no_warning(self):
        """恰好 100 不触发 warning。"""
        with self.assertNoLogs(level=logging.WARNING):
            cfg = GenerateConfig(cross_seq_diverge_start_combo=100)
        self.assertEqual(cfg.cross_seq_diverge_start_combo, 100)


class TestCrossSeqBanCompatibility(unittest.TestCase):
    """测试 enable_cross_sequence_ban 与 beam search / combo_token_size 的互斥校验。"""

    def setUp(self):
        _reset_sanitize_warn_state()

    def test_beam_search_incompatible_disables(self):
        """开启 cross_seq_ban + num_beams>1 应触发 warning 并禁用。"""
        with self.assertLogs(level=logging.WARNING) as cm:
            cfg = GenerateConfig(
                enable_cross_sequence_ban=True,
                num_beams=4,
                combo_token_size=3,
            )
        self.assertTrue(any("incompatible with beam search" in msg for msg in cm.output))
        self.assertFalse(cfg.enable_cross_sequence_ban)

    def test_variable_num_beams_incompatible_disables(self):
        """开启 cross_seq_ban + variable_num_beams 包含>1 应触发 warning 并禁用。"""
        with self.assertLogs(level=logging.WARNING) as cm:
            cfg = GenerateConfig(
                enable_cross_sequence_ban=True,
                num_beams=1,
                variable_num_beams=[1, 4],
                combo_token_size=3,
            )
        self.assertTrue(any("incompatible with beam search" in msg for msg in cm.output))
        self.assertFalse(cfg.enable_cross_sequence_ban)

    def test_combo_token_size_lt2_disables(self):
        """开启 cross_seq_ban + combo_token_size<2 应触发 warning 并禁用。"""
        with self.assertLogs(level=logging.WARNING) as cm:
            cfg = GenerateConfig(
                enable_cross_sequence_ban=True,
                combo_token_size=1,
            )
        self.assertTrue(any("combo_token_size" in msg for msg in cm.output))
        self.assertFalse(cfg.enable_cross_sequence_ban)

    def test_num_return_sequences_le1_disables(self):
        """开启 cross_seq_ban + num_return_sequences<=1 应触发 warning 并禁用。"""
        with self.assertLogs(level=logging.WARNING) as cm:
            cfg = GenerateConfig(
                enable_cross_sequence_ban=True,
                combo_token_size=3,
                num_beams=1,
                num_return_sequences=1,
            )
        self.assertTrue(any("num_return_sequences" in msg for msg in cm.output))
        self.assertFalse(cfg.enable_cross_sequence_ban)

    def test_multiple_incompatible_reports_all(self):
        """同时命中多条不兼容时，一次性报告所有原因。"""
        with self.assertLogs(level=logging.WARNING) as cm:
            cfg = GenerateConfig(
                enable_cross_sequence_ban=True,
                num_beams=4,
                combo_token_size=1,
                num_return_sequences=1,
            )
        # 单条 warning 中同时包含所有不兼容原因
        combined = " ".join(cm.output)
        self.assertIn("beam search", combined)
        self.assertIn("combo_token_size", combined)
        self.assertIn("num_return_sequences", combined)
        self.assertFalse(cfg.enable_cross_sequence_ban)

    def test_exceeds_max_diverge_depth_warns(self):
        """合法配置但 num_return_sequences 超过 max_diverge_depth 应告警采样质量。"""
        with self.assertLogs(level=logging.WARNING) as cm:
            cfg = GenerateConfig(
                enable_cross_sequence_ban=True,
                combo_token_size=3,
                num_beams=1,
                num_return_sequences=10,  # 10-1=9 > _MAX_DIVERGE_DEPTH=8
            )
        self.assertTrue(any("diverge depth" in msg for msg in cm.output))
        # 不禁用，只警告
        self.assertTrue(cfg.enable_cross_sequence_ban)

    def test_update_triggers_revalidation(self):
        """通过 update() 修改字段后应重新校验兼容性。"""
        cfg = GenerateConfig(
            enable_cross_sequence_ban=True,
            combo_token_size=3,
            num_beams=1,
            num_return_sequences=2,
        )
        self.assertTrue(cfg.enable_cross_sequence_ban)
        # update 为不兼容配置后应自动禁用
        with self.assertLogs(level=logging.WARNING):
            cfg.update({"num_beams": 4})
        self.assertFalse(cfg.enable_cross_sequence_ban)

    def test_update_clamps_diverge_start_combo_negative(self):
        """通过 update() 传入负值应 clamp 到 0。"""
        cfg = GenerateConfig()
        with self.assertLogs(level=logging.WARNING) as cm:
            cfg.update({"cross_seq_diverge_start_combo": -10})
        self.assertEqual(cfg.cross_seq_diverge_start_combo, 0)
        self.assertTrue(any("negative" in msg for msg in cm.output))

    def test_update_clamps_diverge_start_combo_non_integer(self):
        """通过 update() 传入非整数应 fallback 到 0。"""
        cfg = GenerateConfig()
        with self.assertLogs(level=logging.WARNING) as cm:
            cfg.update({"cross_seq_diverge_start_combo": "abc"})
        self.assertEqual(cfg.cross_seq_diverge_start_combo, 0)
        self.assertTrue(any("non-integer" in msg for msg in cm.output))

    def test_update_and_pop_clamps_diverge_start_combo(self):
        """通过 update_and_pop() 传入负值应 clamp 到 0。"""
        cfg = GenerateConfig()
        with self.assertLogs(level=logging.WARNING):
            remaining = cfg.update_and_pop({"cross_seq_diverge_start_combo": -5})
        self.assertEqual(cfg.cross_seq_diverge_start_combo, 0)
        # cross_seq_diverge_start_combo 是已知字段，应被 pop 掉
        self.assertNotIn("cross_seq_diverge_start_combo", remaining)

    def test_valid_config_no_warning(self):
        """合法配置不触发 warning，开关保持启用。"""
        with self.assertNoLogs(level=logging.WARNING):
            cfg = GenerateConfig(
                enable_cross_sequence_ban=True,
                combo_token_size=3,
                num_beams=1,
                num_return_sequences=2,
            )
        self.assertTrue(cfg.enable_cross_sequence_ban)

    def test_diverge_depth_warning_dedup(self):
        """采样质量软告警同一实例不重复输出；修改 num_return_sequences 后可重新告警。"""
        with self.assertLogs(level=logging.WARNING) as cm:
            cfg = GenerateConfig(
                enable_cross_sequence_ban=True,
                combo_token_size=3,
                num_beams=1,
                num_return_sequences=10,  # 9 > 8
            )
        depth_warnings = [m for m in cm.output if "diverge depth" in m]
        self.assertEqual(len(depth_warnings), 1)
        # 第二次 update 不改变 num_return_sequences → 不应再次告警
        with self.assertNoLogs(level=logging.WARNING):
            cfg.update({"combo_token_size": 4})
        # 修改 num_return_sequences 到新的超 depth 值 → 标志重置，应再次告警
        with self.assertLogs(level=logging.WARNING) as cm:
            cfg.update({"num_return_sequences": 12})
        self.assertTrue(any("diverge depth" in m for m in cm.output))

    def test_reenable_after_incremental_update(self):
        """先建后 update 场景：enable 因条件不足被降级，update 补齐条件后应自动重新启用。"""
        # 构造时 num_return_sequences=0 → 不满足条件 → enable 被降级为 False
        with self.assertLogs(level=logging.WARNING):
            cfg = GenerateConfig(
                enable_cross_sequence_ban=True,
                combo_token_size=3,
                num_beams=1,
                num_return_sequences=0,  # 不满足 >1
            )
        self.assertFalse(cfg.enable_cross_sequence_ban)
        # update 补齐条件 → 自动重新启用
        cfg.update({"num_return_sequences": 4})
        self.assertTrue(cfg.enable_cross_sequence_ban)

    def test_two_phase_update_and_pop_reenable(self):
        """模拟 request_extractor 两次 update_and_pop：第一次缺条件降级，第二次补齐后重新启用。"""
        # 第一阶段：config_json 包含 enable + combo_token_size，但缺 num_return_sequences
        with self.assertLogs(level=logging.WARNING):
            cfg = GenerateConfig(
                enable_cross_sequence_ban=True,
                combo_token_size=3,
                num_beams=1,
            )
        self.assertFalse(cfg.enable_cross_sequence_ban)  # 被降级
        # 第二阶段：顶层 kwargs 提供 num_return_sequences
        leftover = cfg.update_and_pop({"num_return_sequences": 4})
        self.assertTrue(cfg.enable_cross_sequence_ban)  # 重新启用
        self.assertEqual(leftover, {})  # num_return_sequences 被消费

    def test_no_failsafe_misfire_when_never_enabled(self):
        """用户从未开启 enable_cross_sequence_ban 时，即使其他条件满足也不应误报。"""
        with self.assertNoLogs(level=logging.WARNING):
            cfg = GenerateConfig(
                combo_token_size=3,
                num_beams=1,
                num_return_sequences=2,
                # enable_cross_sequence_ban 默认 False，用户从未开启
            )
        self.assertFalse(cfg.enable_cross_sequence_ban)

    def test_explicit_disable_in_update_not_overridden(self):
        """用户在 update 中显式传 enable=False 时，自动重启用不应覆盖其意图。"""
        # 构造：enable=True 但 num_return_sequences 不足 → 降级
        with self.assertLogs(level=logging.WARNING):
            cfg = GenerateConfig(
                enable_cross_sequence_ban=True,
                combo_token_size=3,
                num_beams=1,
                num_return_sequences=1,
            )
        self.assertFalse(cfg.enable_cross_sequence_ban)
        # 用户显式关闭 + 补齐条件 → 不应被重新启用
        cfg.update({"enable_cross_sequence_ban": False, "num_return_sequences": 4})
        self.assertFalse(cfg.enable_cross_sequence_ban)


class TestCrossLanguageConstantSync(unittest.TestCase):
    """Python/C++ 共享常量 + 启用条件同步断言，防止单侧修改导致漂移。

    常量同步策略（无源码解析）：
      Python 侧：本测试硬编码期望值，直接与 Python 常量比对。
      C++ 侧：RecommendationLogitsProcessor.cc 中 static_assert 钉住常量值。
      任一侧改值 → 对应断言编译/运行失败 → 错误信息明确指向需同步的另一侧。
    启用条件同步：通过真值表比对——直接构造 GenerateConfig 并验证
    enable_cross_sequence_ban 的最终值。C++ 侧必须对相同输入产生相同结论。
    """

    def test_diverge_start_combo_warn_threshold_sync(self):
        """确保 Python _DIVERGE_START_COMBO_WARN_THRESHOLD == C++ kDivergeStartComboWarnThreshold(=100)。

        C++ 侧通过 static_assert(kDivergeStartComboWarnThreshold == 100) 保证编译期不变。
        若 C++ 侧修改了值，需同步更新此处期望值和 Python 常量。
        """
        EXPECTED_CPP_VALUE = 100  # mirrors C++ kDivergeStartComboWarnThreshold
        self.assertEqual(
            _DIVERGE_START_COMBO_WARN_THRESHOLD, EXPECTED_CPP_VALUE,
            f"Python({_DIVERGE_START_COMBO_WARN_THRESHOLD}) != C++ expected({EXPECTED_CPP_VALUE}), "
            f"constants drifted! Check RecommendationLogitsProcessor.cc static_assert.")

    def test_max_diverge_depth_sync(self):
        """确保 Python _MAX_DIVERGE_DEPTH == C++ kMaxDivergeDepth(=8)。

        C++ 侧通过 static_assert(kMaxDivergeDepth == 8) 保证编译期不变。
        若 C++ 侧修改了值，需同步更新此处期望值和 Python 常量。
        """
        EXPECTED_CPP_VALUE = 8  # mirrors C++ kMaxDivergeDepth
        self.assertEqual(
            _MAX_DIVERGE_DEPTH, EXPECTED_CPP_VALUE,
            f"Python({_MAX_DIVERGE_DEPTH}) != C++ expected({EXPECTED_CPP_VALUE}), "
            f"constants drifted! Check RecommendationLogitsProcessor.cc static_assert.")

    def test_enable_conditions_sync(self):
        """SYNC 真值表比对：验证 Python 侧启用条件的行为语义。

        C++ RecommendationLogitsProcessor.cc::fromGenerateInput 中 enable_cross_seq_ban 的
        计算逻辑必须与以下真值表一致（取反关系）。
        修改任一侧条件时，此测试将失败（Python 侧），同时 C++ 侧的
        RecommendationLogitsProcessorTest::testEnableConditionsTruthTable 也应失败。

        三项禁用条件：
        1. has_num_beams()  ↔  C++: hasNumBeams()
        2. combo_token_size < 2  ↔  C++: combo_token_size >= 2 (取反)
        3. num_return_sequences <= 1  ↔  C++: num > 1 (取反)
        """
        # 真值表：(num_beams, combo_token_size, num_return_sequences) -> expected_enabled
        truth_table = [
            # 全部满足 → 启用
            (1, 3, 4, True),
            (1, 2, 2, True),
            # beam search 不兼容 → 禁用
            (2, 3, 4, False),
            (4, 3, 4, False),
            # combo_token_size < 2 → 禁用
            (1, 1, 4, False),
            (1, 0, 4, False),
            # num_return_sequences <= 1 → 禁用
            (1, 3, 1, False),
            # 多条件同时不满足 → 禁用
            (2, 1, 1, False),
        ]
        for num_beams, combo_size, num_ret, expected in truth_table:
            with self.subTest(num_beams=num_beams, combo_size=combo_size, num_ret=num_ret):
                cfg = GenerateConfig(
                    enable_cross_sequence_ban=True,
                    num_beams=num_beams,
                    combo_token_size=combo_size,
                    num_return_sequences=num_ret,
                )
                self.assertEqual(
                    cfg.enable_cross_sequence_ban, expected,
                    f"Input({num_beams}, {combo_size}, {num_ret}): "
                    f"expected enabled={expected}, got {cfg.enable_cross_sequence_ban}")


if __name__ == "__main__":
    unittest.main()
