"""单元测试：recommendation_parser

覆盖关键场景：
1. 开关关闭时不解析
2. combo_token_size<=0 时不解析
3. 正常解析并填充（含 pos 不连续场景）
4. 已有 banned_combo_token_ids 与解析结果合并去重
5. auto-fill end_think_token_ids 的白名单/覆盖/encode 异常/空 banned 时不触发
"""

import unittest
from types import SimpleNamespace
from typing import List, Optional

from rtp_llm.frontend.recommendation_parser import parse_and_fill_banned_combo


class FakeTokenizer:
    """最小的 fake tokenizer：C<N> -> N + 10000，保证每个语义 ID 是一个 token。

    默认 name_or_path 不含 qwen3，不命中 auto-fill 白名单，与旧版本行为一致。
    新测试需要 auto-fill 时显式覆写 name_or_path 为 'qwen3/...' 触发白名单。
    """

    unk_token_id = -1
    name_or_path = "fake/non-qwen"
    model_type = ""
    encode_raises = False

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token.startswith("C")
        return 10000 + int(token[1:])

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if self.encode_raises:
            raise RuntimeError("mock encode failure")
        # 将整段 prelude 字符串映射为四个固定 token，足以驱动 auto-fill 测试
        return [1001, 1002, 1003, 1004]


def _make_config(
    auto_parse: bool,
    combo_token_size: int,
    banned: Optional[List[List[int]]] = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        auto_parse_banned_combo=auto_parse,
        combo_token_size=combo_token_size,
        banned_combo_token_ids=list(banned) if banned else [],
    )


class RecommendationParserTest(unittest.TestCase):
    PROMPT = (
        "你是一个专业的电商推荐系统...已推荐曝光的商品序列和位置:"
        "pos0:C1071C2997C4163,pos1:C741C3248C4162,pos2:C122C3449C4118,"
        "pos4:C1790C2171C4228,pos9:C575C2599C4165.输出商品长度为10的三层码本语义ID序列."
    )

    def test_switch_off_should_not_parse(self):
        cfg = _make_config(auto_parse=False, combo_token_size=3)
        n = parse_and_fill_banned_combo(self.PROMPT, cfg, FakeTokenizer())
        self.assertEqual(0, n)
        self.assertEqual([], cfg.banned_combo_token_ids)

    def test_combo_size_disabled_should_not_parse(self):
        cfg = _make_config(auto_parse=True, combo_token_size=0)
        n = parse_and_fill_banned_combo(self.PROMPT, cfg, FakeTokenizer())
        self.assertEqual(0, n)
        self.assertEqual([], cfg.banned_combo_token_ids)

    def test_parse_and_fill_enabled(self):
        cfg = _make_config(auto_parse=True, combo_token_size=3)
        n = parse_and_fill_banned_combo(self.PROMPT, cfg, FakeTokenizer())
        # prompt 里有 5 个 posN:C..C..C.. 条目
        self.assertEqual(5, n)
        self.assertIn([11071, 12997, 14163], cfg.banned_combo_token_ids)
        self.assertIn([10575, 12599, 14165], cfg.banned_combo_token_ids)

    def test_merge_with_existing_dedup(self):
        existing = [[11071, 12997, 14163]]  # pos0 商品已在 banned 中
        cfg = _make_config(auto_parse=True, combo_token_size=3, banned=existing)
        n = parse_and_fill_banned_combo(self.PROMPT, cfg, FakeTokenizer())
        # pos0 已有，应去重，只追加剩余 4 项
        self.assertEqual(4, n)
        self.assertEqual(5, len(cfg.banned_combo_token_ids))

    def test_empty_prompt_no_op(self):
        cfg = _make_config(auto_parse=True, combo_token_size=3)
        n = parse_and_fill_banned_combo("", cfg, FakeTokenizer())
        self.assertEqual(0, n)
        self.assertEqual([], cfg.banned_combo_token_ids)

    def test_auto_fill_skipped_when_user_preset(self):
        """用户已显式配 end_think_token_ids 时，auto-fill 应跳过不覆盖。"""
        tok = FakeTokenizer()
        tok.name_or_path = "qwen3/fake"  # 命中白名单，验证 "已设值不覆盖" 优先级
        cfg = _make_config(auto_parse=True, combo_token_size=3)
        cfg.end_think_token_ids = [42, 43]
        parse_and_fill_banned_combo(self.PROMPT, cfg, tok)
        self.assertEqual([42, 43], cfg.end_think_token_ids)

    def test_auto_fill_silent_on_encode_failure(self):
        """白名单命中但 encode 抛异常时，end_think_token_ids 保持未设置。"""
        tok = FakeTokenizer()
        tok.name_or_path = "qwen3/fake"
        tok.encode_raises = True
        cfg = _make_config(auto_parse=True, combo_token_size=3)
        parse_and_fill_banned_combo(self.PROMPT, cfg, tok)
        self.assertIsNone(getattr(cfg, "end_think_token_ids", None))

    def test_auto_fill_skipped_when_banned_combo_empty(self):
        """prompt 不含 posN 且 existing banned_combo 为空 → 不触发 auto-fill。"""
        tok = FakeTokenizer()
        tok.name_or_path = "qwen3/fake"
        cfg = _make_config(auto_parse=True, combo_token_size=3)
        parse_and_fill_banned_combo("plain prompt without pos patterns", cfg, tok)
        self.assertEqual([], cfg.banned_combo_token_ids)
        self.assertIsNone(getattr(cfg, "end_think_token_ids", None))


if __name__ == "__main__":
    unittest.main()
