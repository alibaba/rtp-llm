"""Unit tests for xgrammar_bootstrap per-model think intercepted-token wiring.

Covers ``_resolve_excluded_token_ids`` / ``bootstrap_grammar_config``: a model
declares its thinking-phase intercepted token strings, bootstrap encodes them to ids
and writes ``grammar_config.excluded_token_ids`` (gated by enable_strict_thinking,
independent of the grammar backend).
"""

from types import SimpleNamespace
from unittest import TestCase, main

from rtp_llm.async_decoder_engine.xgrammar_bootstrap import (
    _resolve_excluded_token_ids,
    bootstrap_grammar_config,
)


class _Tok:
    def __init__(self, mapping):
        self._mapping = mapping

    def encode(self, text, add_special_tokens=True):
        return list(self._mapping.get(text, []))


def _model(strs, mapping, enable_strict_thinking=True):
    # model.tokenizer.tokenizer.encode(...) + model.get_think_excluded_token_strs()
    # + model.model_config.generate_env_config.enable_strict_thinking(服务启动开关)
    inner = _Tok(mapping)
    model = SimpleNamespace(
        tokenizer=SimpleNamespace(tokenizer=inner),
        model_config=SimpleNamespace(
            generate_env_config=SimpleNamespace(
                enable_strict_thinking=enable_strict_thinking
            )
        ),
    )
    model.get_think_excluded_token_strs = lambda: list(strs)
    return model


class XGrammarBootstrapTest(TestCase):
    def test_resolve_single_special_token(self):
        model = _model(["｜DSML｜"], {"｜DSML｜": [128825]})
        self.assertEqual(_resolve_excluded_token_ids(model), [128825])

    def test_resolve_empty_declaration(self):
        model = _model([], {"｜DSML｜": [128825]})
        self.assertEqual(_resolve_excluded_token_ids(model), [])

    def test_resolve_multi_token_skipped(self):
        # marker 编码成多 token → 无法当单 token rewrite,跳过(自然 gate 非匹配 tokenizer)。
        model = _model(["｜DSML｜"], {"｜DSML｜": [30, 128825]})
        self.assertEqual(_resolve_excluded_token_ids(model), [])

    def test_resolve_no_method_returns_empty(self):
        # 模型没声明 get_think_excluded_token_strs → 空。
        model = SimpleNamespace(tokenizer=SimpleNamespace(tokenizer=_Tok({})))
        self.assertEqual(_resolve_excluded_token_ids(model), [])

    def _engine_config(self):
        grammar_config = SimpleNamespace(
            grammar_backend="none",  # 后端关:验证 excluded 与 grammar 独立
            tokenizer_info_json="placeholder",
            excluded_token_ids=[],
        )
        return SimpleNamespace(grammar_config=grammar_config), grammar_config

    def test_bootstrap_switch_on_sets_excluded(self):
        # 服务启动开关 ON + 模型声明 → excluded_token_ids 填充(grammar 后端关也生效)。
        engine_config, grammar_config = self._engine_config()
        model = _model(
            ["｜DSML｜"], {"｜DSML｜": [128825]}, enable_strict_thinking=True
        )
        bootstrap_grammar_config(engine_config, model)
        self.assertEqual(list(grammar_config.excluded_token_ids), [128825])
        self.assertEqual(grammar_config.tokenizer_info_json, "")  # 后端关 → 清空

    def test_bootstrap_switch_off_keeps_empty(self):
        # 服务启动开关 OFF(默认)→ 即使模型声明也不 rewrite(行为不变)。
        engine_config, grammar_config = self._engine_config()
        model = _model(
            ["｜DSML｜"], {"｜DSML｜": [128825]}, enable_strict_thinking=False
        )
        bootstrap_grammar_config(engine_config, model)
        self.assertEqual(list(grammar_config.excluded_token_ids), [])

    def test_bootstrap_switch_on_but_model_declares_nothing(self):
        # 开关 ON 但模型没声明 excluded tokens → 空。
        engine_config, grammar_config = self._engine_config()
        model = _model([], {"｜DSML｜": [128825]}, enable_strict_thinking=True)
        bootstrap_grammar_config(engine_config, model)
        self.assertEqual(list(grammar_config.excluded_token_ids), [])


if __name__ == "__main__":
    main()
