"""Unit tests for ``rtp_llm.dash_sc.app`` helpers (echo_prefix startup derivation)."""

from __future__ import annotations

from unittest import TestCase, main
from unittest.mock import patch

from rtp_llm.dash_sc import app as bg_app
from rtp_llm.dash_sc.app import _derive_echo_prefix_ids


class _EnvCfg:
    def __init__(self, think_mode: int = 1, think_start_tag: str = "<think>\n"):
        self.think_mode = think_mode
        self.think_start_tag = think_start_tag


class _ModelCfg:
    ckpt_path = "/tmp/ckpt"
    tokenizer_path = "/tmp/tok"
    model_type = "fake"


class _FakeTokenizer:
    def __init__(self, *, ids=None, raise_exc: bool = False):
        self._ids = ids or []
        self._raise = raise_exc
        self.encode_calls: list[tuple[str, bool]] = []

    def encode(self, text, add_special_tokens=True):
        if self._raise:
            raise RuntimeError("tokenizer.encode failed")
        self.encode_calls.append((text, add_special_tokens))
        return list(self._ids)


class _BaseTok:
    def __init__(self, tok):
        self.tokenizer = tok


class DeriveEchoPrefixIdsTest(TestCase):
    def test_encodes_think_start_tag(self) -> None:
        tok = _FakeTokenizer(ids=[154841])
        with patch.object(
            bg_app.TokenizerFactory, "create", return_value=_BaseTok(tok)
        ):
            ids = _derive_echo_prefix_ids(_ModelCfg(), _EnvCfg())
        self.assertEqual(ids, [154841])
        # Must encode without special tokens so only the tag bytes become ids.
        self.assertEqual(tok.encode_calls, [("<think>\n", False)])

    def test_disabled_when_think_mode_off(self) -> None:
        with patch.object(bg_app.TokenizerFactory, "create") as create:
            ids = _derive_echo_prefix_ids(_ModelCfg(), _EnvCfg(think_mode=0))
        self.assertEqual(ids, [])
        create.assert_not_called()

    def test_disabled_when_tag_empty(self) -> None:
        with patch.object(bg_app.TokenizerFactory, "create") as create:
            ids = _derive_echo_prefix_ids(_ModelCfg(), _EnvCfg(think_start_tag=""))
        self.assertEqual(ids, [])
        create.assert_not_called()

    def test_fail_open_on_tokenizer_error(self) -> None:
        with patch.object(
            bg_app.TokenizerFactory, "create", side_effect=RuntimeError("no tokenizer")
        ):
            ids = _derive_echo_prefix_ids(_ModelCfg(), _EnvCfg())
        self.assertEqual(ids, [])


if __name__ == "__main__":
    main()
