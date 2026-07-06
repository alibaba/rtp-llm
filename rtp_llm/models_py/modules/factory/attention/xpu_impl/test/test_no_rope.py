"""Unit test: RopeStyle.No models must NOT invoke _apply_rope.

Verifies that both XpuVllmPrefillImpl and XpuVllmDecodeImpl skip RoPE
application when rope_config.style == RopeStyle.No or rope_config is None.
CPU-runnable (no XPU device required).
"""

from unittest import TestCase, main, SkipTest
from unittest.mock import patch

import torch

try:
    from rtp_llm.ops import RopeStyle
    from rtp_llm.models_py.modules.factory.attention.xpu_impl.vllm_flash_attn import (
        _need_rope,
        _split_qkv_and_rope,
    )
    _IMPORT_OK = True
except Exception as _e:
    _IMPORT_OK = False
    _IMPORT_ERR = _e


class _FakeRopeConfig:
    def __init__(self, style):
        self.style = style
        self.dim = 0
        self.base = 10000.0
        self.scale = 1.0
        self.is_neox_style = True


class _FakeAttnConfigs:
    def __init__(self, rope_config=None, need_rope_kv_cache=False):
        self.rope_config = rope_config
        self.need_rope_kv_cache = need_rope_kv_cache
        self.head_num = 4
        self.kv_head_num = 4
        self.size_per_head = 16


class _FakeAttnInputs:
    def __init__(self):
        self.is_prefill = True
        self.input_lengths = None
        self.prefix_lengths = None
        self.position_ids = None
        self.sequence_lengths = None


class TestNeedRope(TestCase):
    def setUp(self):
        if not _IMPORT_OK:
            raise SkipTest(f"Import failed: {_IMPORT_ERR}")

    def test_no_rope_style_returns_false(self):
        cfg = _FakeAttnConfigs(rope_config=_FakeRopeConfig(RopeStyle.No))
        self.assertFalse(_need_rope(cfg))

    def test_rope_config_none_returns_false(self):
        cfg = _FakeAttnConfigs(rope_config=None)
        self.assertFalse(_need_rope(cfg))

    def test_base_rope_style_returns_true(self):
        cfg = _FakeAttnConfigs(rope_config=_FakeRopeConfig(RopeStyle.Base))
        self.assertTrue(_need_rope(cfg))

    def test_need_rope_kv_cache_overrides(self):
        cfg = _FakeAttnConfigs(rope_config=None, need_rope_kv_cache=True)
        self.assertTrue(_need_rope(cfg))


class TestSplitQkvNoRope(TestCase):
    """Verify _split_qkv_and_rope does NOT call _apply_rope when need_rope=False."""

    def setUp(self):
        if not _IMPORT_OK:
            raise SkipTest(f"Import failed: {_IMPORT_ERR}")

    @patch(
        "rtp_llm.models_py.modules.factory.attention.xpu_impl.vllm_flash_attn._apply_rope",
        side_effect=AssertionError("_apply_rope should not be called for No-RoPE"),
    )
    def test_no_rope_skips_apply_rope(self, mock_apply):
        num_heads = 4
        num_kv_heads = 4
        head_dim = 16
        total_tokens = 2
        qkv_size = (num_heads + 2 * num_kv_heads) * head_dim
        qkv = torch.randn(total_tokens, qkv_size)

        rope_config = _FakeRopeConfig(RopeStyle.No)
        attn_inputs = _FakeAttnInputs()

        # need_rope=False: must NOT invoke _apply_rope
        q, k, v = _split_qkv_and_rope(
            qkv, attn_inputs, num_heads, num_kv_heads,
            head_dim, rope_config, need_rope=False,
        )

        mock_apply.assert_not_called()
        # Verify shapes
        self.assertEqual(q.shape, (total_tokens, num_heads, head_dim))
        self.assertEqual(k.shape, (total_tokens, num_kv_heads, head_dim))
        self.assertEqual(v.shape, (total_tokens, num_kv_heads, head_dim))

    @patch(
        "rtp_llm.models_py.modules.factory.attention.xpu_impl.vllm_flash_attn._apply_rope",
        side_effect=AssertionError("_apply_rope should not be called for None rope_config"),
    )
    def test_none_rope_config_skips_apply_rope(self, mock_apply):
        num_heads = 4
        num_kv_heads = 4
        head_dim = 16
        total_tokens = 2
        qkv_size = (num_heads + 2 * num_kv_heads) * head_dim
        qkv = torch.randn(total_tokens, qkv_size)

        attn_inputs = _FakeAttnInputs()
        cfg = _FakeAttnConfigs(rope_config=None)

        # _need_rope should return False for None rope_config
        need_rope = _need_rope(cfg)
        self.assertFalse(need_rope)

        q, k, v = _split_qkv_and_rope(
            qkv, attn_inputs, num_heads, num_kv_heads,
            head_dim, None, need_rope=False,
        )

        mock_apply.assert_not_called()


if __name__ == "__main__":
    main()
