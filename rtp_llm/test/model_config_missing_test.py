"""Fail-fast contract tests for missing config.json in checkpoint dirs.

This branch changed KimiK25's ``_read_top_config`` and DeepSeekVLV2's
``_create_config`` from silently returning ``{}``/``None`` to raising a
``FileNotFoundError`` that carries the offending checkpoint path. These
tests pin that behavior so it cannot silently regress.
"""

import tempfile
from unittest import TestCase, main

from rtp_llm.models.deepseek_vl2.deepseek_vl2 import DeepSeekVLV2
from rtp_llm.models.kimi_k25.kimi_k25 import _read_top_config


class KimiK25MissingConfigTest(TestCase):
    def test_read_top_config_raises_with_path(self):
        with tempfile.TemporaryDirectory() as ckpt_dir:
            with self.assertRaises(FileNotFoundError) as cm:
                _read_top_config(ckpt_dir)
            self.assertIn(ckpt_dir, str(cm.exception))


class DeepSeekVLV2MissingConfigTest(TestCase):
    def test_create_config_raises_with_path(self):
        with tempfile.TemporaryDirectory() as ckpt_dir:
            with self.assertRaises(FileNotFoundError) as cm:
                DeepSeekVLV2._create_config(ckpt_dir)
            self.assertIn(ckpt_dir, str(cm.exception))


if __name__ == "__main__":
    main()
