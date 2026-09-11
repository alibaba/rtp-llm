"""Frozen module options must not fall through to mutable process environment."""

import importlib.util
import os
import unittest
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

_PATH = Path(__file__).resolve().parents[1] / "chunk_env.py"
_SPEC = importlib.util.spec_from_file_location("dsv4_chunk_options_tested", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class FrozenChunkOptionsTest(unittest.TestCase):
    def test_two_snapshots_ignore_conflicting_process_environment(self):
        first = MappingProxyType({"DSV4_CHUNK_TOKENS": "8192"})
        second = MappingProxyType({"DSV4_FP8_INDEXER_SCORE_CHUNK_ROWS": "2048"})
        with patch.dict(os.environ, {"DSV4_CHUNK_TOKENS": "123"}, clear=True):
            self.assertEqual(self.read(first), 8192)
            self.assertEqual(self.read(second), 2048)
            os.environ["DSV4_CHUNK_TOKENS"] = "456"
            self.assertEqual(self.read(first), 8192)
            self.assertEqual(self.read(second), 2048)

    def test_empty_snapshot_uses_default_without_environment_fallback(self):
        with patch.dict(os.environ, {"DSV4_CHUNK_TOKENS": "123"}, clear=True):
            self.assertEqual(self.read(MappingProxyType({})), 16384)
            self.assertEqual(self.read(None), 123)

    def test_snapshot_retains_global_precedence_and_minimum(self):
        self.assertEqual(
            self.read(
                {
                    "DSV4_CHUNK_TOKENS": "4096",
                    "DSV4_FP8_INDEXER_SCORE_CHUNK_ROWS": "2048",
                }
            ),
            4096,
        )
        self.assertEqual(self.read({"DSV4_CHUNK_TOKENS": "-1"}), 0)

    @staticmethod
    def read(options):
        return _MODULE.dsv4_chunk_tokens_from_env(
            "DSV4_FP8_INDEXER_SCORE_CHUNK_ROWS", options=options
        )


if __name__ == "__main__":
    unittest.main()
