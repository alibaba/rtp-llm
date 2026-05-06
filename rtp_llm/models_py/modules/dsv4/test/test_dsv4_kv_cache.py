"""Compatibility wrapper for the split DSV4 KV-cache tests."""

import unittest

from rtp_llm.models_py.modules.dsv4.test.test_dsv4_attention import (
    TestAttentionOutput,
    TestSWAKVCache,
)
from rtp_llm.models_py.modules.dsv4.test.test_dsv4_compressor import (
    TestCompressedKVCache,
    TestCompressorPrecision,
    TestSequenceLengthBoundaries,
)
from rtp_llm.models_py.modules.dsv4.test.test_dsv4_indexer import TestIndexerTopk


if __name__ == "__main__":
    unittest.main()
