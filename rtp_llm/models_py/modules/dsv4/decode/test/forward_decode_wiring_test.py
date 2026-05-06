"""Import and signature tests for the current DSV4 decode wiring."""

import inspect
import os
import sys
import unittest

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


class TestForwardDecodeAPISurface(unittest.TestCase):
    def test_attention_decode_requires_metadata_and_optional_kv_cache(self):
        from rtp_llm.models_py.modules.dsv4.attention import Attention

        sig = inspect.signature(Attention.forward_decode)
        self.assertIn("x", sig.parameters)
        self.assertIn("attn_metadata", sig.parameters)
        self.assertIn("kv_cache", sig.parameters)
        self.assertTrue(callable(getattr(Attention, "_forward_decode_body", None)))

    def test_compressor_decode_paths_exist(self):
        from rtp_llm.models_py.modules.dsv4.compressor import Compressor

        self.assertIn("start_pos", inspect.signature(Compressor.forward).parameters)
        self.assertIn(
            "start_pos", inspect.signature(Compressor.forward_decode).parameters
        )
        self.assertIn(
            "start_pos",
            inspect.signature(Compressor.forward_decode_vectorized).parameters,
        )

    def test_indexer_decode_paths_exist(self):
        from rtp_llm.models_py.modules.dsv4.indexer import Indexer

        self.assertIn("offset", inspect.signature(Indexer.forward).parameters)
        self.assertIn("out_topk_buffer", inspect.signature(Indexer.forward_decode).parameters)
        self.assertIn(
            "out_topk_buffer",
            inspect.signature(Indexer.forward_decode_vectorized).parameters,
        )

    def test_block_and_transformer_decode_surface(self):
        from rtp_llm.models_py.modules.dsv4.block import Block
        from rtp_llm.models_py.modules.dsv4.transformer import V4Transformer

        self.assertIn("attn_metadata", inspect.signature(Block.forward_decode).parameters)
        self.assertIn(
            "attn_metadata", inspect.signature(V4Transformer.forward_decode).parameters
        )
        self.assertIn("kv_cache", inspect.signature(V4Transformer.forward_decode).parameters)

    def test_decode_forward_helper_surface(self):
        from rtp_llm.models_py.modules.dsv4.decode import forward as decode_forward

        self.assertTrue(callable(decode_forward.forward_decode))
        self.assertTrue(callable(decode_forward.forward_layers))
        self.assertTrue(callable(decode_forward.build_metadata_eager))


if __name__ == "__main__":
    unittest.main()
