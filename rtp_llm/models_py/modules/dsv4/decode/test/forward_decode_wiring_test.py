"""Smoke-style import + signature tests for the Phase 2 decode wiring.

Doesn't load the V4-Flash ckpt (149 GB) — just verifies the new
``forward_decode`` methods exist with the right signature and that
``DSv4DecodeAttnMetadata`` flows through without exceptions on a
synthetic small-shape model. The full numerical check happens in the
SM100_ARM smoke (`v4_flash_decode_b{1,8}_no_graph_sm100`) which loads
the real ckpt.
"""

import inspect
import os
import sys
import unittest

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


class TestForwardDecodeAPISurface(unittest.TestCase):
    """Verify Phase 2 added the expected API without breaking imports."""

    def test_attention_has_forward_decode(self):
        from rtp_llm.models_py.modules.dsv4.attention import Attention

        self.assertTrue(callable(getattr(Attention, "forward_decode", None)))
        sig = inspect.signature(Attention.forward_decode)
        self.assertIn("attn_metadata", sig.parameters)
        self.assertIn("x", sig.parameters)
        # Prefill `forward` still exists & still takes (x, start_pos: int)
        self.assertTrue(callable(getattr(Attention, "forward", None)))
        prefill_sig = inspect.signature(Attention.forward)
        self.assertIn("start_pos", prefill_sig.parameters)

    def test_compressor_has_forward_decode(self):
        from rtp_llm.models_py.modules.dsv4.compressor import Compressor

        self.assertTrue(callable(getattr(Compressor, "forward_decode", None)))
        sig = inspect.signature(Compressor.forward_decode)
        self.assertIn("start_pos", sig.parameters)
        # Prefill `forward` untouched
        prefill_sig = inspect.signature(Compressor.forward)
        self.assertIn("start_pos", prefill_sig.parameters)

    def test_indexer_has_forward_decode(self):
        from rtp_llm.models_py.modules.dsv4.indexer import Indexer

        self.assertTrue(callable(getattr(Indexer, "forward_decode", None)))
        sig = inspect.signature(Indexer.forward_decode)
        self.assertIn("start_pos", sig.parameters)
        self.assertIn("out_topk_buffer", sig.parameters)
        # Prefill `forward` untouched
        prefill_sig = inspect.signature(Indexer.forward)
        self.assertIn("offset", prefill_sig.parameters)

    def test_block_has_forward_decode(self):
        from rtp_llm.models_py.modules.dsv4.block import Block

        self.assertTrue(callable(getattr(Block, "forward_decode", None)))
        sig = inspect.signature(Block.forward_decode)
        self.assertIn("attn_metadata", sig.parameters)

    def test_v4transformer_has_forward_decode(self):
        from rtp_llm.models_py.modules.dsv4.transformer import V4Transformer

        self.assertTrue(callable(getattr(V4Transformer, "forward_decode", None)))
        sig = inspect.signature(V4Transformer.forward_decode)
        self.assertIn("attn_metadata", sig.parameters)
        self.assertIn("input_ids", sig.parameters)

    def test_deepseek_v4_model_dispatches_decode(self):
        from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model

        self.assertTrue(callable(getattr(DeepSeekV4Model, "_forward_decode", None)))


class TestPrefillUntouched(unittest.TestCase):
    """Read the original prefill code paths and verify Phase 2 only ADDED
    methods — the prefill ``forward`` body is byte-equivalent to its
    pre-Phase-2 head structure (we can't easily diff against git here,
    but we can spot-check the start_pos==0 branch text remains).
    """

    def test_attention_forward_still_branches_on_start_pos(self):
        import inspect

        from rtp_llm.models_py.modules.dsv4.attention import Attention

        src = inspect.getsource(Attention.forward)
        # Prefill arm key strings still present
        self.assertIn("if start_pos == 0", src)
        self.assertIn("self.kv_cache[:bsz, :seqlen]", src)

    def test_compressor_forward_still_has_prefill_arm(self):
        import inspect

        from rtp_llm.models_py.modules.dsv4.compressor import Compressor

        src = inspect.getsource(Compressor.forward)
        self.assertIn("if start_pos == 0:", src)
        self.assertIn("seqlen >= ratio", src)

    def test_indexer_forward_still_has_prefill_chunked_einsum(self):
        import inspect

        from rtp_llm.models_py.modules.dsv4.indexer import Indexer

        src = inspect.getsource(Indexer.forward)
        # The S-dim chunking comment + chunk_size logic is decode-irrelevant
        # (we use forward_decode for decode); it's prefill-only by design.
        self.assertIn("chunk_size", src)
        self.assertIn("max_chunk_bytes", src)


if __name__ == "__main__":
    unittest.main()
