"""SM90 registration of the QSA indexer instance rope/score buffers.

The fused QSA text-rope kernel already dispatches on SM90, but the indexer
instance buffers feeding the decode/target/prefill call sites were first
registered on SM120 only.  Those call sites therefore saw
``rope_inv_freq() -> None`` and ``score_weights() -> None`` on SM90,
rebuilt the constant tables on every call, and kept the fused Base-RoPE
branch cold.

On SM90 the instance buffers must:

* hold the same constant ``float32 [32]`` frequency table as the shared
  per-geometry ``cached_inv_freq`` cache (bit-equal: both are built with
  the reference ``pow``/``divide`` kernels);
* keep the ``[128, n_heads]`` score table constant at ``1/sqrt(head_dim)``
  for every sliced row count, matching the fallback tables the call sites
  build when the accessors decline;
* feed the fused text-rope kernel (Base transport and equal-axis text
  MRoPE transport) byte-identically to the eager rebuild;
* remain frozen constants across CUDA-graph capture and replay, because
  the decode graph replays the same registered buffer.
"""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import (
    Qwen4ExpQSAIndexer,
    build_base_rope,
    build_qsa_rope_from_logical_positions,
    cached_inv_freq,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.rope_triton import (
    try_build_text_rope,
)

MODULE = "rtp_llm.models_py.modules.qwen4_exp.qsa.rope_triton"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class QsaIndexerBufferSM90Test(unittest.TestCase):
    HEADS = 2
    D = 128

    def setUp(self):
        torch.manual_seed(271)
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("SM90 fast path")
        # Concrete device index so buffer-device comparisons never see the
        # implicit "cuda" form.
        self.device = torch.device("cuda", torch.cuda.current_device())

    def config(self, style="Mrope", **kwargs):
        config = dict(
            style=style,
            dim=64,
            base=1000000.0,
            scale=1.0,
            index_factor=3,
            mrope_interleaved=True,
            mrope_dim1=16,
            mrope_dim2=8,
            mrope_dim3=8,
            indexer_is_neox_style=True,
        )
        config.update(kwargs)
        return SimpleNamespace(**config)

    def make_indexer(self, rope_config, *, n_heads=HEADS):
        return Qwen4ExpQSAIndexer(
            torch.randn(
                (n_heads + 1) * self.D, 32, device="cuda", dtype=torch.bfloat16
            ),
            torch.zeros(self.D, device="cuda", dtype=torch.bfloat16),
            torch.zeros(self.D, device="cuda", dtype=torch.bfloat16),
            n_heads=n_heads,
            kv_heads=1,
            head_dim=self.D,
            token_budget=2048,
            compress_ratio=4,
            norm_eps=1e-6,
            rope_config=rope_config,
        )

    def fallback_weights(self, rows):
        return torch.full(
            (rows, self.HEADS),
            1.0 / math.sqrt(self.D),
            dtype=torch.float32,
            device=self.device,
        )

    def test_rope_buffer_matches_shared_cache_table(self):
        config = self.config("Base")
        indexer = self.make_indexer(config)
        frequency = indexer.rope_inv_freq(config)
        self.assertIsNotNone(frequency, "the indexer rope buffer declined SM90")
        shared = cached_inv_freq(config, self.device)
        self.assertTrue(torch.equal(frequency, shared))
        self.assertEqual(tuple(frequency.shape), (32,))
        self.assertEqual(frequency.dtype, torch.float32)
        self.assertEqual(frequency.device, self.device)
        self.assertTrue(frequency.is_contiguous())
        # persistent=False: the frozen constant never enters checkpoints.
        self.assertNotIn("_rope_inv_freq", indexer.state_dict())
        self.assertNotIn("_score_weight_buffer", indexer.state_dict())
        # Config drift declines so callers keep the reference path.
        self.assertIsNone(indexer.rope_inv_freq(self.config(base=10000.0)))
        # Each registered geometry pins its own table.
        other_config = self.config("Base", base=10000.0)
        other = self.make_indexer(other_config)
        other_frequency = other.rope_inv_freq(other_config)
        self.assertIsNotNone(other_frequency)
        self.assertTrue(
            torch.equal(other_frequency, cached_inv_freq(other_config, self.device))
        )
        self.assertFalse(torch.equal(other_frequency, frequency))

    def test_score_weights_match_call_site_fallbacks(self):
        indexer = self.make_indexer(self.config())
        for rows in (1, 2, 3, 7, 16, 127, 128):
            weights = indexer.score_weights(rows, self.device)
            self.assertIsNotNone(weights, f"rows={rows}")
            self.assertTrue(
                torch.equal(weights, self.fallback_weights(rows)),
                f"rows={rows}",
            )
        # Row overflow and non-positive rows decline to the call-site path.
        self.assertIsNone(indexer.score_weights(129, self.device))
        self.assertIsNone(indexer.score_weights(0, self.device))
        # Geometry drift declines.
        indexer.n_heads = 3
        self.assertIsNone(indexer.score_weights(4, self.device))

    def test_fused_rope_consumes_the_instance_buffer(self):
        config = self.config("Base", index_factor=3)
        indexer = self.make_indexer(config)
        frequency = indexer.rope_inv_freq(config)
        self.assertIsNotNone(frequency)
        positions = torch.arange(40, device=self.device, dtype=torch.int64)
        transported = positions.repeat_interleave(3)
        kwargs = dict(token_count=40, dtype=torch.bfloat16, device=self.device)
        # The fused kernel accepts the instance buffer directly, so the
        # decode-chain Base-RoPE branch is reachable on SM90.
        self.assertIsNotNone(
            try_build_text_rope(
                positions,
                config,
                "Base",
                dtype=torch.bfloat16,
                device=self.device,
                inv_freq=frequency,
            )
        )
        fused = build_base_rope(transported, config, inv_freq=frequency, **kwargs)
        with patch(MODULE + ".try_build_text_rope", return_value=None):
            eager = build_base_rope(transported, config, **kwargs)
        torch.testing.assert_close(fused[0], eager[0], rtol=0, atol=0)
        torch.testing.assert_close(fused[1], eager[1], rtol=0, atol=0)
        # The target/prefill call sites build from logical positions and
        # fuse equal-axis text MRoPE transports with the same buffer.
        mrope = self.config("Mrope")
        mrope_indexer = self.make_indexer(mrope)
        mrope_frequency = mrope_indexer.rope_inv_freq(mrope)
        self.assertIsNotNone(mrope_frequency)
        self.assertTrue(
            torch.equal(mrope_frequency, cached_inv_freq(mrope, self.device))
        )
        fused_mrope = build_qsa_rope_from_logical_positions(
            positions, mrope, inv_freq=mrope_frequency, **kwargs
        )
        with patch(MODULE + ".try_build_text_rope", return_value=None):
            eager_mrope = build_qsa_rope_from_logical_positions(
                positions, mrope, **kwargs
            )
        torch.testing.assert_close(fused_mrope[0], eager_mrope[0], rtol=0, atol=0)
        torch.testing.assert_close(fused_mrope[1], eager_mrope[1], rtol=0, atol=0)

    def test_buffer_constants_survive_cuda_graph_capture(self):
        config = self.config("Base", index_factor=1)
        indexer = self.make_indexer(config)
        frequency = indexer.rope_inv_freq(config)
        self.assertIsNotNone(frequency)

        def build():
            return build_qsa_rope_from_logical_positions(
                positions,
                config,
                token_count=16,
                dtype=torch.bfloat16,
                device=self.device,
                inv_freq=frequency,
            )

        positions = torch.zeros(16, device=self.device, dtype=torch.int64)
        for _ in range(3):  # warm the Triton compile outside capture
            build()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_cos, captured_sin = build()
        for prefix in (0, 127, 4093, 32749):
            positions.copy_(torch.arange(prefix, prefix + 16, device=self.device))
            graph.replay()
            with patch(MODULE + ".try_build_text_rope", return_value=None):
                eager_cos, eager_sin = build_qsa_rope_from_logical_positions(
                    positions,
                    config,
                    token_count=16,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
            torch.testing.assert_close(captured_cos, eager_cos, rtol=0, atol=0)
            torch.testing.assert_close(captured_sin, eager_sin, rtol=0, atol=0)
        # The frozen buffers are unchanged by capture and replay.
        self.assertTrue(
            torch.equal(indexer.rope_inv_freq(config), cached_inv_freq(config, self.device))
        )
        self.assertTrue(
            torch.equal(indexer.score_weights(16, self.device), self.fallback_weights(16))
        )


if __name__ == "__main__":
    unittest.main()
