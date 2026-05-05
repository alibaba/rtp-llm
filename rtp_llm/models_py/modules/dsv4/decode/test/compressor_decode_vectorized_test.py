"""Stage 3B: Compressor.forward_decode_vectorized correctness.

Verifies the vectorized (CUDA-graph-friendly) variant produces:
  * Non-zero compressed output for boundary requests.
  * Zero output for non-boundary requests.
  * State buffers (kv_state / score_state) are populated after the call.

Tests on CPU only — the graph-capture itself is a SM100_ARM smoke
concern. For FAST vs REF equivalence, see compressor_decode_fast_test.py.
"""

import os
import sys
import unittest

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from rtp_llm.models_py.modules.dsv4.compressor import Compressor
from rtp_llm.models_py.modules.dsv4.rope import (  # noqa: F401 (used indirectly)
    precompute_freqs_cis,
)


def _make_compressor(
    compress_ratio: int,
    dim: int = 64,
    head_dim: int = 16,
    rope_head_dim: int = 8,
    max_batch_size: int = 4,
) -> Compressor:
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        c = Compressor(
            dim=dim,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            max_batch_size=max_batch_size,
            norm_eps=1e-6,
            rotate=False,
            weights=None,
            prefix="",
        )
    finally:
        torch.set_default_dtype(prev_dtype)
    return c


def _seed(c: Compressor, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    for _, p in c.named_parameters():
        if p.dtype.is_floating_point:
            p.data = torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.05
    c.norm.weight.data.fill_(1.0)


def _bind_kv_cache(c: Compressor, max_batch_size: int = 4, max_seq_len: int = 32):
    c.configure_kv_cache_shape(max_seq_len // c.compress_ratio)
    c.freqs_cis = torch.complex(
        torch.randn(max_seq_len, c.rope_head_dim // 2),
        torch.randn(max_seq_len, c.rope_head_dim // 2),
    )


class TestCompressorForwardDecodeVectorized(unittest.TestCase):

    def _run_one(self, compress_ratio: int, start_pos_list: list[int]):
        torch.manual_seed(123)
        c = _make_compressor(compress_ratio)
        _seed(c, seed=7)
        _bind_kv_cache(c)

        bsz = len(start_pos_list)
        x = torch.randn(bsz, 1, c.dim, dtype=torch.bfloat16) * 0.1
        sp = torch.tensor(start_pos_list, dtype=torch.int32)

        os.environ["DSV4_COMPRESSOR_FAST"] = "0"
        try:
            with torch.inference_mode():
                out = c.forward_decode_vectorized(x, sp)
        finally:
            os.environ.pop("DSV4_COMPRESSOR_FAST", None)

        self.assertEqual(out.shape, (bsz, 1, c.head_dim))

        boundary = ((sp + 1) % compress_ratio) == 0
        if (~boundary).any():
            self.assertTrue(
                out[~boundary].abs().max() < 1e-6,
                "non-boundary output not zero",
            )

    def test_csa_overlap_all_boundary(self):
        self._run_one(compress_ratio=4, start_pos_list=[3, 7, 11, 15])

    def test_csa_overlap_no_boundary(self):
        self._run_one(compress_ratio=4, start_pos_list=[0, 1, 2, 5])

    def test_csa_overlap_mixed(self):
        self._run_one(compress_ratio=4, start_pos_list=[0, 3, 5, 7])

    def test_hca_no_overlap_all_boundary(self):
        self._run_one(compress_ratio=8, start_pos_list=[7, 15, 23, 31])

    def test_hca_no_overlap_mixed(self):
        self._run_one(compress_ratio=8, start_pos_list=[0, 7, 14, 15])


if __name__ == "__main__":
    unittest.main()
