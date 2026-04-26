"""Stage 3B: Indexer.forward_decode_vectorized vs forward_decode.

Verifies the vectorized (CUDA-graph-friendly) variant produces the
same topk indices as the loop variant for the leading K positions of
each request (where K = min(index_topk, compressed_len[r])).

The loop variant truncates topk to ``min(K, T_r)`` per request; the
vectorized variant always returns ``index_topk`` slots, with the leading
valid prefix matching the loop variant's output. This test compares
only that valid prefix.

Tests on CPU only — graph capture itself is a SM100_ARM smoke concern.
"""

import copy
import os
import sys
import unittest

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from rtp_llm.models_py.modules.dsv4.indexer import Indexer


def _make_indexer(
    compress_ratio: int = 4,
    dim: int = 64,
    q_lora_rank: int = 32,
    index_n_heads: int = 2,
    index_head_dim: int = 16,
    rope_head_dim: int = 8,
    index_topk: int = 4,
    max_batch_size: int = 4,
    max_seq_len: int = 32,
) -> Indexer:
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        idx = Indexer(
            dim=dim,
            q_lora_rank=q_lora_rank,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            rope_head_dim=rope_head_dim,
            index_topk=index_topk,
            compress_ratio=compress_ratio,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            norm_eps=1e-6,
            weights=None,
            prefix="",
        )
    finally:
        torch.set_default_dtype(prev_dtype)
    return idx


def _seed(idx: Indexer, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    for _, p in idx.named_parameters():
        if p.dtype.is_floating_point:
            try:
                p.data = torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.05
            except (NotImplementedError, RuntimeError):
                tmp = torch.randn(p.shape, generator=g, dtype=torch.float32) * 0.05
                p.data = tmp.to(p.dtype)
    idx.compressor.norm.weight.data.fill_(1.0)


def _bind_freqs_cis(idx: Indexer, max_seq_len: int = 32, prefill_seed: int = 17):
    fc = torch.complex(
        torch.randn(max_seq_len, idx.rope_head_dim // 2),
        torch.randn(max_seq_len, idx.rope_head_dim // 2),
    )
    idx.freqs_cis = fc
    idx.compressor.kv_cache = (
        idx.kv_cache
    )  # bind compressor's external kv_cache to indexer's
    idx.compressor.freqs_cis = fc
    # Pre-populate kv_cache with non-zero values so the score topk has no ties
    # at uninitialized slots. In production these slots are filled by previous
    # decode steps; in unit-test isolation we have to fake it.
    g = torch.Generator().manual_seed(prefill_seed)
    idx.kv_cache.copy_(
        torch.randn(idx.kv_cache.shape, generator=g, dtype=idx.kv_cache.dtype) * 0.5
    )


class TestIndexerForwardDecodeVectorized(unittest.TestCase):

    def _run_one(
        self,
        start_pos_list: list[int],
        compress_ratio: int = 4,
        index_topk: int = 4,
        max_seq_len: int = 32,
    ):
        torch.manual_seed(11)
        idx_loop = _make_indexer(
            compress_ratio=compress_ratio,
            index_topk=index_topk,
            max_seq_len=max_seq_len,
        )
        _seed(idx_loop, seed=3)
        _bind_freqs_cis(idx_loop, max_seq_len=max_seq_len)

        idx_vec = copy.deepcopy(idx_loop)

        bsz = len(start_pos_list)
        x = torch.randn(bsz, 1, idx_loop.dim, dtype=torch.bfloat16) * 0.1
        qr = torch.randn(bsz, 1, idx_loop.q_lora_rank, dtype=torch.bfloat16) * 0.1
        sp = torch.tensor(start_pos_list, dtype=torch.int32)

        out_loop = torch.full((bsz, 1, index_topk), -1, dtype=torch.int32)
        out_vec = torch.full((bsz, 1, index_topk), -1, dtype=torch.int32)

        with torch.inference_mode():
            idx_loop.forward_decode(x, qr, sp, out_loop)
            idx_vec.forward_decode_vectorized(x, qr, sp, out_vec)

        # For each request, valid prefix length = min(index_topk, compressed_len[r])
        # where compressed_len[r] = (start_pos[r] + 1) // ratio.
        for r in range(bsz):
            T_r = (start_pos_list[r] + 1) // compress_ratio
            k_valid = min(index_topk, T_r)
            if k_valid == 0:
                # Both should be all -1.
                self.assertTrue(
                    bool((out_loop[r] == -1).all()),
                    f"loop[{r}] should be all -1 (T_r=0)",
                )
                self.assertTrue(
                    bool((out_vec[r] == -1).all()), f"vec[{r}] should be all -1 (T_r=0)"
                )
                continue
            loop_idxs = out_loop[r, 0, :k_valid].sort()[0]
            vec_idxs = out_vec[r, 0, :k_valid].sort()[0]
            self.assertTrue(
                torch.equal(loop_idxs, vec_idxs),
                f"req r={r} (sp={start_pos_list[r]}, T_r={T_r}, "
                f"k_valid={k_valid}): loop={loop_idxs.tolist()} "
                f"vec={vec_idxs.tolist()}",
            )

    def test_csa_4_short(self):
        """ratio=4, sp=[3, 7, 11, 15]: T_r = [1, 2, 3, 4]. K=4."""
        self._run_one(start_pos_list=[3, 7, 11, 15])

    def test_csa_4_longer_than_topk(self):
        """ratio=4, sp=[19, 23, 27, 31]: T_r = [5, 6, 7, 8] all > K=4."""
        self._run_one(start_pos_list=[19, 23, 27, 31])

    def test_csa_4_includes_zero(self):
        """ratio=4, sp=[0, 3, 7, 15]: T_r = [0, 1, 2, 4]; r=0 has no compressed
        entries yet, output must be all -1 for that req."""
        self._run_one(start_pos_list=[0, 3, 7, 15])


if __name__ == "__main__":
    unittest.main()
