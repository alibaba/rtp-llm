"""Stage 3B: Indexer.forward_decode_vectorized correctness.

Verifies the vectorized (CUDA-graph-friendly) variant:
  * Returns valid topk indices (>= 0) for requests with compressed_len > 0.
  * Returns all -1 for requests where compressed_len == 0.
  * Indices are within range [0, compressed_len).

Tests on CPU only — graph capture itself is a SM100_ARM smoke concern.
For FAST vs REF equivalence, see indexer_decode_fast_test.py.
"""

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


def _bind_state(idx: Indexer, max_seq_len: int = 32, prefill_seed: int = 17):
    fc = torch.complex(
        torch.randn(max_seq_len, idx.rope_head_dim // 2),
        torch.randn(max_seq_len, idx.rope_head_dim // 2),
    )
    idx.freqs_cis = fc
    idx.compressor.freqs_cis = fc
    idx.compressor.configure_kv_cache_shape(max_seq_len // idx.compress_ratio)
    bsz = idx.max_batch_size
    device = next(idx.parameters()).device
    idx.compressor._bind_state_from_pool(bsz, is_fresh_prefill=True, device=device)
    idx.compressor._bind_kv_cache_from_pool(
        bsz, is_fresh_prefill=True, device=device, dtype=torch.bfloat16
    )
    idx._bind_kv_cache_from_pool(
        bsz, is_fresh_prefill=True, device=device, dtype=torch.bfloat16
    )
    g = torch.Generator().manual_seed(prefill_seed)
    idx.kv_cache.copy_(
        torch.randn(idx.kv_cache.shape, generator=g, dtype=idx.kv_cache.dtype) * 0.5
    )
    idx.compressor.kv_cache = idx.kv_cache


class TestIndexerForwardDecodeVectorized(unittest.TestCase):

    def _run_one(
        self,
        start_pos_list: list[int],
        compress_ratio: int = 4,
        index_topk: int = 4,
        max_seq_len: int = 32,
    ):
        torch.manual_seed(11)
        idx = _make_indexer(
            compress_ratio=compress_ratio,
            index_topk=index_topk,
            max_seq_len=max_seq_len,
        )
        _seed(idx, seed=3)
        _bind_state(idx, max_seq_len=max_seq_len)

        bsz = len(start_pos_list)
        x = torch.randn(bsz, 1, idx.dim, dtype=torch.bfloat16) * 0.1
        qr = torch.randn(bsz, 1, idx.q_lora_rank, dtype=torch.bfloat16) * 0.1
        sp = torch.tensor(start_pos_list, dtype=torch.int32)

        out = torch.full((bsz, 1, index_topk), -1, dtype=torch.int32)

        os.environ["DSV4_INDEXER_FAST"] = "0"
        try:
            with torch.inference_mode():
                idx.forward_decode_vectorized(x, qr, sp, out)
        finally:
            os.environ.pop("DSV4_INDEXER_FAST", None)

        for r in range(bsz):
            T_r = (start_pos_list[r] + 1) // compress_ratio
            k_valid = min(index_topk, T_r)
            if k_valid == 0:
                self.assertTrue(
                    bool((out[r] == -1).all()),
                    f"req[{r}] should be all -1 (T_r=0)",
                )
                continue
            valid_idxs = out[r, 0, :k_valid]
            self.assertTrue(
                bool((valid_idxs >= 0).all()),
                f"req[{r}] valid prefix has negative index: {valid_idxs.tolist()}",
            )
            self.assertTrue(
                bool((valid_idxs < T_r).all()),
                f"req[{r}] index out of range [0, {T_r}): {valid_idxs.tolist()}",
            )

    def test_csa_4_short(self):
        self._run_one(start_pos_list=[3, 7, 11, 15])

    def test_csa_4_longer_than_topk(self):
        self._run_one(start_pos_list=[19, 23, 27, 31])

    def test_csa_4_includes_zero(self):
        self._run_one(start_pos_list=[0, 3, 7, 15])


if __name__ == "__main__":
    unittest.main()
