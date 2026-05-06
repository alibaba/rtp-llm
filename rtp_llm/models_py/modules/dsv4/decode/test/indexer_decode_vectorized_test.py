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
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from rtp_llm.models_py.modules.dsv4.indexer import Indexer
from rtp_llm.utils.model_weight import W


def _as_linear(weight: torch.Tensor, scale: torch.Tensor) -> nn.Linear:
    del scale
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False, dtype=torch.bfloat16)
    linear.weight.data.copy_(weight.to(torch.bfloat16))
    return linear


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
        weights = {
            W.v4_indexer_wq_b_w: torch.randn(
                index_n_heads * index_head_dim, q_lora_rank, dtype=torch.float32
            )
            * 0.05,
            W.v4_indexer_wq_b_s: torch.ones(1, 1, dtype=torch.int32),
            W.v4_indexer_weights_proj_w: torch.randn(
                index_n_heads, dim, dtype=torch.bfloat16
            )
            * 0.05,
            W.v4_indexer_compressor_ape: torch.zeros(
                compress_ratio, 2 * index_head_dim, dtype=torch.float32
            ),
            W.v4_indexer_compressor_wkv: torch.randn(
                2 * index_head_dim, dim, dtype=torch.float32
            )
            * 0.05,
            W.v4_indexer_compressor_wgate: torch.randn(
                2 * index_head_dim, dim, dtype=torch.float32
            )
            * 0.05,
            W.v4_indexer_compressor_norm: torch.ones(
                index_head_dim, dtype=torch.bfloat16
            ),
        }
        with patch(
            "rtp_llm.models_py.modules.dsv4.attention._v4_fp8_linear",
            side_effect=_as_linear,
        ):
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
                layer_weights=weights,
            )
            idx.compressor.configure_kv_cache_shape(max_seq_len // compress_ratio)
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
    del prefill_seed
    fc = torch.complex(
        torch.randn(max_seq_len, idx.rope_head_dim // 2),
        torch.randn(max_seq_len, idx.rope_head_dim // 2),
    )
    idx.freqs_cis = fc
    idx.compressor.freqs_cis = fc


def _move_indexer(idx: Indexer, device: torch.device) -> Indexer:
    idx.to(device)
    idx.weights_proj = idx.weights_proj.to(device)
    idx.freqs_cis = idx.freqs_cis.to(device)
    idx.compressor.ape = idx.compressor.ape.to(device)
    idx.compressor.wkv = idx.compressor.wkv.to(device)
    idx.compressor.wgate = idx.compressor.wgate.to(device)
    idx.compressor.norm.weight = idx.compressor.norm.weight.to(device)
    idx.compressor.freqs_cis = idx.compressor.freqs_cis.to(device)
    return idx


class TestIndexerForwardDecodeVectorized(unittest.TestCase):

    def _run_one(
        self,
        start_pos_list: list[int],
        compress_ratio: int = 4,
        index_topk: int = 4,
        max_seq_len: int = 32,
    ):
        if not torch.cuda.is_available():
            self.skipTest("Indexer decode uses CUDA-only v4_indexer_score")
        device = torch.device("cuda:0")
        torch.manual_seed(11)
        idx_loop = _make_indexer(
            compress_ratio=compress_ratio,
            index_topk=index_topk,
            max_seq_len=max_seq_len,
        )
        _seed(idx_loop, seed=3)
        _bind_freqs_cis(idx_loop, max_seq_len=max_seq_len)
        _move_indexer(idx_loop, device)

        idx_vec = _move_indexer(copy.deepcopy(idx_loop), device)

        bsz = len(start_pos_list)
        x = torch.randn(bsz, 1, idx_loop.dim, device=device, dtype=torch.bfloat16) * 0.1
        qr = (
            torch.randn(
                bsz, 1, idx_loop.q_lora_rank, device=device, dtype=torch.bfloat16
            )
            * 0.1
        )
        sp = torch.tensor(start_pos_list, device=device, dtype=torch.int32)

        out_loop = torch.full(
            (bsz, 1, index_topk), -1, device=device, dtype=torch.int32
        )
        out_vec = torch.full(
            (bsz, 1, index_topk), -1, device=device, dtype=torch.int32
        )
        kv_fixture = torch.randn(
            bsz,
            max_seq_len // compress_ratio,
            idx_loop.head_dim,
            device=device,
            dtype=torch.bfloat16,
        )

        def bind_fixture(self, bsz, is_fresh_prefill, device, dtype):
            del is_fresh_prefill
            self.kv_cache = kv_fixture[:bsz].to(device=device, dtype=dtype).clone()

        idx_loop._bind_kv_cache_from_pool = types.MethodType(bind_fixture, idx_loop)
        idx_vec._bind_kv_cache_from_pool = types.MethodType(bind_fixture, idx_vec)
        idx_loop.compressor.forward_decode = lambda *args, **kwargs: None
        idx_vec.compressor.forward_decode_vectorized = lambda *args, **kwargs: None

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
