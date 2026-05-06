"""Decode attention contract tests that do not require a framework KVCache."""

import os
import sys
import unittest

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from rtp_llm.models_py.modules.dsv4.attention import _sparse_attn
from rtp_llm.models_py.modules.dsv4.decode.decode_attn_metadata import (
    build_decode_metadata,
)
from rtp_llm.models_py.modules.dsv4.decode.sparse_attn_decode_op import (
    SparseAttnV4DecodeOp,
)


class TestAttentionDecodeContracts(unittest.TestCase):
    def test_decode_metadata_left_aligned_window_indices(self):
        meta = build_decode_metadata(
            start_pos=torch.tensor([5], dtype=torch.int32),
            q_len=1,
            window_size=8,
            head_dim=32,
            max_seq_len=64,
            compress_ratios=[0],
            index_topk=4,
            device=torch.device("cpu"),
        )
        self.assertEqual(meta.topk_window_idxs[0, 0].tolist(), [0, 1, 2, 3, 4, 5, -1, -1])

    def test_sparse_decode_op_matches_reference_on_cpu(self):
        torch.manual_seed(11)
        B, S, H, D = 2, 1, 4, 32
        T, K = 16, 8
        q = (torch.randn(B, S, H, D) * 0.1).to(torch.bfloat16)
        kv = (torch.randn(B, T, D) * 0.1).to(torch.bfloat16)
        sink = torch.randn(H, dtype=torch.float32) * 0.1
        topk = torch.randint(0, T, (B, S, K), dtype=torch.int32)
        op = SparseAttnV4DecodeOp(n_heads=H, head_dim=D, softmax_scale=D**-0.5)

        out = op.forward(q, kv, sink, topk)
        ref = _sparse_attn(q, kv, sink, topk, D**-0.5)

        self.assertTrue(torch.equal(out, ref))


if __name__ == "__main__":
    unittest.main()
