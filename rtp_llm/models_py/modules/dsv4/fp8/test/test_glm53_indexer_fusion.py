"""GLM NoPE Hadamard quantization and request-local score coordinates."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from rtp_llm.models_py.modules.dsv4.fp8._indexer_hadamard_quant_triton import (
    indexer_hadamard_quant_fold,
)
from rtp_llm.models_py.modules.dsv4.fp8._indexer_q_quant_triton import (
    indexer_q_fp8_quant_fold,
)
from rtp_llm.models_py.modules.dsv4.fp8.indexer import IndexerFP8
from rtp_llm.models_py.triton_kernels.sparse_mla.fused_prefill_rope_hadamard import (
    hadamard_transform_128,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("tokens", [0, 1, 7, 48, 4097])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fp8_bytes_and_fold_exact(tokens, dtype):
    torch.manual_seed(9305)
    q = torch.randn(1, tokens, 32, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(1, tokens, 32, device="cuda", dtype=dtype)
    if tokens:
        q[:, 0, 0] = 0
    expected = indexer_q_fp8_quant_fold(hadamard_transform_128(q), w)
    actual = indexer_hadamard_quant_fold(q, w)
    assert torch.equal(actual[0].view(torch.uint8), expected[0].view(torch.uint8))
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


@pytest.mark.parametrize("chunk", [7, 64, 16384])
def test_request_chunks_preserve_ragged_topk(chunk):
    torch.manual_seed(9306)
    q_lengths = [7, 17, 41]
    k_lengths = [0, 19, 4103]
    m = sum(q_lengths)
    n = sum(k_lengths)
    q = (torch.randn(m, 32, 128, device="cuda") * 0.2).to(torch.float8_e4m3fn)
    w = torch.randn(m, 32, device="cuda") * 0.03
    key = (torch.randn(n, 128, device="cuda") * 0.2).to(torch.float8_e4m3fn)
    key_scale = torch.rand(n, device="cuda") + 0.5
    starts = []
    ends = []
    segments = []
    q_offset = k_offset = 0
    for q_len, k_len in zip(q_lengths, k_lengths):
        starts.append(torch.full((q_len,), k_offset, device="cuda", dtype=torch.int32))
        ends.append(
            torch.linspace(0, k_len, q_len, device="cuda").to(torch.int32) + k_offset
        )
        segments.append((q_offset, q_offset + q_len, k_offset, k_offset + k_len))
        q_offset += q_len
        k_offset += k_len
    ks = torch.cat(starts)
    ke = torch.cat(ends)
    meta = SimpleNamespace(M=m, T=n, ks=ks, ke=ke, score_segments=None)
    op = IndexerFP8.__new__(IndexerFP8)
    torch.nn.Module.__init__(op)
    op.index_topk = 512
    op.compress_ratio = 4
    op.prefill_topk_backend = "topk_v3_tie_break"
    with patch.dict(
        os.environ,
        {
            "DSV4_INDEXER_TOPK_CANONICALIZE": "1",
            "DSV4_FP8_INDEXER_SCORE_CHUNK_ROWS": str(chunk),
        },
    ):
        ref = op._prefill_score_topk(q, w, key, key_scale, meta)
        meta.score_segments = tuple(segments)
        meta.score_relative_ks = torch.zeros_like(ks)
        meta.score_relative_ke = ke - ks
        actual = op._prefill_score_topk(q, w, key, key_scale, meta)
    assert torch.equal(ref, actual)
    assert torch.all(actual[:7] == -1)
    assert torch.all(actual[24:][actual[24:] >= 0] < 4103)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
