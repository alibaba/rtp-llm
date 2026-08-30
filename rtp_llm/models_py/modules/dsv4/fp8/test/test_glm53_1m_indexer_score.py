"""GLM-5.3 1M CUDA-graph warmup shape for FP8 paged indexer score."""

import torch

from rtp_llm.models_py.modules.dsv4.fp8._indexer_q_quant_triton import (
    indexer_q_fp8_quant_fold,
)
from rtp_llm.models_py.modules.dsv4.fp8._indexer_score import (
    fp8_paged_indexer_score,
    has_fp8_paged_mqa_logits,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


def _run_score_then_topk(batch: int, next_n: int) -> None:
    if not torch.cuda.is_available() or not has_fp8_paged_mqa_logits():
        print("SKIP: CUDA DeepGEMM paged MQA logits required")
        return

    heads, head_dim = 32, 128
    block_size = 32
    max_ctx_len = 1048576 // 4
    blocks_per_request = max_ctx_len // block_size

    q = torch.zeros(
        batch, next_n, heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    weights = torch.ones(
        batch, next_n, heads, dtype=torch.bfloat16, device="cuda"
    )
    q_fp8, w_fold = indexer_q_fp8_quant_fold(q, weights)

    # CUDA-graph warmup initializes every logical entry to physical page zero.
    pool = torch.zeros(block_size, 132, dtype=torch.uint8, device="cuda")
    block_table = torch.zeros(
        batch, blocks_per_request, dtype=torch.int32, device="cuda"
    )
    lengths = torch.full(
        (batch, next_n), max_ctx_len, dtype=torch.int32, device="cuda"
    )
    logits = fp8_paged_indexer_score(
        q_fp8, w_fold.view(batch * next_n, heads), pool, block_table, lengths,
        block_size=block_size, max_ctx_len=max_ctx_len,
    )
    torch.cuda.synchronize()

    output = torch.full(
        (batch * next_n, 512), -1, dtype=torch.int32, device="cuda"
    )
    workspace = torch.empty(1 << 20, dtype=torch.uint8, device="cuda")
    rtp_llm_ops.dsv4_persistent_topk(
        logits, lengths.view(-1), output, workspace, 512, max_ctx_len
    )
    torch.cuda.synchronize()
    assert logits.shape == (batch * next_n, max_ctx_len)


def test_glm53_1m_decode_b8_score_then_topk() -> None:
    _run_score_then_topk(batch=8, next_n=1)


def test_glm53_1m_eagle_verify_b4_score_then_topk() -> None:
    # EAGLE gen_num_per_cycle=3 captures four target-verify tokens per request.
    _run_score_then_topk(batch=4, next_n=4)
