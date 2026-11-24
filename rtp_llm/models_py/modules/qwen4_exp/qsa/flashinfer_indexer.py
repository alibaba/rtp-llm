"""FlashInfer QSA selection from contiguous prefill or paged decode keys."""

import os

import torch
import torch.nn.functional as F

from rtp_llm.models_py.utils.cutlass import setup_cutlass_import_path


def use_flashinfer_qsa():
    backend = os.environ.get("QWEN4_QSA_BACKEND", "native")
    if backend not in ("native", "flashinfer"):
        raise ValueError("QWEN4_QSA_BACKEND must be native or flashinfer")
    return backend == "flashinfer"


def use_flashinfer_indexer():
    backend = os.environ.get("QWEN4_QSA_INDEXER_BACKEND")
    if backend is None:
        return use_flashinfer_qsa()
    if backend not in ("native", "flashinfer"):
        raise ValueError("QWEN4_QSA_INDEXER_BACKEND must be native or flashinfer")
    return backend == "flashinfer"


def use_fused_indexer_prep():
    return (
        os.environ.get("QWEN4_QSA_FUSED_INDEXER_PREP") == "1"
        and use_flashinfer_indexer()
    )


def select_flashinfer_prefill_blocks(q, block_keys):
    """Run the FROST scorer and radix TopK used by qsa_select_forward.

    QWEN4_QSA_INDEXER_CHUNK_SIZE controls query tokens per score window.
    Zero processes all queries in one window.
    """
    chunk_size = int(os.environ.get("QWEN4_QSA_INDEXER_CHUNK_SIZE", "4096"))
    if chunk_size < 0 or chunk_size % 4:
        raise ValueError(
            "QWEN4_QSA_INDEXER_CHUNK_SIZE must be zero or a positive multiple of 4"
        )

    setup_cutlass_import_path()
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.indexer_forward import (
        indexer_forward_top_k_wrapper,
    )
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.QSA.selection import (
        _constant_weights,
    )

    tokens, heads, dim = q.shape
    padded_tokens = tokens
    if 0 < chunk_size < tokens and tokens % 4:
        # FROST requires ratio-aligned query windows. The added key is only
        # causal for a padding row, so it never changes the real selection.
        padded_tokens = (tokens + 3) // 4 * 4
        q = F.pad(q, (0, 0, 0, 0, 0, padded_tokens - tokens))
        if block_keys.shape[1] < padded_tokens // 4:
            block_keys = F.pad(block_keys, (0, 0, 0, 1))
    weights = _constant_weights(padded_tokens * heads, q.device).view(
        1, padded_tokens, heads
    )
    result = indexer_forward_top_k_wrapper(
        q.unsqueeze(0),
        block_keys.unsqueeze(2),
        weights,
        top_k=512,
        ratio=4,
        qhead_per_kv_head=heads,
        sm_scale=dim**-0.5,
        microbatch_rows=chunk_size,
        topk_indices_global=False,
        return_softmax=False,
        deterministic=True,
    )
    return result["indices"][0, :tokens]


def try_flashinfer_prefill_selection(q, block_keys):
    if not (
        q.is_cuda
        and q.dtype == block_keys.dtype == torch.bfloat16
        and q.device == block_keys.device
        and torch.version.hip is None
        and torch.cuda.get_device_capability(q.device) in ((10, 0), (10, 3), (10, 7))
        and q.shape[0] == 1
        and q.shape[2:] == (8, 128)
        and block_keys.shape[1] > 0
        and not torch.cuda.is_current_stream_capturing()
    ):
        return None
    from rtp_llm.models_py.kernels.cuda.qsa.qsa_topk_triton import expand_qsa_block_ids

    block_ids = select_flashinfer_prefill_blocks(q[0], block_keys)
    return expand_qsa_block_ids(block_ids).unsqueeze(0)


def select_flashinfer_prefix_blocks(
    q, cache, table, cu, prefixes, kv_lens, max_q, max_k
):
    """Select logical compressed block IDs from the prefix and the new pages."""
    setup_cutlass_import_path()
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.indexer_forward import (
        indexer_forward_top_k_wrapper,
    )
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.QSA.selection import (
        _constant_weights,
    )

    rows, heads, dim = q.shape
    weights = _constant_weights(rows * heads, q.device).view(rows, heads)
    return indexer_forward_top_k_wrapper(
        q,
        cache.unsqueeze(2),
        weights,
        top_k=512,
        ratio=4,
        qhead_per_kv_head=heads,
        sm_scale=dim**-0.5,
        cu_seqlens_q=cu,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        q_causal_offsets=prefixes,
        microbatch_rows=0,
        topk_indices_global=False,
        return_softmax=False,
        deterministic=True,
        block_table=table,
        kv_lens=kv_lens,
    )["indices"]


def prepare_flashinfer_decode_selection(q, cache, table, *, query_len=1):
    if not (
        q.is_cuda
        and q.dtype == cache.dtype == torch.bfloat16
        and torch.version.hip is None
        and torch.cuda.get_device_capability(q.device) in ((10, 0), (10, 3), (10, 7))
        and q.shape[1:] == (8, 128)
        and cache.shape[1:] == (32, 128)
        and table.stride(1) == 1
        and 1 <= query_len <= 4
    ):
        return None

    setup_cutlass_import_path()
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.gated_attention_block.kernels.qsa_score_paged import (
        compile_qsa_score_paged,
        run_qsa_score_paged,
    )
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.gated_attention_block.qsa_topk import (
        qsa_topk_decode_kernel_module,
    )

    # Page-table capacity stays fixed while visible lengths change under replay.
    rows, heads, _ = q.shape
    cols = table.shape[1] * 32
    try:
        with torch.cuda.device(q.device):
            topk = qsa_topk_decode_kernel_module()
            cluster, threads = topk.default_knobs_for(rows, cols)
            capacity = topk.cols_cap_for(cols, cluster, threads)
            if capacity // (cluster * threads) > topk.MAX_EPT:
                return None
            topk_recipe = topk.compile_qsa_topk_decode(
                cols_cap=capacity,
                top_k=512,
                cluster_ctas=cluster,
                threads_per_cta=threads,
            )
        score_recipe = compile_qsa_score_paged(
            h_i=heads,
            s_q=query_len,
            page_size=128,
            has_block_lens=False,
            device=q.device.index,
        )
    except NotImplementedError:
        return None
    return run_qsa_score_paged, score_recipe, topk, topk_recipe


def try_flashinfer_decode_selection(
    q, cache, table, lengths, *, recipe=None, kv_lengths=None
):
    """Select paged blocks for one query or a contiguous query window per request."""
    query_len = lengths.shape[1] if lengths.ndim == 2 else 1
    if recipe is None:
        recipe = prepare_flashinfer_decode_selection(
            q, cache, table, query_len=query_len
        )
        if recipe is None:
            return None

    from rtp_llm.models_py.kernels.cuda.qsa.qsa_topk_triton import (
        expand_qsa_block_ids,
        mask_qsa_decode_pages,
    )

    run_qsa_score_paged, score_recipe, topk, topk_recipe = recipe
    # MTP draft windows can have trailing padding. Anchor the causal window at
    # the first query and keep the actual per-query lengths for output masking.
    lengths = lengths.reshape(-1)
    if kv_lengths is None:
        kv_lengths = (
            lengths[::query_len] + (query_len - 1) if query_len > 1 else lengths
        )
    rows, _, dim = q.shape
    cols = table.shape[1] * 32
    scores = torch.empty((rows, cols), device=q.device, dtype=torch.float32)
    counts = torch.empty_like(lengths)
    block_ids = torch.empty((rows, 512), device=q.device, dtype=torch.int32)
    stream = torch.cuda.current_stream(q.device).cuda_stream
    run_qsa_score_paged(
        score_recipe,
        q,
        cache,
        table,
        kv_lengths,
        scores,
        counts,
        None,
        top_k=512,
        scale=dim**-0.5,
        stream=stream,
    )
    # FROST treats physical page zero as valid. Mask invalid pages before TopK;
    # expansion also drops masked IDs when fewer than 512 valid blocks remain.
    mask_qsa_decode_pages(scores, table, counts, cache.shape[0], query_len=query_len)
    topk.run_qsa_topk_decode(topk_recipe, scores, counts, block_ids, stream=stream)
    return expand_qsa_block_ids(block_ids, lengths, scores)
