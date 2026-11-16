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


def try_flashinfer_decode_selection(q, cache, table, lengths):
    if not (
        q.is_cuda
        and q.dtype == cache.dtype == torch.bfloat16
        and torch.version.hip is None
        and torch.cuda.get_device_capability(q.device) in ((10, 0), (10, 3), (10, 7))
        and q.shape[1:] == (8, 128)
        and cache.shape[1:] == (32, 128)
        and table.stride(1) == 1
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

    from rtp_llm.models_py.kernels.cuda.qsa.qsa_topk_triton import (
        expand_qsa_block_ids,
        mask_qsa_decode_pages,
    )

    # Page-table capacity stays fixed while visible lengths change under replay.
    rows, heads, dim = q.shape
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
            h_i=heads, s_q=1, page_size=128, has_block_lens=False, device=q.device.index
        )
    except NotImplementedError:
        return None

    scores = torch.empty((rows, cols), device=q.device, dtype=torch.float32)
    counts = torch.empty_like(lengths)
    block_ids = torch.empty((rows, 512), device=q.device, dtype=torch.int32)
    stream = torch.cuda.current_stream(q.device).cuda_stream
    run_qsa_score_paged(
        score_recipe,
        q,
        cache,
        table,
        lengths,
        scores,
        counts,
        None,
        top_k=512,
        scale=dim**-0.5,
        stream=stream,
    )
    # FROST treats physical page zero as valid. Mask invalid pages before TopK;
    # expansion also drops masked IDs when fewer than 512 valid blocks remain.
    mask_qsa_decode_pages(scores, table, counts, cache.shape[0])
    topk.run_qsa_topk_decode(topk_recipe, scores, counts, block_ids, stream=stream)
    return expand_qsa_block_ids(block_ids, lengths, scores)
