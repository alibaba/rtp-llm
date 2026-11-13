"""FlashInfer BF16 SDSA scorer for cold QSA prefill."""

import os

import torch
import torch.nn.functional as F

from rtp_llm.models_py.utils.cutlass import setup_cutlass_import_path


def use_flashinfer_qsa():
    backend = os.environ.get("QWEN4_QSA_BACKEND", "native")
    if backend not in ("native", "flashinfer"):
        raise ValueError("QWEN4_QSA_BACKEND must be native or flashinfer")
    return backend == "flashinfer"


def try_flashinfer_prefill_selection(
    indexer, q, block_keys, positions, workspace_bytes
):
    if not (
        q.is_cuda
        and q.dtype == block_keys.dtype == torch.bfloat16
        and q.device == block_keys.device
        and torch.version.hip is None
        and torch.cuda.get_device_capability(q.device) in ((10, 0), (10, 3), (10, 7))
        and q.shape[-1] == 128
        and q.shape[1] > 0
        and block_keys.shape[1] > 0
        and not torch.cuda.is_current_stream_capturing()
    ):
        return None

    try:
        setup_cutlass_import_path()
        from flashinfer.cute_dsl.availability import is_cute_dsl_experimental_available
        from flashinfer.sdsa import indexer_scores
        from flashinfer.sdsa.shapes import MAX_ROWS, PAGE
    except ImportError:
        return None
    if q.shape[-2] > MAX_ROWS or not is_cute_dsl_experimental_available():
        return None

    batch, seq_len, heads, dim = q.shape
    key_len = block_keys.shape[1]
    pages = (key_len + PAGE - 1) // PAGE
    if pages * PAGE != key_len:
        block_keys = F.pad(block_keys, (0, 0, 0, pages * PAGE - key_len))
    key_cache = block_keys.reshape(batch * pages, PAGE, dim).contiguous()
    block_table = torch.arange(batch * pages, dtype=torch.int32, device=q.device)
    block_table = block_table.view(batch, pages)
    batch_ids = torch.arange(batch, dtype=torch.int32, device=q.device)

    # One SDSA query group holds all of a token's heads, sharing its request's
    # keys. Keep the per-head logits bounded and reuse QSA's causal top-k and tail.
    bytes_per_token = batch * key_len * (heads + 2) * 4
    chunk_size = max(1, min(seq_len, workspace_bytes // bytes_per_token))
    selected = None
    for start in range(0, seq_len, chunk_size):
        stop = min(start + chunk_size, seq_len)
        tokens = stop - start
        requests = batch_ids.repeat_interleave(tokens)
        # SDSA leaves logits beyond its group length unwritten. Score every
        # compressed key here and let the native selector apply causal visibility.
        lengths = torch.full_like(requests, key_len)
        queries = q[:, start:stop].reshape(-1, dim).contiguous()
        logits = torch.empty(
            (queries.shape[0], key_len), dtype=torch.float32, device=q.device
        )
        indexer_scores(
            queries,
            key_cache,
            block_table,
            lengths,
            requests=requests,
            scores=logits,
        )
        scores = logits.view(batch, tokens, heads, key_len).relu_().sum(dim=2)
        scores.mul_(dim**-0.5)
        chunk = indexer._select_causal_scores(scores, positions[start:stop])
        if chunk_size == seq_len:
            return chunk
        if selected is None:
            selected = torch.empty(
                (batch, seq_len, indexer.max_selected),
                device=q.device,
                dtype=torch.int32,
            )
        selected[:, start:stop].copy_(chunk)
        del logits, scores
    return selected
