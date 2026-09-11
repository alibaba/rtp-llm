"""Prepare FP4 Indexer inputs on model-owned PPU streams."""

import torch
import torch.nn.functional as F

from rtp_llm.platforms.ppu.kernels.cuda.ppu_fp4_indexer import quantize_q


def prepare_decode_indexer_overlap(
    indexer, x, qr, start_pos, position_ids, metadata, q_producer_stream, streams
):
    """Overlap compressor, weights and Q, joining before the caller scores K.

    The caller binds and releases the Indexer pools. Only Q waits for QR;
    the independent compressor and weights consume the earlier X dependency.
    All auxiliary work is joined before returning, including on exceptions.
    """
    current = torch.cuda.current_stream(x.device)
    query_stream, weight_stream = streams["q"], streams["weights"]
    all_streams = (current, q_producer_stream, query_stream, weight_stream)
    if any(stream.device != x.device for stream in all_streams):
        raise ValueError("Indexer streams must belong to the input device")
    if len({stream.cuda_stream for stream in all_streams}) != 4:
        raise ValueError("Indexer requires distinct producer and preparation streams")
    capturing = torch.cuda.is_current_stream_capturing()
    batch = x.shape[0]
    try:
        for stream in (query_stream, weight_stream):
            stream.wait_stream(current)
            if not capturing:
                x.record_stream(stream)
                metadata.positions.record_stream(stream)
        if not capturing:
            qr.record_stream(query_stream)
        indexer.compressor.forward_decode_vectorized(
            x, start_pos, meta=metadata, position_ids=position_ids
        )
        with torch.cuda.stream(weight_stream):
            weights = F.linear(x.reshape(batch, -1), indexer.weights_proj)
        with torch.cuda.stream(query_stream):
            query_stream.wait_stream(q_producer_stream)
            query_stream.wait_stream(weight_stream)
            if not capturing:
                weights.record_stream(query_stream)
            q = (
                indexer._compute_indexer_q(qr, None, apply_rope=False)
                .reshape(batch, indexer.n_heads, 128)
                .contiguous()
            )
            q, scales, weights = quantize_q(
                q,
                weights,
                indexer.weight_scale,
                torch.view_as_real(indexer.freqs_cis).flatten(-2),
                metadata.positions,
            )
    finally:
        current.wait_stream(query_stream)
        current.wait_stream(weight_stream)
    if not capturing:
        for tensor in (q, scales, weights):
            tensor.record_stream(current)
    return q, scales, weights
