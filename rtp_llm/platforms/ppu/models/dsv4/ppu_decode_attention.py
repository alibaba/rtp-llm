"""Decode preparation on model-owned streams with unchanged math and pools."""

import torch
from rtp_llm.models_py.modules.dsv4.fp8.decode.compute_qkv import (
    decode_compute_kv,
    decode_compute_q_a,
    decode_compute_q_b,
    decode_select_freqs,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.output_proj import decode_output_proj
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import CSA_KV, HCA_KV


def decode_attention_overlap(attn, x, metadata, streams):
    """Fork independent preparation, joining before any attention pool read.

    The caller retains the native cache binding and its try/finally lifecycle.
    Streams are prepared at construction and shared across this model's layers.
    """
    bsz, q_len, _ = x.shape
    if q_len != 1:
        raise ValueError("PPU Decode overlap requires one token per request")
    start_pos = metadata.start_pos[:bsz]
    position_ids = metadata.position_ids[:bsz]
    attn._ensure_freqs_cis_bound()
    freqs = decode_select_freqs(attn, position_ids)
    current = torch.cuda.current_stream(x.device)
    capturing = torch.cuda.is_current_stream_capturing()
    active = [streams["kv"]]
    if attn.compress_ratio:
        active.append(streams["compressor"])
    if attn.indexer is not None:
        active.append(streams["indexer"])
    for stream in active:
        stream.wait_stream(current)
        if not capturing:
            for tensor in (x, freqs, start_pos, position_ids):
                tensor.record_stream(stream)
    try:
        with torch.cuda.stream(streams["kv"]):
            kv = decode_compute_kv(attn, x, freqs)
            attn._decode_write_swa_fp8(kv, bsz, q_len, metadata)
        if attn.compress_ratio:
            with torch.cuda.stream(streams["compressor"]):
                attn._decode_update_compressor(
                    x, bsz, q_len, start_pos, position_ids, metadata
                )
        qr = decode_compute_q_a(attn, x)
        if attn.indexer is not None:
            streams["indexer"].wait_stream(current)
            if not capturing:
                qr.record_stream(streams["indexer"])
            with torch.cuda.stream(streams["indexer"]):
                attn._decode_update_indexer(
                    x, qr, bsz, q_len, start_pos, position_ids, metadata
                )
        q = decode_compute_q_b(attn, qr, freqs)
    finally:
        # Also join partially submitted work before the caller unbinds pools
        # or releases intermediates after an exception.
        for stream in active:
            current.wait_stream(stream)

    if attn.compress_ratio == 0:
        output = attn._forward_decode_swa_only(q, bsz, q_len, metadata)
    else:
        if attn.compress_ratio == 4:
            indices = metadata.topk_buffer_compressed[:bsz]
            tag = CSA_KV
        elif attn.compress_ratio == 128:
            indices = metadata.topk_total_by_ratio[128][:bsz, :, attn.window_size :]
            tag = HCA_KV
        else:
            raise ValueError("Unsupported PPU Decode compression ratio")
        output = attn._forward_decode_compressed(
            q, indices, bsz, q_len, metadata, cmp_attn_type=tag
        )
    return decode_output_proj(attn, output, freqs, bsz, q_len)
