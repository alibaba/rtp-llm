"""Decode preparation on model-owned streams with unchanged math and pools."""

import torch
from rtp_llm.models_py.modules.dsv4.fp8.decode.compute_qkv import (
    decode_compute_kv,
    decode_compute_q_a,
    decode_compute_q_b,
    decode_select_freqs,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.output_proj import decode_output_proj
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import CSA_KV, HCA_KV, SWA_KV


def decode_attention_overlap(attn, x, metadata, streams, *, input_quantized=None):
    """Fork independent preparation, joining before any attention pool read.

    The caller retains the native cache binding and its try/finally lifecycle.
    Streams are prepared at construction and shared across this model's layers.
    """
    bsz, q_len, _ = x.shape
    if q_len not in (1, 2, 3, 4):
        raise ValueError("PPU Decode overlap requires 1/2/3/4 tokens per request")
    start_pos = metadata.start_pos[:bsz]
    position_ids = metadata.position_ids[: bsz * q_len]
    attn._ensure_freqs_cis_bound()
    shared_freqs = getattr(metadata, "rope_freqs_by_source", {})
    if shared_freqs:
        try:
            freqs = shared_freqs[id(attn.freqs_cis)][: bsz * q_len]
        except KeyError as error:
            raise RuntimeError(
                "Decode RoPE source changed; rebuild the metadata Graph"
            ) from error
    else:
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
        qr, kv, q_quantized = None, None, None
        swa_written = False
        # The fused norm indexes every B*Q row and uses its own rotary phase.
        if attn._decode_qkv_projection is not None:
            from rtp_llm.platforms.ppu.kernels.ppu_qkv_norm import normalize_decode_qkv

            raw = (
                attn._decode_qkv_projection(x)
                if input_quantized is None
                else attn._decode_qkv_projection.forward_quantized(
                    *input_quantized
                ).view(bsz, q_len, 1536)
            )
            from rtp_llm.platforms.ppu.modules.linear.fp8_linear import PpuFp8Linear

            if (
                getattr(attn, "_fuse_qkv_quant", False)
                and type(attn.wq_b) is PpuFp8Linear
                and attn.wq_b.quantization == "v2_column"
            ):
                swa_cache = attn._pool_view_3d_fp8(SWA_KV)
                swa_slots = metadata.pool_write_slot_mappings.get(SWA_KV)
                swa_written = (
                    swa_cache is not None
                    and swa_slots is not None
                    and swa_slots.numel() > 0
                )
                qr, kv, q_quantized = normalize_decode_qkv(
                    raw,
                    attn.q_norm,
                    attn.kv_norm,
                    freqs,
                    attn.eps,
                    quantize_q=True,
                    swa_cache=swa_cache,
                    slot_mapping=swa_slots,
                )
            else:
                qr, kv = normalize_decode_qkv(
                    raw, attn.q_norm, attn.kv_norm, freqs, attn.eps
                )
            streams["kv"].wait_stream(current)
            if not capturing:
                kv.record_stream(streams["kv"])
        with torch.cuda.stream(streams["kv"]):
            if kv is None:
                kv = decode_compute_kv(attn, x, freqs)
            if not swa_written:
                attn._decode_write_swa_fp8(kv, bsz, q_len, metadata)
        if attn.compress_ratio:
            with torch.cuda.stream(streams["compressor"]):
                attn._decode_update_compressor(
                    x, bsz, q_len, start_pos, position_ids, metadata
                )
        if qr is None:
            qr = decode_compute_q_a(attn, x)
        if attn.indexer is not None:
            indexer_options = {}
            # Multi-query indexer owns chronological ring updates on its
            # outer stream; only single-query decode uses its inner fork.
            if attn._decode_indexer_streams is None or q_len > 1:
                streams["indexer"].wait_stream(current)
            else:
                indexer_options["q_producer_stream"] = current
            if not capturing:
                qr.record_stream(streams["indexer"])
            with torch.cuda.stream(streams["indexer"]):
                attn._decode_update_indexer(
                    x,
                    qr,
                    bsz,
                    q_len,
                    start_pos,
                    position_ids,
                    metadata,
                    **indexer_options,
                )
        if q_quantized is None:
            q = decode_compute_q_b(attn, qr, freqs)
        else:
            from rtp_llm.models_py.modules.dsv4.fp8.decode.compute_qkv import (
                fused_rmsnorm_rope,
            )

            q = attn.wq_b.forward_quantized(*q_quantized).view(
                bsz, q_len, attn.n_heads, attn.head_dim
            )
            q = fused_rmsnorm_rope(q, None, freqs, attn.rope_head_dim, eps=attn.eps)
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
