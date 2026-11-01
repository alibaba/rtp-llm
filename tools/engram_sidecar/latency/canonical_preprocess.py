"""Experimental owner-side canonicalization using unchanged main quant kernels.

This is not a production checkpoint loader. The caller publishes a generation
only after every chunk validates and completes; exceptions must discard partial
outputs. GPU workspace is bounded by chunk_rows. Persisted output is CPU uint8.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CanonicalChunk:
    row_start: int
    q_bytes: torch.Tensor
    scale_bytes: torch.Tensor


def unpack_scales(scales, groups):
    words = scales.to(torch.int64) & 0xffffffff
    return torch.stack([(words >> (8 * shift)) & 255 for shift in range(4)], -1).reshape(scales.shape[0], -1)[:, :groups].to(torch.uint8).contiguous()


def canonicalize_chunks(raw_weight, raw_scales, *, variant, quant_module,
                        device, chunk_rows=64):
    if variant not in ('legacy', 'v2'):
        raise ValueError('canonical format requires explicit legacy or v2; auto is forbidden')
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError('chunk_rows must be a positive integer')
    if (raw_weight.device.type != 'cpu' or raw_scales.device.type != 'cpu'
            or raw_weight.dtype != torch.uint8 or raw_scales.dtype != torch.uint8
            or raw_weight.ndim != 2 or raw_scales.ndim != 2
            or not raw_weight.is_contiguous() or not raw_scales.is_contiguous()):
        raise ValueError('raw tables must be contiguous CPU uint8 matrices')
    rows, dim = raw_weight.shape
    if rows <= 0 or dim not in (32, 64, 128, 256):
        raise ValueError('experimental table requires R>0 and D in 32/64/128/256')
    groups = dim // 32
    if raw_scales.shape != (rows, groups):
        raise ValueError('raw scale shape must be [R,D/32]')
    device = torch.device(device)
    if device.type != 'cuda' or device.index is None:
        raise ValueError('explicit CUDA device index required')
    with torch.cuda.device(device):
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            # Decode exactly the original lookup's E4M3 bytes and exponent-bit scale.
            raw = raw_weight[start:end].to(device)
            raw_s = raw_scales[start:end].to(device)
            values = raw.view(torch.float8_e4m3fn).float()
            factors = (raw_s.to(torch.int32) << 23).view(torch.float32)
            if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(factors).all()):
                raise ValueError(f'non-finite raw FP8/scale in rows [{start},{end})')
            dequantized = values * factors.repeat_interleave(32, dim=1)
            rounded = dequantized.to(torch.bfloat16)
            if not bool(torch.isfinite(dequantized).all()) or not bool(torch.isfinite(rounded).all()):
                raise ValueError(f'dequantization/BF16 overflow in rows [{start},{end})')
            q = torch.empty_like(rounded, dtype=torch.float8_e4m3fn)
            aligned = (end - start + 3) // 4 * 4
            scales = torch.zeros(((groups + 3) // 4, aligned), dtype=torch.int32, device=device).T[:end-start]
            quant_module.quant(rounded, q, scales, variant == 'v2')
            q_cpu = q.view(torch.uint8).cpu().contiguous()
            s_cpu = unpack_scales(scales, groups).cpu()
            # CPU copies synchronize the selected chunk; owner may safely persist it.
            yield CanonicalChunk(start, q_cpu, s_cpu)


def canonicalize_table(raw_weight, raw_scales, *, variant, quant_module,
                       device, chunk_rows=64):
    """Small-fixture convenience; production loader should stream chunk outputs."""
    q = torch.empty_like(raw_weight)
    s = torch.empty_like(raw_scales)
    for chunk in canonicalize_chunks(raw_weight, raw_scales, variant=variant,
                                     quant_module=quant_module, device=device,
                                     chunk_rows=chunk_rows):
        end = chunk.row_start + chunk.q_bytes.shape[0]
        q[chunk.row_start:end] = chunk.q_bytes
        s[chunk.row_start:end] = chunk.scale_bytes
    return q, s
