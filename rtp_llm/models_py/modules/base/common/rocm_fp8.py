"""ROCm consumers for E4M3FN checkpoint storage (no dense BF16 copy)."""

import torch


def native_fp8_dtype(device):
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
    if arch == "gfx950":
        return torch.float8_e4m3fn
    if arch.startswith("gfx94"):
        return torch.float8_e4m3fnuz
    raise RuntimeError(f"Qwen4 FP8 consumer has not been validated for {arch}")


def convert_fn_weight(weight, scale, dtype):
    """Reinterpret FN as FNUZ with compensating scale; preserve source bytes."""
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError("expected E4M3FN checkpoint storage")
    if dtype == torch.float8_e4m3fn:
        return weight, scale
    if dtype != torch.float8_e4m3fnuz:
        raise ValueError("expected native E4M3 dtype")
    # FN negative zero is FNUZ NaN; do not mutate shared checkpoint storage.
    raw = weight.view(torch.uint8).clone()
    raw.masked_fill_(raw == 128, 0)
    return raw.view(dtype), scale * 2.0


def quantize_input(rows, *, scale=None, output=None):
    """Graph-capturable native FP8 quantization, including zero padded rows."""
    dtype = output.dtype if output is not None else native_fp8_dtype(rows.device)
    maximum = torch.finfo(dtype).max
    if scale is None:
        scale = rows.float().abs().amax().clamp_min(1e-12).reshape(1) / maximum
    quantized = (rows.float() / scale).clamp(-maximum, maximum).to(dtype)
    if output is not None:
        output.zero_()
        output[: rows.shape[0]].copy_(quantized)
        quantized = output
    return quantized, scale


def fn_decode_table(device):
    # Decode on CPU once: gfx942 need not support native FN conversion kernels.
    return (
        torch.arange(256, dtype=torch.uint8, device="cpu")
        .view(torch.float8_e4m3fn)
        .to(dtype=torch.bfloat16, device="cpu")
        .to(device)
    )


def vocab_rows(tokens, weight, scale, table, text_mask=None):
    valid = (tokens >= 0) & (tokens < weight.shape[0])
    if text_mask is not None and text_mask.numel():
        valid = valid & text_mask.bool()
    indices = tokens.clamp(0, weight.shape[0] - 1).long()
    raw = weight.view(torch.uint8).index_select(0, indices)
    values = table.index_select(0, raw.reshape(-1).long()).reshape(raw.shape)
    values = values * scale.to(torch.bfloat16)
    return torch.where(valid[:, None], values, 0)
