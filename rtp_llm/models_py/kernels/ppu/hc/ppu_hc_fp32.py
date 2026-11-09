"""Reference FP32 HC projection for DSpARK precision validation."""

import os

import torch
import torch.nn.functional as F

_PREFILL_BF16_RMS = os.getenv("DSV4_PPU_HC_RMS_BF16_INPUT", "0") == "1"
if _PREFILL_BF16_RMS:
    if os.getenv("ROLE_TYPE") != "PREFILL":
        raise RuntimeError("HC RMS BF16 input requires PREFILL worker role")
    from rtp_llm.models_py.kernels.ppu.runtime import require_ppu_op

    require_ppu_op("rtp_llm_ppu", "hc_rms_square_bf16_prefill")
    require_ppu_op("rtp_llm_ppu", "hc_rms_tail_bf16_prefill")


def fp32_hc_prenorm_partials(
    x, weight, *, rowwise=False, row_gemv_stream_pool=None, prepare_q3=False
):
    if x.ndim != 2 or x.shape[1] != 16384 or weight.shape != (24, 16384):
        raise ValueError("FP32 HC requires [M,16384] input and [24,16384] weight")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.float32:
        raise TypeError("FP32 HC requires BF16 input and FP32 weight")
    if any(
        not t.is_cuda or t.device != x.device or not t.is_contiguous()
        for t in (x, weight)
    ):
        raise ValueError("FP32 HC inputs must be contiguous on one PPU")
    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("FP32 HC requires TF32 disabled")
    if (
        not rowwise
        and 1024 <= x.shape[0] <= 32768
        and x.data_ptr() % 8 == 0
        and torch.cuda.get_device_name(x.device) == "ZW-M890P"
        and not torch.cuda.is_current_stream_capturing()
    ):
        try:
            xf, squares = torch.ops.rtp_llm_ppu.hc_prepare(x)
        except RuntimeError as exc:
            # Only the probe's prelaunch geometry rejection permits fallback.
            if not str(exc).startswith("Unexpected live Torch reduction geometry"):
                raise
        else:
            return F.linear(xf, weight).unsqueeze(0), squares.unsqueeze(0)
    prepared = None
    if (
        prepare_q3
        and rowwise
        and tuple(x.shape) == (24, 16384)
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.is_cuda
        and torch.cuda.get_device_name(x.device) == "ZW-M890P"
    ):
        xf, prepared = torch.ops.hc_prepare_q3_probe.prepare(x, False)
    else:
        xf = x.float()
    if rowwise and x.shape[0]:
        from rtp_llm.models_py.kernels.ppu.runtime import try_fp32_rowwise_linear

        dots = try_fp32_rowwise_linear(xf, weight, row_gemv_stream_pool, lanes=4)
        if dots is None:
            dots = torch.cat([F.linear(row[None], weight) for row in xf])
        if (
            xf.shape[0] in (3, 6, 8, 24, 48)
            and torch.cuda.get_device_name(xf.device) == "ZW-M890P"
        ):
            squares = (
                prepared
                if prepared is not None
                else torch.ops.rtp_llm_ppu.hc_row_reduce(xf.square(), False)
            ).squeeze(-1)
        else:
            squares = torch.cat([row[None].square().sum(-1) for row in xf])
    else:
        dots = F.linear(xf, weight)
        squares = xf.square().sum(-1)
    return dots.unsqueeze(0), squares.unsqueeze(0)


def fp32_hc_prenorm_gemm(x, weight, out, squares, requested_splits=None):
    del requested_splits
    dots, sums = fp32_hc_prenorm_partials(x, weight)
    if out.shape != dots.shape or squares.shape != sums.shape:
        raise ValueError("FP32 HC outputs must have shapes [1,M,24] and [1,M]")
    if any(t.dtype != torch.float32 or t.device != x.device for t in (out, squares)):
        raise ValueError("FP32 HC outputs must be FP32 on the input device")
    out.copy_(dots)
    squares.copy_(sums)
    return out


def fp32_hc_rms_norm(x, weight, eps, *, rowwise=False, prepare_q3=False):
    if (
        not rowwise
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.shape[0] in (16384, 32768)
        and x.shape[1] == 4096
        and x.is_contiguous()
        and x.is_cuda
        and weight.shape == (4096,)
        and weight.dtype in (torch.bfloat16, torch.float32)
        and weight.device == x.device
        and weight.is_contiguous()
        and torch.cuda.get_device_name(x.device) == "ZW-M890P"
    ):
        # Preserve the reference reduction; fuse only its two rounded FP32
        # multiplies and the final BF16 store.
        if _PREFILL_BF16_RMS and not torch.cuda.is_current_stream_capturing():
            squares = torch.ops.rtp_llm_ppu.hc_rms_square_bf16_prefill(x)
            inv = torch.rsqrt(squares.mean(-1, keepdim=True) + eps)
            return torch.ops.rtp_llm_ppu.hc_rms_tail_bf16_prefill(x, inv, weight)
        xf = x.float()
        inv = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
        return torch.ops.rtp_llm_ppu.hc_rms_tail(xf, inv, weight)
    if rowwise and x.numel():
        # Native verification normalizes each committed/proposed token alone.
        # PPU's batched reduction can round differently at BF16 midpoints.
        rows = x.reshape(-1, x.shape[-1])
        if (
            x.dtype == torch.bfloat16
            and rows.shape[0] in (3, 6, 8, 24, 48)
            and rows.shape[1] == 4096
            and rows.is_contiguous()
            and rows.is_cuda
            and weight.shape == (4096,)
            and weight.dtype in (torch.bfloat16, torch.float32)
            and weight.device == x.device
            and weight.is_contiguous()
            and torch.cuda.get_device_name(x.device) == "ZW-M890P"
        ):
            prepared_inv = None
            if (
                prepare_q3
                and tuple(rows.shape) == (24, 4096)
                and rows.dtype == torch.bfloat16
                and rows.is_contiguous()
                and rows.is_cuda
                and eps == 1e-6
                and torch.cuda.get_device_name(rows.device) == "ZW-M890P"
            ):
                xf, mean, prepared_inv = torch.ops.hc_prepare_q3_v3_probe.prepare_rms(
                    rows, eps
                )
            else:
                xf = rows.float()
                mean = torch.ops.rtp_llm_ppu.hc_row_reduce(xf.square(), True)
            inv = prepared_inv if prepared_inv is not None else torch.rsqrt(mean + eps)
            return torch.ops.rtp_llm_ppu.hc_rms_tail(xf, inv, weight).view_as(x)
        if (
            rows.shape[0] in (3, 6)
            and rows.shape[1] == 4096
            and rows.is_contiguous()
            and rows.is_cuda
            and torch.cuda.get_device_name(rows.device) == "ZW-M890P"
        ):
            xf = rows.float()
            mean = torch.ops.rtp_llm_ppu.hc_row_reduce(xf.square(), True)
            return (weight * (xf * torch.rsqrt(mean + eps))).to(x.dtype).view_as(x)
        return torch.cat(
            [fp32_hc_rms_norm(row[None], weight, eps) for row in rows]
        ).view_as(x)
    xf = x.float()
    return (weight * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps))).to(
        x.dtype
    )
