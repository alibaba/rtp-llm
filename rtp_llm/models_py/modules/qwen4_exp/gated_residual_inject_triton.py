"""Graph-safe Qwen4 gated residual writeback."""

import torch
import triton
import triton.language as tl


@triton.jit
def _inject_kernel(
    hyper_ptr,
    sublayer_ptr,
    gate_ptr,
    output_ptr,
    TOTAL: tl.constexpr,
    HIDDEN: tl.constexpr,
    BRANCHES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < TOTAL
    row = offset // (BRANCHES * HIDDEN)
    column = offset % (BRANCHES * HIDDEN)
    sublayer_offset = row * HIDDEN + column % HIDDEN
    gate_offset = row * BRANCHES + column // HIDDEN
    hyper = tl.load(hyper_ptr + offset, mask=valid, other=0).to(tl.float32)
    sublayer = tl.load(sublayer_ptr + sublayer_offset, mask=valid, other=0).to(
        tl.float32
    )
    gate = tl.load(gate_ptr + gate_offset, mask=valid, other=0).to(tl.float32)
    # Eager PyTorch materializes the product in BF16 before the BF16 add.
    injection = (sublayer * gate).to(tl.bfloat16).to(tl.float32)
    tl.store(output_ptr + offset, hyper + injection, mask=valid)


def is_supported(
    hyper_input: torch.Tensor,
    sublayer_out: torch.Tensor,
    inject_weights: torch.Tensor,
) -> bool:
    bf16_inputs = (
        hyper_input.dtype
        == sublayer_out.dtype
        == inject_weights.dtype
        == torch.bfloat16
    )
    same_prefix_shape = (
        hyper_input.shape[:-1]
        == sublayer_out.shape[:-1]
        == inject_weights.shape[:-1]
    )
    return (
        hyper_input.is_cuda
        and torch.version.hip is None
        and bf16_inputs
        and hyper_input.device == sublayer_out.device == inject_weights.device
        and hyper_input.is_contiguous()
        and sublayer_out.is_contiguous()
        and inject_weights.is_contiguous()
        and hyper_input.dim() >= 2
        and sublayer_out.dim() == inject_weights.dim() == hyper_input.dim()
        and same_prefix_shape
        and inject_weights.shape[-1] == 4
        and hyper_input.shape[-1] == 4 * sublayer_out.shape[-1]
    )


def fused_inject(
    hyper_input: torch.Tensor,
    sublayer_out: torch.Tensor,
    inject_weights: torch.Tensor,
) -> torch.Tensor:
    if not is_supported(hyper_input, sublayer_out, inject_weights):
        raise ValueError("unsupported Qwen4 gated residual injection layout")
    output = torch.empty_like(hyper_input)
    total = hyper_input.numel()
    if total:
        _inject_kernel[(triton.cdiv(total, 256),)](
            hyper_input,
            sublayer_out,
            inject_weights,
            output,
            TOTAL=total,
            HIDDEN=int(sublayer_out.shape[-1]),
            BRANCHES=4,
            BLOCK=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output
