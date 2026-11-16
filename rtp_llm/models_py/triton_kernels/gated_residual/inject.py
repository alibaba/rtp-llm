"""Graph-safe Qwen4 gated residual writeback."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["TOTAL"])
def _inject_kernel(
    hyper_ptr,
    sublayer_ptr,
    gate_ptr,
    output_ptr,
    TOTAL,
    HIDDEN: tl.constexpr,
    BRANCHES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
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


@triton.jit
def _inject_group_tile(
    hyper_ptr, sublayer_ptr, gate_ptr, output_ptr, HIDDEN: tl.constexpr, BLOCK: tl.constexpr
):
    tile = tl.program_id(0)
    tiles_per_branch: tl.constexpr = HIDDEN // BLOCK
    row = (tile // (4 * tiles_per_branch)).to(tl.int64)
    group = (tile // tiles_per_branch) % 4
    column = tile % tiles_per_branch * BLOCK + tl.arange(0, BLOCK)
    offset = row * (4 * HIDDEN) + group * HIDDEN + column
    hyper = tl.load(hyper_ptr + offset).to(tl.float32)
    sublayer = tl.load(sublayer_ptr + row * HIDDEN + column).to(tl.float32)
    gate = tl.load(gate_ptr + row * 4 + group).to(tl.float32)
    injection = (sublayer * gate).to(tl.bfloat16).to(tl.float32)
    tl.store(output_ptr + offset, hyper + injection)


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
        hyper_input.shape[:-1] == sublayer_out.shape[:-1] == inject_weights.shape[:-1]
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
        hidden = sublayer_out.shape[-1]
        if (
            hidden == 8192
            and torch.cuda.get_device_capability(hyper_input.device) in ((10, 0), (10, 3))
        ):
            # One CTA reads a scalar gate and contiguous columns from its branch.
            block = 2048 if total >= 64 * 4 * hidden else 1024
            _inject_group_tile[(total // block,)](
                hyper_input,
                sublayer_out,
                inject_weights,
                output,
                HIDDEN=hidden,
                BLOCK=block,
                num_warps=4,
                enable_fp_fusion=False,
            )
            return output
        _inject_kernel[(triton.cdiv(total, 256),)](
            hyper_input,
            sublayer_out,
            inject_weights,
            output,
            TOTAL=total,
            HIDDEN=hidden,
            BRANCHES=4,
            BLOCK=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output


def try_fused_inject(hyper_input, sublayer_out, inject_weights):
    if not is_supported(hyper_input, sublayer_out, inject_weights):
        return None
    return fused_inject(hyper_input, sublayer_out, inject_weights)
