"""Fused masked SiLU-mul and per-token INT8 quantization for PPU."""

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice


@triton.jit
def _silu_and_mul_masked_per_token_quant_int8_kernel(
    input_ptr,
    stride_input_0,
    stride_input_1,
    stride_input_2,
    output_ptr,
    stride_output_0,
    stride_output_1,
    stride_output_2,
    output_scale_ptr,
    stride_output_scale_0,
    stride_output_scale_1,
    stride_output_scale_2,
    masked_m_ptr,
    size_n,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    expert_id = tl.program_id(2)
    token_id = tl.program_id(1)
    block_num_per_expert = tl.num_programs(1)
    token_num_cur_expert = tl.load(masked_m_ptr + expert_id)

    stride_input_0 = tl.cast(stride_input_0, dtype=tl.int64)
    stride_output_0 = tl.cast(stride_output_0, dtype=tl.int64)
    stride_input_1 = tl.cast(stride_input_1, dtype=tl.int64)
    stride_output_1 = tl.cast(stride_output_1, dtype=tl.int64)

    offs_in_d = tl.arange(0, BLOCK_N)
    input_ptr_offs = input_ptr + expert_id * stride_input_0 + offs_in_d
    output_ptr_offs = output_ptr + expert_id * stride_output_0 + offs_in_d
    output_scale_offs = output_scale_ptr + expert_id * stride_output_scale_0

    for token_index in tl.range(
        token_id, token_num_cur_expert, block_num_per_expert, num_stages=NUM_STAGES
    ):
        # RTP-LLM's W1 output is laid out as up followed by gate.
        gate = tl.load(
            input_ptr_offs + token_index * stride_input_1 + size_n,
            mask=offs_in_d < size_n,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            input_ptr_offs + token_index * stride_input_1,
            mask=offs_in_d < size_n,
            other=0.0,
        )
        gate = gate / (1 + tl.exp(-gate))
        gate = gate.to(input_ptr.dtype.element_ty)
        gate_up = up * gate

        absmax = tl.maximum(tl.max(tl.abs(gate_up)), 1e-10)
        output_s = absmax / 127.0
        output_q = tl.clamp(gate_up / output_s, -128.0, 127.0)
        output_q = tldevice.round(output_q).to(tl.int8)

        tl.store(
            output_ptr_offs + token_index * stride_output_1,
            output_q,
            mask=offs_in_d < size_n,
        )
        tl.store(
            output_scale_offs + token_index * stride_output_scale_1,
            output_s,
        )


def silu_and_mul_masked_per_token_quant_int8_fwd(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    masked_m: torch.Tensor,
) -> None:
    """Fuse masked SiLU-mul with whole-row dynamic INT8 quantization.

    ``input`` is ``[experts, max_tokens, 2 * hidden]`` in ``up | gate``
    layout. Each valid expert-token row receives one FP32 scale.
    """

    assert input.is_contiguous()
    assert output.is_contiguous()
    assert output.dtype == torch.int8
    assert output_scale.dtype == torch.float32
    assert input.ndim == 3
    assert input.shape[-1] % 2 == 0
    assert masked_m.shape == (input.shape[0],)

    size_n = input.shape[-1] // 2
    assert output.shape == (*input.shape[:-1], size_n)
    assert output_scale.shape == (*output.shape[:-1], 1)

    expert_num = input.shape[0]
    block_num_per_expert = 64 if expert_num < 4 else 32
    block_n = triton.next_power_of_2(size_n)

    _silu_and_mul_masked_per_token_quant_int8_kernel[
        (1, block_num_per_expert, expert_num)
    ](
        input,
        *input.stride(),
        output,
        *output.stride(),
        output_scale,
        *output_scale.stride(),
        masked_m,
        size_n,
        BLOCK_N=block_n,
        NUM_STAGES=6,
        num_warps=1,
    )
