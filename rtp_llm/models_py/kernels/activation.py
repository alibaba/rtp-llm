import torch
import triton
import triton.language as tl


@triton.jit
def _silu_and_mul_kernel(
    output_ptr,
    input_ptr,
    # Tensor dimensions
    N: tl.int32,
    # Row strides for jumping between batches
    input_row_stride: tl.int32,
    output_row_stride: tl.int32,
    # Meta-parameter for tuning
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)  # Batch dimension
    pid_n_block = tl.program_id(axis=1)  # N-dimension block

    input_row_start_ptr = input_ptr + pid_b * input_row_stride
    output_row_start_ptr = output_ptr + pid_b * output_row_stride

    n_offsets = pid_n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    value_ptrs = input_row_start_ptr + n_offsets
    gate_ptrs = input_row_start_ptr + N + n_offsets
    output_ptrs = output_row_start_ptr + n_offsets

    mask = n_offsets < N
    gate = tl.load(gate_ptrs, mask=mask)
    value = tl.load(value_ptrs, mask=mask)

    silu_gate = gate * tl.sigmoid(gate.to(tl.float32))
    output = silu_gate * value

    tl.store(output_ptrs, output, mask=mask)


def silu_and_mul(
    output_tensor: torch.Tensor, input_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Computes SiLU(gate) * value in a fused Triton kernel.
    Assumes input_tensor has shape [B, 2*N] and is contiguous.
    """
    B, D = input_tensor.shape
    assert D % 2 == 0, "Last dimension must be even (2*N)"
    N = D // 2

    # Kernel launch grid
    grid = lambda meta: (B, triton.cdiv(N, meta["BLOCK_SIZE_N"]))

    # Heuristic for block size
    BLOCK_SIZE_N = 1024 if N > 1024 else triton.next_power_of_2(N)

    _silu_and_mul_kernel[grid](
        output_tensor,
        input_tensor,
        N,
        input_tensor.stride(0),
        output_tensor.stride(0),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return output_tensor
