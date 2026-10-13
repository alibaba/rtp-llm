import triton
import triton.language as tl


@triton.jit
def linear_offset_64(index, stride):
    """Compute a linear element offset without overflowing int32."""
    # tl.cast handles both runtime tensors and compile-time scalar values. The
    # latter occur for static loop induction variables in chunk kernels and do
    # not consistently provide Tensor.to() across Triton versions.
    return tl.cast(index, tl.int64) * stride
