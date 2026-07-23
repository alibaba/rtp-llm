import triton
import triton.language as tl


@triton.jit
def linear_offset_64(index, stride):
    """Compute a linear element offset without overflowing int32."""
    return index.to(tl.int64) * stride
