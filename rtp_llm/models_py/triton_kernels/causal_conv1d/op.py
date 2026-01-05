import triton
import triton.language as tl


# assume x always greater than 1
@triton.jit
def cal_block_idx(x, seq_size_per_block):
    return (x - 1) // seq_size_per_block
