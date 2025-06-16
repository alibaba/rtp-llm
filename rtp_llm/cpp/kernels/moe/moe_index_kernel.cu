#include "rtp_llm/cpp/kernels/moe/moe_index_kernel.h"

namespace rtp_llm {

__global__ void
genSourceRowKernelRevert(int64_t* expert_rows, int* expert_rows_dst, int token_num, int top_k, int start_expert) {
    int const idx       = blockIdx.x * blockDim.x + threadIdx.x;
    int const token_idx = idx / top_k;
    int const k_idx     = idx % top_k;
    if (idx < token_num * top_k) {
        if (expert_rows[idx] >= 0) {
            expert_rows_dst[idx] = expert_rows[idx] + start_expert;
        } else {
            expert_rows_dst[idx] = expert_rows[idx];
        }
    }
}

void genSourceRowRevert(
    int64_t* expert_rows, int* expert_rows_dst, int token_num, int top_k, int start_expert, cudaStream_t stream) {
    int const threads = 256;
    int const blocks  = token_num * top_k / 256 + 1;

    genSourceRowKernelRevert<<<blocks, threads, 0, stream>>>(
        expert_rows, expert_rows_dst, token_num, top_k, start_expert);
}
}  // namespace rtp_llm