#pragma once

#ifdef USING_ROCM
#include "rtp_llm/cpp/rocm/hip_utils.h"
#else
#include "rtp_llm/cpp/cuda/cuda_utils.h"
#endif

namespace rtp_llm {
void genSourceRowRevert(
    int64_t* expert_rows, int* expert_rows_dst, int token_num, int top_k, int start_expert, cudaStream_t stream);
}  // namespace rtp_llm