#pragma once

#if USING_CUDA
#include <cuda_runtime.h>
#elif USING_ROCM
#include <hip/hip_runtime.h>
#endif

namespace rtp_llm {

template<typename T>
void invoke_debug_kernel2(
    T* data, int start_row, int start_col, int m, int n, int row_len, int info_id, cudaStream_t stream);

}  // namespace rtp_llm
