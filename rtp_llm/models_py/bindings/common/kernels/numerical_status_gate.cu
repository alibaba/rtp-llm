#include "rtp_llm/models_py/bindings/common/kernels/numerical_status_gate.h"
#include <torch/all.h>

#include <cmath>
#include <limits>

#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/launch_utils.h"
#elif USING_ROCM
#include "rtp_llm/models_py/bindings/rocm/kernels/rocm_utils.h"
#endif

namespace rtp_llm {

template<typename T>
__global__ void numericalStatusGateKernel(T*             logits,
                                          const int32_t* status,
                                          const int32_t* row_to_status,
                                          uint8_t*       failure_mask,
                                          int            rows,
                                          int            status_rows,
                                          int            vocab_size,
                                          int            status_scope) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const int status_row = status_scope == 1 ? 0 : row_to_status[row];
    const bool failed    = status_row < status_rows && status[status_row] != 0;
    if (threadIdx.x == 0) {
        failure_mask[row] = static_cast<uint8_t>(failed);
    }
    if (!failed) {
        return;
    }

    const int64_t row_offset = static_cast<int64_t>(row) * vocab_size;
    for (int col = threadIdx.x; col < vocab_size; col += blockDim.x) {
        logits[row_offset + col] = col == 0 ? static_cast<T>(0) : static_cast<T>(-INFINITY);
    }
}

template<typename T>
void invokeNumericalStatusGate(T*             logits,
                               const int32_t* status,
                               const int32_t* row_to_status,
                               uint8_t*       failure_mask,
                               int            rows,
                               int            status_rows,
                               int            vocab_size,
                               int            status_scope,
#if USING_CUDA
                               cudaStream_t stream) {
#elif USING_ROCM
                               hipStream_t stream) {
#endif
    numericalStatusGateKernel<T><<<rows, 256, 0, stream>>>(
        logits, status, row_to_status, failure_mask, rows, status_rows, vocab_size, status_scope);
}

template void invokeNumericalStatusGate<float>(float*, const int32_t*, const int32_t*, uint8_t*, int, int, int, int,
#if USING_CUDA
                                               cudaStream_t);
#else
                                               hipStream_t);
#endif
template void invokeNumericalStatusGate<at::Half>(
    at::Half*, const int32_t*, const int32_t*, uint8_t*, int, int, int, int,
#if USING_CUDA
                                                  cudaStream_t);
#else
                                                  hipStream_t);
#endif
template void invokeNumericalStatusGate<at::BFloat16>(
    at::BFloat16*, const int32_t*, const int32_t*, uint8_t*, int, int, int, int,
#if USING_CUDA
                                                      cudaStream_t);
#else
                                                      hipStream_t);
#endif

}  // namespace rtp_llm
