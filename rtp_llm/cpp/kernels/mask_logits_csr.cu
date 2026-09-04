#include "rtp_llm/cpp/kernels/mask_logits_csr.h"

#include "rtp_llm/cpp/kernels/mask_logits.h"

#if USING_CUDA
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#endif

#if USING_ROCM
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
#endif

namespace rtp_llm {
namespace {

// One block handles one beam. Each thread owns vocabulary positions and checks
// membership in the sorted CSR row. Unlike the earlier shared-memory prototype,
// this has no fan-out-dependent shared-memory limit and is valid for a wide root.
template<typename T>
__global__ void csrMaskLogitsKernel(
    T* logits, const int* states, const int* row_ptr, const int* col_idx, int vocab_size, int state_count) {
    const int beam  = blockIdx.x;
    const int state = states[beam];
    T*        row   = logits + static_cast<size_t>(beam) * vocab_size;
    if (state < 0 || state >= state_count) {
        for (int token = threadIdx.x; token < vocab_size; token += blockDim.x) {
            row[token] = static_cast<T>(-INFINITY);
        }
        return;
    }

    const int begin = row_ptr[state];
    const int end   = row_ptr[state + 1];
    for (int token = threadIdx.x; token < vocab_size; token += blockDim.x) {
        int low  = begin;
        int high = end;
        while (low < high) {
            const int middle = low + (high - low) / 2;
            if (col_idx[middle] < token) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        if (low == end || col_idx[low] != token) {
            row[token] = static_cast<T>(-INFINITY);
        }
    }
}

}  // namespace

template<typename T>
void invokeCSRMaskLogits(T*         logits,
                         const int* states,
                         const int* row_ptr,
                         const int* col_idx,
                         int        batch_size,
                         int        vocab_size,
                         int        state_count,
#if USING_CUDA
                         cudaStream_t stream)
#elif USING_ROCM
                         hipStream_t stream)
#endif
{
    if (batch_size <= 0 || vocab_size <= 0) {
        return;
    }
    csrMaskLogitsKernel<T><<<batch_size, 256, 0, stream>>>(logits, states, row_ptr, col_idx, vocab_size, state_count);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
#endif
    check_cuda_error();
}

template void invokeCSRMaskLogits<float>(float*,
                                         const int*,
                                         const int*,
                                         const int*,
                                         int,
                                         int,
                                         int,
#if USING_CUDA
                                         cudaStream_t);
#elif USING_ROCM
                                         hipStream_t);
#endif

template void invokeCSRMaskLogits<half>(half*,
                                        const int*,
                                        const int*,
                                        const int*,
                                        int,
                                        int,
                                        int,
#if USING_CUDA
                                        cudaStream_t);
#elif USING_ROCM
                                        hipStream_t);
#endif

template void invokeCSRMaskLogits<__nv_bfloat16>(__nv_bfloat16*,
                                                 const int*,
                                                 const int*,
                                                 const int*,
                                                 int,
                                                 int,
                                                 int,
#if USING_CUDA
                                                 cudaStream_t);
#elif USING_ROCM
                                                 hipStream_t);
#endif

}  // namespace rtp_llm
