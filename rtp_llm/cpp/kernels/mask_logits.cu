#include "rtp_llm/cpp/kernels/mask_logits.h"
#include "rtp_llm/cpp/kernels/util.h"

#if USING_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // For half
#include <cuda_bf16.h>  // For __nv_bfloat16
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
#endif

#if USING_ROCM
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#include "rtp_llm/cpp/rocm/hip_host_utils.h"
#endif

namespace rtp_llm {

// Batch version kernel for processing multiple beams
template<typename T>
__global__ void
mask_logits(const int batch_size, const int vocab_size, T* logits_batch, const uint8_t* __restrict__ mask_batch) {
    int batch_idx = blockIdx.y;
    int vocab_idx = threadIdx.x + (blockIdx.x * blockDim.x);

    if (batch_idx < batch_size && vocab_idx < vocab_size) {
        int global_idx = batch_idx * vocab_size + vocab_idx;
        if (mask_batch[global_idx]) {
            logits_batch[global_idx] = NegativeInfinity<T>();
        }
    }
}

template<typename T>
void invokeMaskLogits(T* logits_batch,
                      const uint8_t* __restrict__ mask_batch,
                      const int    batch_size,
                      const int    vocab_size,
                      cudaStream_t stream) {
    dim3 block, grid;

    block.x = 64;
    block.y = 1;
    block.z = 1;
    grid.x  = (vocab_size + block.x - 1) / block.x;
    grid.y  = batch_size;
    grid.z  = 1;

    mask_logits<<<grid, block, 0, stream>>>(batch_size, vocab_size, logits_batch, mask_batch);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
#endif
    check_cuda_error();
}

template void invokeMaskLogits<float>(float* logits_batch,
                                      const uint8_t* __restrict__ mask_batch,
                                      const int    batch_size,
                                      const int    vocab_size,
                                      cudaStream_t stream);
template void invokeMaskLogits<half>(half* logits_batch,
                                     const uint8_t* __restrict__ mask_batch,
                                     const int    batch_size,
                                     const int    vocab_size,
                                     cudaStream_t stream);
template void invokeMaskLogits<__nv_bfloat16>(__nv_bfloat16* logits_batch,
                                              const uint8_t* __restrict__ mask_batch,
                                              const int    batch_size,
                                              const int    vocab_size,
                                              cudaStream_t stream);

}  // namespace rtp_llm
