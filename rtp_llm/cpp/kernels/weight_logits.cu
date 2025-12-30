#include "rtp_llm/cpp/kernels/weight_logits.h"

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

template<typename T>
__device__ T addWeight(T logits, float weight) {
    return logits + weight;
}

template<>
__device__ __half addWeight<__half>(__half logits, float weight) {
    __half weight_half = __float2half(weight);
    return logits + weight_half;
}

template<>
__device__ __nv_bfloat16 addWeight<__nv_bfloat16>(__nv_bfloat16 logits, float weight) {
    __nv_bfloat16 weight_bf16 = __float2bfloat16(weight);
    return logits + weight_bf16;
}

// Batch version kernel for processing multiple beams
template<typename T>
__global__ void weight_logits(const int batch_size,
                              const int vocab_size,
                              const int weight_size,
                              T*        logits_batch,
                              const int* __restrict__ batch_idx,
                              const int* __restrict__ vocab_idx,
                              const float* __restrict__ weight_batch) {
    int weight_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (weight_idx < weight_size) {
        int b_idx = batch_idx[weight_idx];
        int v_idx = vocab_idx[weight_idx];
        if (b_idx < batch_size && v_idx < vocab_size) {
            int global_idx           = b_idx * vocab_size + v_idx;
            logits_batch[global_idx] = addWeight(logits_batch[global_idx], weight_batch[weight_idx]);
        }
    }
}

template<typename T>
void invokeWeightLogits(T* logits_batch,
                        const int* __restrict__ batch_idx,
                        const int* __restrict__ vocab_idx,
                        const float* __restrict__ weight_batch,
                        const int    batch_size,
                        const int    vocab_size,
                        const int    weight_size,
                        cudaStream_t stream) {
    dim3 block, grid;

    block.x = 32;
    block.y = 1;
    block.z = 1;
    grid.x  = (weight_size + block.x - 1) / block.x;
    grid.y  = 1;
    grid.z  = 1;

    weight_logits<<<grid, block, 0, stream>>>(
        batch_size, vocab_size, weight_size, logits_batch, batch_idx, vocab_idx, weight_batch);
#if USING_CUDA
    check_cuda_value(cudaPeekAtLastError());
#endif
    check_cuda_error();
}

template void invokeWeightLogits<float>(float* logits_batch,
                                        const int* __restrict__ batch_idx,
                                        const int* __restrict__ vocab_idx,
                                        const float* __restrict__ weight_batch,
                                        const int    batch_size,
                                        const int    vocab_size,
                                        const int    weight_size,
                                        cudaStream_t stream);
template void invokeWeightLogits<half>(half* logits_batch,
                                       const int* __restrict__ batch_idx,
                                       const int* __restrict__ vocab_idx,
                                       const float* __restrict__ weight_batch,
                                       const int    batch_size,
                                       const int    vocab_size,
                                       const int    weight_size,
                                       cudaStream_t stream);
template void invokeWeightLogits<__nv_bfloat16>(__nv_bfloat16* logits_batch,
                                                const int* __restrict__ batch_idx,
                                                const int* __restrict__ vocab_idx,
                                                const float* __restrict__ weight_batch,
                                                const int    batch_size,
                                                const int    vocab_size,
                                                const int    weight_size,
                                                cudaStream_t stream);

}  // namespace rtp_llm
