#include <cuda_runtime.h>
#include <cuda_fp16.h> // For half
#include <cuda_bf16.h> // For __nv_bfloat16
#include "maga_transformer/cpp/kernels/mask_logits.h"
#if USING_CUDA
#include "maga_transformer/cpp/cuda/cuda_utils.h"
#endif
#if USING_ROCM
#include "maga_transformer/cpp/rocm/hip_utils.h"
#endif

namespace rtp_llm {

#ifndef CUDART_INF_FP16
#define CUDART_INF_FP16 __ushort_as_half((unsigned short)0x7C00U)
#endif

#ifndef CUDART_INF_BF16
#define CUDART_INF_BF16 __ushort_as_bfloat16((unsigned short)0x7F80U)
#endif

template <typename T>
__device__ T NegativeInfinity() {
    return -INFINITY;
}

template <>
__device__ __half NegativeInfinity<__half>() {
    return -CUDART_INF_FP16;
}

template <>
__device__ __nv_bfloat16 NegativeInfinity<__nv_bfloat16>() {
    return -CUDART_INF_BF16;
}

template<typename T>
__global__ void mask_logits(const int N, T* logits, const uint8_t* __restrict__ mask) {
    int i = threadIdx.x + (blockIdx.x * blockDim.x);

    if (i < N && mask[i]) {
        logits[i] = NegativeInfinity<T>();
    }
}

template<typename T>
void invokeMaskLogits(T* logits, const uint8_t* __restrict__ mask, const int vector_len, cudaStream_t stream) {
    dim3 block, grid;

    block.x = 64;
    block.y = 1;
    block.z = 1;
    grid.x  = (vector_len + block.x - 1) / block.x;
    grid.y  = 1;
    grid.z  = 1;

    mask_logits<<<grid, block, 0, stream>>>(vector_len, logits, mask);

    sync_check_cuda_error();
}

template void invokeMaskLogits<float>(float* logits, const uint8_t* __restrict__ mask, const int vector_len, cudaStream_t stream);
template void invokeMaskLogits<half>(half* logits, const uint8_t* __restrict__ mask, const int vector_len, cudaStream_t stream);
template void invokeMaskLogits<__nv_bfloat16>(__nv_bfloat16* logits, const uint8_t* __restrict__ mask, const int vector_len, cudaStream_t stream);

}  // namespace rtp_llm
