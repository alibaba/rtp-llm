#include "maga_transformer/cpp/kernels/hello_world.h"
#if USING_CUDA
#include "maga_transformer/cpp/cuda/cuda_utils.h"
#endif
#if USING_ROCM
#include "maga_transformer/cpp/rocm/hip_utils.h"
#endif

namespace rtp_llm {

template<typename T>
__global__ void vector_add(const int N, const T* a, const T* b, T* c) {
    int i = threadIdx.x + (blockIdx.x * blockDim.x);

    if (i < N) {
        c[i] = b[i] + a[i];
    }
}

template<typename T>
void invokeHelloWorld(const T* a, const T* b, T* c, const int vector_len, cudaStream_t stream) {
    dim3 block, grid;

    block.x = 64;
    block.y = 1;
    block.z = 1;
    grid.x  = (vector_len + block.x - 1) / block.x;
    grid.y  = 1;
    grid.z  = 1;

    vector_add<<<grid, block, 0, stream>>>(vector_len, a, b, c);

    sync_check_cuda_error();
}

template void invokeHelloWorld(const float* a, const float* b, float* c, const int vector_len, cudaStream_t stream);

}  // namespace rtp_llm
