#include "rtp_llm/models_py/bindings/common/kernels/input_embedding_overlay_kernel.h"

#if USING_CUDA
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif
#if USING_ROCM
#include <hip/hip_runtime.h>
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
#endif

namespace rtp_llm {
namespace {

constexpr int kThreadsPerBlock = 256;
#if USING_ROCM
constexpr int kWarpSize = 64;
#else
constexpr int kWarpSize = 32;
#endif
constexpr int kRowsPerBlock = kThreadsPerBlock / kWarpSize;

template<typename T>
__global__ void inputEmbeddingOverlayKernel(T* __restrict__ inputs_embeds,
                                            const T* __restrict__ overrides,
                                            int32_t* __restrict__ metadata,
                                            int token_count,
                                            int hidden_size) {
    const int warp_in_block = threadIdx.x / kWarpSize;
    const int lane          = threadIdx.x % kWarpSize;
    const int row           = blockIdx.x * kRowsPerBlock + warp_in_block;
    if (row >= token_count || metadata[row] == 0) {
        return;
    }

    const int64_t row_offset = static_cast<int64_t>(row) * hidden_size;
    for (int col = lane; col < hidden_size; col += kWarpSize) {
        inputs_embeds[row_offset + col] = overrides[row_offset + col];
    }

    // Every lane must finish reading the active flag and copying the row
    // before lane zero makes this metadata slot reusable by the next replay.
    __syncwarp();
    if (lane == 0) {
        metadata[row] = 0;
    }
}

}  // namespace

template<typename T>
void invokeInputEmbeddingOverlay(
    T* inputs_embeds, const T* overrides, int32_t* metadata, int token_count, int hidden_size, cudaStream_t stream) {
    if (inputs_embeds == nullptr || overrides == nullptr || metadata == nullptr || token_count <= 0
        || hidden_size <= 0) {
        return;
    }
    const int blocks = (token_count + kRowsPerBlock - 1) / kRowsPerBlock;
    inputEmbeddingOverlayKernel<T>
        <<<blocks, kThreadsPerBlock, 0, stream>>>(inputs_embeds, overrides, metadata, token_count, hidden_size);
}

template void invokeInputEmbeddingOverlay<float>(float*, const float*, int32_t*, int, int, cudaStream_t);
template void invokeInputEmbeddingOverlay<half>(half*, const half*, int32_t*, int, int, cudaStream_t);
#ifdef ENABLE_BF16
#if USING_ROCM
template void
invokeInputEmbeddingOverlay<amd_bfloat16>(amd_bfloat16*, const amd_bfloat16*, int32_t*, int, int, cudaStream_t);
#else
template void
invokeInputEmbeddingOverlay<__nv_bfloat16>(__nv_bfloat16*, const __nv_bfloat16*, int32_t*, int, int, cudaStream_t);
#endif
#endif

}  // namespace rtp_llm
