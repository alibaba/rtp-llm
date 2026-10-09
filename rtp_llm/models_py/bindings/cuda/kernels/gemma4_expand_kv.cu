#include "rtp_llm/models_py/bindings/cuda/kernels/gemma4_expand_kv.h"

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace rtp_llm {
namespace {

constexpr int kKvHeads        = 2;
constexpr int kQueryHeads     = 16;
constexpr int kHeadDim        = 512;
constexpr int kGroupSize      = kQueryHeads / kKvHeads;
constexpr int kVectorBytes    = sizeof(uint4);
constexpr int kBf16PerVector  = kVectorBytes / sizeof(__nv_bfloat16);
constexpr int kVectorsPerHead = kHeadDim / kBf16PerVector;

__global__ void gemma4ExpandKvHeads8Bf16Kernel(
    const uint4* k, const uint4* v, uint4* expanded_k, uint4* expanded_v, int64_t vector_count) {
    const int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_index >= vector_count) {
        return;
    }
    const int64_t vector_in_head = output_index % kVectorsPerHead;
    const int64_t query_head     = (output_index / kVectorsPerHead) % kQueryHeads;
    const int64_t token          = output_index / (kVectorsPerHead * kQueryHeads);
    const int64_t kv_head        = query_head / kGroupSize;
    const int64_t input_index    = (token * kKvHeads + kv_head) * kVectorsPerHead + vector_in_head;
    expanded_k[output_index]     = k[input_index];
    expanded_v[output_index]     = v[input_index];
}

constexpr int kSwaKvHeads        = 8;
constexpr int kSwaQueryHeads     = 16;
constexpr int kSwaHeadDim        = 256;
constexpr int kSwaGroupSize      = kSwaQueryHeads / kSwaKvHeads;
constexpr int kSwaVectorsPerHead = kSwaHeadDim / kBf16PerVector;

__global__ void gemma4ExpandKvHeads2Bf16Kernel(
    const uint4* k, const uint4* v, uint4* expanded_k, uint4* expanded_v, int64_t vector_count) {
    const int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_index >= vector_count) {
        return;
    }
    const int64_t vector_in_head = output_index % kSwaVectorsPerHead;
    const int64_t query_head     = (output_index / kSwaVectorsPerHead) % kSwaQueryHeads;
    const int64_t token          = output_index / (kSwaVectorsPerHead * kSwaQueryHeads);
    const int64_t kv_head        = query_head / kSwaGroupSize;
    const int64_t input_index    = (token * kSwaKvHeads + kv_head) * kSwaVectorsPerHead + vector_in_head;
    expanded_k[output_index]     = k[input_index];
    expanded_v[output_index]     = v[input_index];
}

}  // namespace

void invokeGemma4ExpandKvHeads8Bf16(const __nv_bfloat16* k,
                                    const __nv_bfloat16* v,
                                    __nv_bfloat16*       expanded_k,
                                    __nv_bfloat16*       expanded_v,
                                    int64_t              tokens,
                                    cudaStream_t         stream) {
    const int64_t vector_count = tokens * kQueryHeads * kVectorsPerHead;
    constexpr int threads      = 256;
    const int     blocks       = static_cast<int>((vector_count + threads - 1) / threads);
    gemma4ExpandKvHeads8Bf16Kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const uint4*>(k),
                                                                   reinterpret_cast<const uint4*>(v),
                                                                   reinterpret_cast<uint4*>(expanded_k),
                                                                   reinterpret_cast<uint4*>(expanded_v),
                                                                   vector_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void invokeGemma4ExpandKvHeads2Bf16(const __nv_bfloat16* k,
                                    const __nv_bfloat16* v,
                                    __nv_bfloat16*       expanded_k,
                                    __nv_bfloat16*       expanded_v,
                                    int64_t              tokens,
                                    cudaStream_t         stream) {
    const int64_t vector_count = tokens * kSwaQueryHeads * kSwaVectorsPerHead;
    constexpr int threads      = 256;
    const int     blocks       = static_cast<int>((vector_count + threads - 1) / threads);
    gemma4ExpandKvHeads2Bf16Kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const uint4*>(k),
                                                                   reinterpret_cast<const uint4*>(v),
                                                                   reinterpret_cast<uint4*>(expanded_k),
                                                                   reinterpret_cast<uint4*>(expanded_v),
                                                                   vector_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace rtp_llm
