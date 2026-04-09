#include "rtp_llm/models_py/bindings/cuda/kernels/mla_kernels/mla_merge_transpose_kernel.h"

namespace rtp_llm {

#if USING_CUDA
// adapter from sglang/sgl-kernel/csrc/elementwise/concat_mla.cu
constexpr int NUM_LOCAL_HEADS  = 128;
constexpr int QK_NOPE_HEAD_DIM = 128;
constexpr int QK_ROPE_HEAD_DIM = 64;
constexpr int HEAD_CHUNK_SIZE  = 16;
constexpr int NUM_HEAD_CHUNKS  = NUM_LOCAL_HEADS / HEAD_CHUNK_SIZE;
// Fused kernel to concatenate k_nope and k_pe efficiently
template<typename T>
__global__ void concat_mla_k_kernel(T* __restrict__ k,
                                    const T* __restrict__ k_nope,
                                    const T* __restrict__ k_rope,
                                    const int     num_tokens,
                                    const int64_t k_stride_0,
                                    const int     k_stride_1,
                                    const int64_t k_nope_stride_0,
                                    const int     k_nope_stride_1,
                                    const int64_t k_rope_stride_0) {
    const int flat_warp_id  = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int token_id      = flat_warp_id / NUM_HEAD_CHUNKS;
    const int head_chunk_id = flat_warp_id % NUM_HEAD_CHUNKS;
    const int lane_id       = get_lane_id();
    if (token_id >= num_tokens)
        return;

    using NopeVec = int2;  // 8B/thread，32 thread = 256B/row
    using RopeVec = int;   // 4B/thread，32 thread = 128B/row
    static_assert(sizeof(NopeVec) * 32 == QK_NOPE_HEAD_DIM * sizeof(nv_bfloat16), "nope vec mismatch");
    static_assert(sizeof(RopeVec) * 32 == QK_ROPE_HEAD_DIM * sizeof(nv_bfloat16), "rope vec mismatch");

    const int head_row0 = head_chunk_id * HEAD_CHUNK_SIZE;

    const int2* __restrict__ nope_src =
        reinterpret_cast<const int2*>(k_nope + token_id * k_nope_stride_0 + head_row0 * k_nope_stride_1) + lane_id;

    int2* __restrict__ nope_dst = reinterpret_cast<int2*>(k + token_id * k_stride_0 + head_row0 * k_stride_1) + lane_id;

    int* __restrict__ rope_dst =
        reinterpret_cast<int*>(k + token_id * k_stride_0 + head_row0 * k_stride_1 + QK_NOPE_HEAD_DIM) + lane_id;

    const int nope_src_stride_v = (k_nope_stride_1 >> 2);  // int2 covers 4 bf16
    const int nope_dst_stride_v = (k_stride_1 >> 2);
    const int rope_dst_stride_v = (k_stride_1 >> 1);  // int covers 2 bf16

    const int*    rope_base = reinterpret_cast<const int*>(k_rope + token_id * k_rope_stride_0);
    const RopeVec rope_val  = ld_na_global_v1(rope_base + lane_id);

    prefetch_L2(nope_src);
    NopeVec cur = ld_na_global_v2(nope_src);

#pragma unroll
    for (int i = 0; i < HEAD_CHUNK_SIZE; ++i) {
        NopeVec next;
        if (i + 1 < HEAD_CHUNK_SIZE) {
            const int2* next_src = nope_src + nope_src_stride_v;
            prefetch_L2(next_src);
            next = ld_na_global_v2(next_src);
        }

        st_na_global_v2(nope_dst, cur);
        st_na_global_v1(rope_dst, rope_val);

        nope_src += nope_src_stride_v;
        nope_dst += nope_dst_stride_v;
        rope_dst += rope_dst_stride_v;

        cur = next;
    }
}

template<typename T>
void invokeMlaKMerge(T*            k,
                     T*            k_nope,
                     T*            k_rope,
                     const int     num_tokens,
                     const int64_t k_stride_0,
                     const int     k_stride_1,
                     const int64_t k_nope_stride_0,
                     const int     k_nope_stride_1,
                     const int64_t k_rope_stride_0,
                     cudaStream_t  stream) {
    constexpr int num_warps_per_block = 32;
    const int     grid_size           = (num_tokens * NUM_HEAD_CHUNKS + num_warps_per_block - 1) / num_warps_per_block;
    const int     block_size          = num_warps_per_block * 32;

    concat_mla_k_kernel<T><<<grid_size, block_size, 0, stream>>>(
        k, k_nope, k_rope, num_tokens, k_stride_0, k_stride_1, k_nope_stride_0, k_nope_stride_1, k_rope_stride_0);
}
#endif

#define INSTANTIATE_MLA_K_MERGE(T)                                                                                     \
    template void invokeMlaKMerge<T>(T * k_out,                                                                        \
                                     T * k_nope,                                                                       \
                                     T * k_pe,                                                                         \
                                     const int     num_tokens,                                                         \
                                     const int64_t k_stride_0,                                                         \
                                     const int     k_stride_1,                                                         \
                                     const int64_t k_nope_stride_0,                                                    \
                                     const int     k_nope_stride_1,                                                    \
                                     const int64_t k_rope_stride_0,                                                    \
                                     cudaStream_t  stream);

#if USING_CUDA && ENABLE_BF16
INSTANTIATE_MLA_K_MERGE(__nv_bfloat16);
#endif

}  // namespace rtp_llm
