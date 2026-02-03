#include "rtp_llm/cpp/kernels/mla_kernels/mla_merge_transpose_kernel.h"

namespace rtp_llm {

template<typename T>
__global__ void mla_merge_transpose_kernel(T*  q,
                                           T*  k_nope,
                                           T*  k_rope,
                                           T*  v,
                                           T*  qkv,
                                           int token_num,
                                           int head_num,
                                           int nope_head_dim,
                                           int rope_head_dim,
                                           int v_head_dim) {
    // q = [q_nope, q_rope]: [token_num, head_num, nope_dim + rope_dim]
    // k_nope: [token_num, head_num, nope_dim]
    // k_rope: [token_num, 1, rope_dim]
    // v: [bs, head_num, vhead_dim]
    // qkv: [token_num, 3, head_num, (nope_dim + rope_dim)]
    // grid: (nope_dim + rope_dim, 1, 1)
    // block: (token_num, head_num, 1)

    // rope transpose for q_rope and k_rope: [..., rope_dim / 2, 2] => [..., 2, rope_dim / 2]

    int nope_rope_dim = nope_head_dim + rope_head_dim;
    int tidx          = threadIdx.x;

    if (tidx >= nope_rope_dim) {
        return;
    }

    int bs_idx      = blockIdx.x;
    int head_idx    = blockIdx.y;
    int rope_idx    = tidx - nope_head_dim;
    int hidden_size = head_num * nope_rope_dim;

    int q_offset      = bs_idx * head_num * nope_rope_dim + head_idx * nope_rope_dim + tidx;
    int k_nope_offset = bs_idx * head_num * nope_head_dim + head_idx * nope_head_dim + tidx;
    int k_rope_offset = bs_idx * rope_head_dim + rope_idx;  // broadcast to head_num
    int v_offset      = bs_idx * head_num * v_head_dim + head_idx * v_head_dim + tidx;

    int dst_base_offset = bs_idx * 3 * hidden_size + head_idx * nope_rope_dim + tidx;

    if (tidx < nope_head_dim) {
        qkv[dst_base_offset]               = q[q_offset];
        qkv[dst_base_offset + hidden_size] = k_nope[k_nope_offset];
    } else {
        int trans_idx    = rope_idx / 2;
        int trans_offset = trans_idx + (rope_idx % 2 ? 1 : 0) * rope_head_dim / 2 - tidx + nope_head_dim;
        int q_dst        = dst_base_offset + trans_offset;
        int k_dst        = q_dst + hidden_size;
        qkv[q_dst]       = q[q_offset];
        qkv[k_dst]       = k_rope[k_rope_offset];
    }

    // padding 0 for v
    if (tidx < v_head_dim) {
        qkv[dst_base_offset + 2 * hidden_size] = v[v_offset];
    } else {
        qkv[dst_base_offset + 2 * hidden_size] = 0;
    }
}

template<typename T>
void invokeMlaMergeTranspose(T*           q,
                             T*           k_nope,
                             T*           k_rope,
                             T*           v,
                             T*           qkv,
                             int          token_num,
                             int          head_num,
                             int          nope_head_dim,
                             int          rope_head_dim,
                             int          v_head_dim,
                             cudaStream_t stream) {
    dim3 grid(token_num, head_num);
    dim3 block(nope_head_dim + rope_head_dim);

    mla_merge_transpose_kernel<T><<<grid, block, 0, stream>>>(
        q, k_nope, k_rope, v, qkv, token_num, head_num, nope_head_dim, rope_head_dim, v_head_dim);
}

#define INSTANTIATE_MLA_MERGE_TRANSPOSE(T)                                                                             \
    template void invokeMlaMergeTranspose<T>(T * q,                                                                    \
                                             T * k_nope,                                                               \
                                             T * k_rope,                                                               \
                                             T * v,                                                                    \
                                             T * qkv,                                                                  \
                                             int          token_num,                                                   \
                                             int          head_num,                                                    \
                                             int          nope_head_dim,                                               \
                                             int          rope_head_dim,                                               \
                                             int          v_head_dim,                                                  \
                                             cudaStream_t stream);

INSTANTIATE_MLA_MERGE_TRANSPOSE(float);
INSTANTIATE_MLA_MERGE_TRANSPOSE(__half);

#ifdef ENABLE_BF16
INSTANTIATE_MLA_MERGE_TRANSPOSE(__nv_bfloat16);
#endif

template<typename T>
__global__ void mla_qkv_kernel(T*  q,
                               T*  k_nope,
                               T*  k_rope,
                               T*  v,
                               T*  qkv,
                               int token_num,
                               int head_num,
                               int nope_head_dim,
                               int rope_head_dim,
                               int v_head_dim) {
    // q: [token_num, head_num, nope_dim + rope_dim]
    // k_nope: [token_num, head_num, nope_dim]
    // k_rope: [token_num, 1, rope_dim]
    // v: [bs, head_num, vhead_dim]
    // qkv: [token_num, 3, head_num, (nope_dim + rope_dim)]
    // grid: (nope_dim + rope_dim, 1, 1)
    // block: (token_num, head_num, 1)

    // rope transpose for q_rope and k_rope: [..., rope_dim / 2, 2] => [..., 2, rope_dim / 2]

    int nope_rope_dim = nope_head_dim + rope_head_dim;
    int tidx          = threadIdx.x;

    if (tidx >= nope_rope_dim) {
        return;
    }

    int bs_idx      = blockIdx.x;
    int head_idx    = blockIdx.y;
    int rope_idx    = tidx - nope_head_dim;
    int hidden_size = head_num * nope_rope_dim;

    int q_offset      = bs_idx * head_num * nope_rope_dim + head_idx * nope_rope_dim + tidx;
    int k_nope_offset = bs_idx * head_num * nope_head_dim + head_idx * nope_head_dim + tidx;
    int k_rope_offset = bs_idx * rope_head_dim + rope_idx;  // broadcast to head_num
    int v_offset      = bs_idx * head_num * v_head_dim + head_idx * v_head_dim + tidx;

    int dst_base_offset = bs_idx * 3 * hidden_size + head_idx * nope_rope_dim + tidx;

    if (tidx < nope_head_dim) {
        qkv[dst_base_offset]               = q[q_offset];
        qkv[dst_base_offset + hidden_size] = k_nope[k_nope_offset];
    } else {
        qkv[dst_base_offset]               = q[q_offset];
        qkv[dst_base_offset + hidden_size] = k_rope[k_rope_offset];
    }

    // padding 0 for v
    if (tidx < v_head_dim) {
        qkv[dst_base_offset + 2 * hidden_size] = v[v_offset];
    } else {
        qkv[dst_base_offset + 2 * hidden_size] = 0;
    }
}

template<typename T>
void invokeMlaQKVMerge(T*           q,
                       T*           k_nope,
                       T*           k_rope,
                       T*           v,
                       T*           qkv,
                       int          token_num,
                       int          head_num,
                       int          nope_head_dim,
                       int          rope_head_dim,
                       int          v_head_dim,
                       cudaStream_t stream) {
    dim3 grid(token_num, head_num);
    dim3 block(nope_head_dim + rope_head_dim);

    mla_qkv_kernel<T><<<grid, block, 0, stream>>>(
        q, k_nope, k_rope, v, qkv, token_num, head_num, nope_head_dim, rope_head_dim, v_head_dim);
}

#define INSTANTIATE_MLA_QKV_MERGE(T)                                                                                   \
    template void invokeMlaQKVMerge<T>(T * q,                                                                          \
                                       T * k_nope,                                                                     \
                                       T * k_rope,                                                                     \
                                       T * v,                                                                          \
                                       T * qkv,                                                                        \
                                       int          token_num,                                                         \
                                       int          head_num,                                                          \
                                       int          nope_head_dim,                                                     \
                                       int          rope_head_dim,                                                     \
                                       int          v_head_dim,                                                        \
                                       cudaStream_t stream);

INSTANTIATE_MLA_QKV_MERGE(float);
INSTANTIATE_MLA_QKV_MERGE(__half);

#ifdef ENABLE_BF16
INSTANTIATE_MLA_QKV_MERGE(__nv_bfloat16);
#endif

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

#if USING_CUDA
// concat_mla_absorb_q: fuse concat q_nope (512) and q_rope (64) on last dim -> out (576)
constexpr int MLA_Q_A_LAST_DIM = 512;
constexpr int MLA_Q_B_LAST_DIM = 64;

template<typename T>
__global__ void concat_mla_q_merge_kernel(T* __restrict__ a,
                                          T* __restrict__ b,
                                          T* __restrict__ out,
                                          const int     num_items,
                                          const int     dim_1,
                                          const int64_t a_stride_0,
                                          const int     a_stride_1,
                                          const int64_t b_stride_0,
                                          const int     b_stride_1,
                                          const int64_t out_stride_0,
                                          const int     out_stride_1) {
    const int flat_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane_id      = get_lane_id();

    const int idx_0 = flat_warp_id / dim_1;
    const int idx_1 = flat_warp_id % dim_1;

    if (flat_warp_id >= num_items) {
        return;
    }

    using ABufType             = int4;
    constexpr int A_NUM_UNROLL = 2;
    static_assert(sizeof(ABufType) * A_NUM_UNROLL == MLA_Q_A_LAST_DIM * sizeof(T) / 32);
    ABufType a_buf[A_NUM_UNROLL];

    using BBufType             = int;
    constexpr int B_NUM_UNROLL = 1;
    static_assert(sizeof(BBufType) * B_NUM_UNROLL == MLA_Q_B_LAST_DIM * sizeof(T) / 32);
    BBufType b_buf;

    {
        const BBufType* base_addr = reinterpret_cast<const BBufType*>(b + idx_0 * b_stride_0 + idx_1 * b_stride_1);
        b_buf                     = *(base_addr + lane_id);
    }

#pragma unroll
    for (int i = 0; i < A_NUM_UNROLL; ++i) {
        const ABufType* base_addr = reinterpret_cast<const ABufType*>(a + idx_0 * a_stride_0 + idx_1 * a_stride_1);
        a_buf[i]                  = *(base_addr + i * 32 + lane_id);
    }

    {
        BBufType* base_addr =
            reinterpret_cast<BBufType*>(out + idx_0 * out_stride_0 + idx_1 * out_stride_1 + MLA_Q_A_LAST_DIM);
        *(base_addr + lane_id) = b_buf;
    }

#pragma unroll
    for (int i = 0; i < A_NUM_UNROLL; ++i) {
        ABufType* base_addr = reinterpret_cast<ABufType*>(out + idx_0 * out_stride_0 + idx_1 * out_stride_1);
        *(base_addr + i * 32 + lane_id) = a_buf[i];
    }
}

template<typename T>
void invokeMlaQMerge(T*            a,
                     T*            b,
                     T*            out,
                     const int     num_items,
                     const int     dim_1,
                     const int64_t a_stride_0,
                     const int     a_stride_1,
                     const int64_t b_stride_0,
                     const int     b_stride_1,
                     const int64_t out_stride_0,
                     const int     out_stride_1,
                     cudaStream_t  stream) {
    constexpr int num_warps_per_block = 32;
    const int     grid_size           = (num_items + num_warps_per_block - 1) / num_warps_per_block;
    const int     block_size          = num_warps_per_block * 32;

    concat_mla_q_merge_kernel<T><<<grid_size, block_size, 0, stream>>>(
        a, b, out, num_items, dim_1, a_stride_0, a_stride_1, b_stride_0, b_stride_1, out_stride_0, out_stride_1);
}

#define INSTANTIATE_MLA_Q_MERGE(T)                                                                                     \
    template void invokeMlaQMerge<T>(T * a,                                                                            \
                                     T * b,                                                                            \
                                     T * out,                                                                          \
                                     const int     num_items,                                                          \
                                     const int     dim_1,                                                              \
                                     const int64_t a_stride_0,                                                         \
                                     const int     a_stride_1,                                                         \
                                     const int64_t b_stride_0,                                                         \
                                     const int     b_stride_1,                                                         \
                                     const int64_t out_stride_0,                                                       \
                                     const int     out_stride_1,                                                       \
                                     cudaStream_t  stream);

#if ENABLE_BF16
INSTANTIATE_MLA_Q_MERGE(__nv_bfloat16);
#endif
#endif
}  // namespace rtp_llm