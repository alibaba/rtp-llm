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
}  // namespace rtp_llm