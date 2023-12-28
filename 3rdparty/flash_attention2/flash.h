/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cuda.h>
#include <vector>

constexpr int TOTAL_DIM = 0;
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
    using index_t = uint32_t;
    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    // The number of heads.
    int h, h_k;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_h_k_ratio; // precompute h / h_k,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params {

    // The O matrix (output).
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;

    // The stride between rows of O.
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // The pointer to the P matrix.
    void * __restrict__ p_ptr;

    // The pointer to the softmax sum.
    void * __restrict__ softmax_lse_ptr;
    void * __restrict__ softmax_lseaccum_ptr;

    // The dimensions.
    int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded;

    // The scaling factors for the kernel.
    float scale_softmax;
    float scale_softmax_log2;

    // array of length b+1 holding starting offset of each sequence.
    int * __restrict__ cu_seqlens_q;
    int * __restrict__ cu_seqlens_k;

    int *__restrict__ blockmask;

    // The K_new and V_new matrices.
    void * __restrict__ knew_ptr;
    void * __restrict__ vnew_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t knew_batch_stride;
    index_t vnew_batch_stride;
    index_t knew_row_stride;
    index_t vnew_row_stride;
    index_t knew_head_stride;
    index_t vnew_head_stride;

    // The dropout probability (probability of keeping an activation).
    float p_dropout;
    // uint32_t p_dropout_in_uint;
    // uint16_t p_dropout_in_uint16_t;
    uint8_t p_dropout_in_uint8_t;

    // Scale factor of 1 / (1 - p_dropout).
    float rp_dropout;
    float scale_softmax_rp_dropout;

    // Random state.
    // at::PhiloxCudaState philox_args;

    // Pointer to the RNG seed (idx 0) and offset (idx 1).
    // uint64_t * rng_state;

    bool is_bf16;
    bool is_causal;

    // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
    // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
    bool is_seqlens_k_cumulative;

    int num_splits;  // For split-KV version
    bool is_alibi;
    void * __restrict__ linear_bias_slopes;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Headdim> void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);
template<typename T, int Headdim>
void run_mha_fwd_splitkv_dispatch(Flash_fwd_params& params, cudaStream_t stream);

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false);

#define FA_CUDA_CHECK(EXPR)                                             \
    do {                                                                \
        const cudaError_t __err = EXPR;                                 \
        if (__err) {                                                    \
            throw std::runtime_error(std::string("FA CUDA runtime error: ") + (cudaGetErrorString(__err)) + " " \
                                     + __FILE__ + ":" + std::to_string(__LINE__) + " \n"); \
        }                                                               \
    } while (0)

#define FA_CUDA_KERNEL_LAUNCH_CHECK()  FA_CUDA_CHECK(cudaGetLastError())

bool is_sm8x();

bool flash_attention_enabled();
