/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdexcept>
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"
#include "src/fastertransformer/kernels/sampling_topk_kernels.h"
#include "src/fastertransformer/cuda/cuda_utils.h"

namespace fastertransformer {

__global__ void curandInitialize(curandState_t* state, const int size, const unsigned long long random_seed)
{
    if (threadIdx.x + blockIdx.x * blockDim.x < size) {
        curand_init(random_seed, 0, 0, &state[blockIdx.x * blockDim.x + threadIdx.x]);
    }
}

void invokeCurandInitialize(curandState_t*           state,
                            const size_t             batch_size,
                            const unsigned long long random_seed,
                            cudaStream_t             stream)
{
    dim3 block(256);
    dim3 grid((int)(ceil(batch_size * 1.0 / 256)));
    curandInitialize<<<grid, block, 0, stream>>>(state, batch_size, random_seed);
}

__global__ void curandBatchInitialize(curandState_t* states, const int size, const unsigned long long* random_seeds)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        curand_init(random_seeds[idx], 0, 0, &states[idx]);
    }
}

void invokeCurandBatchInitialize(curandState_t*            states,
                                 const size_t              batch_size,
                                 const unsigned long long* random_seeds,
                                 cudaStream_t              stream)
{
    dim3 block(256);
    dim3 grid((int)(ceil(batch_size * 1.0 / 256)));
    curandBatchInitialize<<<grid, block, 0, stream>>>(states, batch_size, random_seeds);
}

template<typename T>
__global__ void addBiasEndMask(T*          logits,
                               const T*    bias,
                               const int*  end_ids,
                               const bool* finished,
                               const int   vocab_size,
                               const int   vocab_size_padded)
{
    int  bid    = blockIdx.x;
    bool finish = finished != nullptr ? finished[bid] : false;
    int  offset = bid * vocab_size_padded;

    const bool IS_FP16   = std::is_same<T, half>::value;
    const T    MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
    for (int tid = threadIdx.x; tid < vocab_size_padded; tid += blockDim.x) {
        if (tid >= vocab_size) {
            logits[offset + tid] = -MAX_T_VAL;
        }
        else if (finish) {
            logits[offset + tid] = (tid == end_ids[bid]) ? MAX_T_VAL : -MAX_T_VAL;
        }
        else {
            if (bias != nullptr) {
                logits[offset + tid] += bias[tid];
            }
        }
    }
}

template<typename T>
void invokeAddBiasEndMask(T*           logits,
                          const T*     bias,
                          const int*   end_ids,
                          const bool*  finished,
                          const int    batch_size,
                          const int    vocab_size,
                          const int    vocab_size_padded,
                          cudaStream_t stream)
{
    dim3 grid(batch_size);
    dim3 block(min(vocab_size_padded, 1024));
    /*n is the vocab_size, e.g., 30000, 7000.... vocab_size is usually very big. */
    addBiasEndMask<<<grid, block, 0, stream>>>(logits, bias, end_ids, finished, vocab_size, vocab_size_padded);
}

template void invokeAddBiasEndMask(float*       logits,
                                   const float* bias,
                                   const int*   end_ids,
                                   const bool*  finished,
                                   const int    batch_size,
                                   const int    vocab_size,
                                   const int    vocab_size_padded,
                                   cudaStream_t stream);

template void invokeAddBiasEndMask(half*        logits,
                                   const half*  bias,
                                   const int*   end_ids,
                                   const bool*  finished,
                                   const int    batch_size,
                                   const int    vocab_size,
                                   const int    vocab_size_padded,
                                   cudaStream_t stream);

template<typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topk_stage1(const T* __restrict log_probs,
                            T*          tmp_log_probs,
                            int*        topk_tmp_id_buf,
                            T*          topk_tmp_val_buf,
                            const bool* finished,
                            const int   max_top_k,
                            const int*  top_ks,
                            const int   vocab_size,
                            const int*  end_ids,
                            const bool* skip_decode)
{
    typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage     temp_storage;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int batch_id = bid / BLOCKS_PER_BEAM_;  // row id for log_probs
    if (skip_decode != nullptr && skip_decode[batch_id]) {
        return;
    }
    const int block_lane = bid % BLOCKS_PER_BEAM_;                              // block id for a beam
    const int k          = (top_ks != nullptr) ? top_ks[batch_id] : max_top_k;  // batch_id = batch index

    const int tmp_log_buf_index  = batch_id * vocab_size;
    const int tmp_topk_buf_index = batch_id * BLOCKS_PER_BEAM_ * max_top_k + block_lane * k;

    TopK_2<T>  partial;
    const bool IS_FP16   = std::is_same<T, half>::value;
    const T    MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    if (finished != nullptr && finished[batch_id] == true) {
        if (tid < k) {
            const int index = tmp_topk_buf_index + tid;
            if (block_lane == 0 && tid == 0) {
                const int end_id        = end_ids[batch_id];
                topk_tmp_id_buf[index]  = tmp_log_buf_index + end_id;
                topk_tmp_val_buf[index] = log_probs[tmp_log_buf_index + end_id];
            }
            else {
                topk_tmp_id_buf[index]  = -1;
                topk_tmp_val_buf[index] = -MAX_T_VAL;
            }
        }
        return;
    }

    for (int elem_id = tid + block_lane * BLOCK_SIZE_; elem_id < vocab_size;
         elem_id += BLOCK_SIZE_ * BLOCKS_PER_BEAM_) {
        int index            = elem_id + tmp_log_buf_index;
        tmp_log_probs[index] = log_probs[index];
    }

    for (int ite = 0; ite < k; ite++) {
        partial.init();
#pragma unroll
        for (int elem_id = tid + block_lane * BLOCK_SIZE_; elem_id < vocab_size;
             elem_id += BLOCK_SIZE_ * BLOCKS_PER_BEAM_) {
            int index = elem_id + tmp_log_buf_index;
            partial.insert(tmp_log_probs[index], index);
        }

        TopK_2<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);

        if (tid == 0) {
            const int index         = tmp_topk_buf_index + ite;
            topk_tmp_id_buf[index]  = total.p;
            topk_tmp_val_buf[index] = total.u;
            tmp_log_probs[total.p]  = -MAX_T_VAL;
        }
        __syncthreads();
    }
}

template<typename T, int BLOCK_SIZE_, int BLOCKS_PER_BEAM_>
__global__ void topk_stage2_sampling(const int* __restrict topk_tmp_id_buf,
                                     T*             topk_tmp_val_buf,
                                     int*           ids,
                                     int*           sequence_length,
                                     bool*          finished,
                                     float*         cum_log_probs,
                                     float*         output_log_probs,
                                     float*         index_log_probs,
                                     int*           token_id_for_index_prob,
                                     const int      max_top_k,
                                     const int*     top_ks,
                                     const float    top_p,
                                     const float*   top_ps,
                                     curandState_t* curandstate,
                                     const int*     end_ids,
                                     const int      vocab_size,
                                     const bool*    skip_decode)
{
    const bool IS_FP16   = std::is_same<T, half>::value;
    const T    MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    const int tid      = threadIdx.x;
    const int batch_id = blockIdx.x;
    if (skip_decode != nullptr && skip_decode[batch_id]) {
        return;
    }

    const int   k              = (top_ks != nullptr) ? top_ks[batch_id] : max_top_k;
    const float prob_threshold = (top_ps != nullptr) ? top_ps[batch_id] : top_p;
    const int   size           = k * BLOCKS_PER_BEAM_;
    const int   stride         = max_top_k * BLOCKS_PER_BEAM_;

    typedef cub::BlockReduce<TopK_2<float>, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage         temp_storage;
    extern __shared__ char                               array[];
    __shared__ float                                     rand_num;
    __shared__ float                                     s_sum;
    __shared__ float                                     s_max;
    T*                                                   s_val = topk_tmp_val_buf + batch_id * stride;
    int*                                                 s_id  = reinterpret_cast<int*>(array);
    if (tid == 0) {
        s_sum = 0.0f;
    }
    TopK_2<float> partial;

    if (finished != nullptr && finished[batch_id] == true) {
        ids[batch_id] = end_ids[batch_id];
        return;
    }

    float* s_val2 = reinterpret_cast<float*>(s_id + k);
    for (int ite = 0; ite < k; ite++) {
        partial.init();
#pragma unroll
        for (int i = tid; i < size; i += BLOCK_SIZE_) {
            partial.insert((float)s_val[i], i);
        }

        TopK_2<float> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<float>);

        if (tid == 0) {
            if (ite == 0) {
                s_max = total.u;
            }
            s_id[ite]      = total.p;
            s_val[total.p] = -MAX_T_VAL;

            // when cum_log_probs are computed, topk_tmp_val_buf (logits_buf_) are already pre-processed by
            // softmax_kernel
            total.u = __expf(total.u - s_max);
            s_val2[ite] = total.u;
            s_sum += total.u;
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (token_id_for_index_prob && index_log_probs) {
            index_log_probs[batch_id] = -10000;
            int token_id = token_id_for_index_prob[batch_id];
            for (int i = 0; i < k; i++){
                if (topk_tmp_id_buf[batch_id * stride + s_id[i]] % vocab_size == token_id) {
                    index_log_probs[batch_id] = logf(s_val2[i]) - logf(s_sum);
                    break;
                }
            }
        }
    }

    if (tid == 0) {
        rand_num = (float)curand_uniform(curandstate + blockIdx.x) * prob_threshold * s_sum;
        for (int i = 0; i < k; i++) {
            float exp_logit = s_val2[i];
            rand_num        = rand_num - exp_logit;
            if (rand_num <= 0.0f || i == k - 1) {
                ids[batch_id] = topk_tmp_id_buf[batch_id * stride + s_id[i]] % vocab_size;
                if (cum_log_probs != nullptr || output_log_probs != nullptr) {
                    float log_prob = logf(exp_logit) - logf(s_sum);
                    if (cum_log_probs != nullptr) {
                        cum_log_probs[batch_id] += log_prob;
                    }
                    if (output_log_probs != nullptr) {
                        // 'output_log_probs' is the probability induced by the top-k sampling.
                        // We normalize the probability 'exp_logit' of the selected token by
                        // the probability 's_sum' of a set of top-k tokens, meaning the log_prob
                        // is the probability of the selected token, conditioned on the event that
                        // it is selected, i.e.,
                        //   log_prob = log P(i | i is in top-k) = log(exp_logit / s_sum).
                        output_log_probs[batch_id] = log_prob;
                    }
                }
                break;
            }
        }
        if (sequence_length != nullptr && finished != nullptr) {
            sequence_length[batch_id] = finished[batch_id] ? sequence_length[batch_id] : sequence_length[batch_id] + 1;
            finished[batch_id]        = ids[batch_id] == end_ids[batch_id] ? true : false;
        }
    }
}

#define CASE_K(K_MIN, K_MAX, BLOCK_SIZE_1_, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_)                                           \
    case K_MIN ... K_MAX:                                                                                              \
        topk_stage1<T, BLOCK_SIZE_1_, BLOCKS_PER_BEAM_>                                                                \
            <<<batch_size * BLOCKS_PER_BEAM_, BLOCK_SIZE_1_, 0, stream>>>(log_probs,                                   \
                                                                          temp_log_probs,                              \
                                                                          topk_tmp_id_buf,                             \
                                                                          topk_tmp_val_buf,                            \
                                                                          finished,                                    \
                                                                          max_top_k,                                   \
                                                                          top_ks,                                      \
                                                                          vocab_size,                                  \
                                                                          end_ids,                                     \
                                                                          skip_decode);                                \
        topk_stage2_sampling<T, BLOCK_SIZE_2_, BLOCKS_PER_BEAM_>                                                       \
            <<<batch_size, BLOCK_SIZE_2_, K_MAX * sizeof(int) + K_MAX * sizeof(float), stream>>>(topk_tmp_id_buf,      \
                                                                                                 topk_tmp_val_buf,     \
                                                                                                 ids,                  \
                                                                                                 sequence_length,      \
                                                                                                 finished,             \
                                                                                                 cum_log_probs,        \
                                                                                                 output_log_probs,     \
                                                                                                 index_log_probs,    \
                                                                                                 token_id_for_index_prob,    \
                                                                                                 max_top_k,            \
                                                                                                 top_ks,               \
                                                                                                 top_p,                \
                                                                                                 top_ps,               \
                                                                                                 curandstate,          \
                                                                                                 end_ids,              \
                                                                                                 vocab_size,           \
                                                                                                 skip_decode);         \
        break;

template<typename T>
void invokeBatchTopKSampling(void*          workspace,
                             size_t&        workspace_size,
                             const T*       log_probs,
                             int*           ids,
                             int*           sequence_length,
                             bool*          finished,
                             float*         cum_log_probs,
                             float*         output_log_probs,
                             float*         index_log_probs,
                             int*           token_id_for_index_prob,
                             curandState_t* curandstate,
                             const int      max_top_k,
                             const int*     top_ks,
                             const float    top_p,
                             const float*   top_ps,
                             const int      vocab_size_padded,
                             const int*     end_ids,
                             cudaStream_t   stream,
                             const int      batch_size,
                             const bool*    skip_decode)
{
    // Not allow an ambiguous inputs top_p and top_ps.
    assert(top_p == 1.0f || top_ps == nullptr);
    const int vocab_size              = vocab_size_padded;
    const int max_block_per_beam      = 8;
    int       temp_log_probs_buf_size = batch_size * vocab_size;                      // type float
    int       topk_tmp_ids_buf_size   = batch_size * max_top_k * max_block_per_beam;  // type int
    int       topk_tmp_val_buf_size   = batch_size * max_top_k * max_block_per_beam;  // type float

    // prevent memory misaligned address
    temp_log_probs_buf_size = (int)(ceil(temp_log_probs_buf_size / 4.)) * 4;
    topk_tmp_ids_buf_size   = (int)(ceil(topk_tmp_ids_buf_size / 4.)) * 4;
    topk_tmp_val_buf_size   = (int)(ceil(topk_tmp_val_buf_size / 4.)) * 4;

    if (workspace == nullptr) {
        workspace_size = sizeof(T) * temp_log_probs_buf_size + sizeof(int) * topk_tmp_ids_buf_size
                         + sizeof(T) * topk_tmp_val_buf_size;
        return;
    }

    T*   temp_log_probs   = (T*)workspace;
    int* topk_tmp_id_buf  = (int*)(temp_log_probs + temp_log_probs_buf_size);
    T*   topk_tmp_val_buf = (T*)(topk_tmp_id_buf + topk_tmp_ids_buf_size);

    switch (max_top_k) {
        CASE_K(1, 16, 128, 128, 8);
        CASE_K(17, 32, 256, 128, 8);
        CASE_K(33, 64, 256, 256, 8);
        CASE_K(65, 1024, 256, 256, 8);
        default:
            throw std::domain_error(fmtstr("top-k kernel supports 1<=k<=1024 but got k=%d", max_top_k));
    }
}

#undef CASE_K

template void invokeBatchTopKSampling(void*          workspace,
                                      size_t&        workspace_size,
                                      const float*   log_probs,
                                      int*           ids,
                                      int*           sequence_length,
                                      bool*          finished_buf,
                                      float*         cum_log_probs,
                                      float*         output_log_probs,
                                      float*         index_log_probs,
                                      int*           token_id_for_index_prob,
                                      curandState_t* curandstate,
                                      const int      max_top_k,
                                      const int*     top_ks,
                                      const float    top_p,
                                      const float*   top_ps,
                                      const int      vocab_size_padded,
                                      const int*     end_ids,
                                      cudaStream_t   stream,
                                      const int      batch_size,
                                      const bool*    skip_decode);

template void invokeBatchTopKSampling(void*          workspace,
                                      size_t&        workspace_size,
                                      const half*    log_probs,
                                      int*           ids,
                                      int*           sequence_length,
                                      bool*          finished_buf,
                                      float*         cum_log_probs,
                                      float*         output_log_probs,
                                      float*          index_log_probs,
                                      int*           token_id_for_index_prob,
                                      curandState_t* curandstate,
                                      const int      max_top_k,
                                      const int*     top_ks,
                                      const float    top_p,
                                      const float*   top_ps,
                                      const int      vocab_size_padded,
                                      const int*     end_ids,
                                      cudaStream_t   stream,
                                      const int      batch_size,
                                      const bool*    skip_decode);

template<typename T>
void invokeTopKSampling(void*          workspace,
                        size_t&        workspace_size,
                        const T*       log_probs,
                        int*           ids,
                        int*           sequence_length,
                        bool*          finished_buf,
                        float*         cum_log_probs,
                        float*         output_log_probs,
                        float*             index_log_probs,
                        int*           token_id_for_index_prob,
                        curandState_t* curandstate,
                        const int      top_k,
                        const float    top_p,
                        const int      vocab_size_padded,
                        const int*     end_ids,
                        cudaStream_t   stream,
                        const int      batch_size,
                        const bool*    skip_decode)
{
    invokeBatchTopKSampling(workspace,
                            workspace_size,
                            log_probs,
                            ids,
                            sequence_length,
                            finished_buf,
                            cum_log_probs,
                            output_log_probs,
                            index_log_probs,
                            token_id_for_index_prob,
                            curandstate,
                            top_k,
                            nullptr,
                            top_p,
                            nullptr,
                            vocab_size_padded,
                            end_ids,
                            stream,
                            batch_size,
                            skip_decode);
}

template void invokeTopKSampling(void*          workspace,
                                 size_t&        workspace_size,
                                 const float*   log_probs,
                                 int*           ids,
                                 int*           sequence_length,
                                 bool*          finished_buf,
                                 float*         cum_log_probs,
                                 float*         output_log_probs,
                                 float*         index_log_probs,
                                 int*           token_id_for_index_prob,
                                 curandState_t* curandstate,
                                 const int      top_k,
                                 const float    top_p,
                                 const int      vocab_size_padded,
                                 const int*     end_ids,
                                 cudaStream_t   stream,
                                 const int      batch_size,
                                 const bool*    skip_decode);

template void invokeTopKSampling(void*          workspace,
                                 size_t&        workspace_size,
                                 const half*    log_probs,
                                 int*           ids,
                                 int*           sequence_length,
                                 bool*          finished_buf,
                                 float*         cum_log_probs,
                                 float*         output_log_probs,
                                 float*          index_log_probs,
                                 int*           token_id_for_index_prob,
                                 curandState_t* curandstate,
                                 const int      top_k,
                                 const float    top_p,
                                 const int      vocab_size_padded,
                                 const int*     end_ids,
                                 cudaStream_t   stream,
                                 const int      batch_size,
                                 const bool*    skip_decode);

template<uint TOP_K_MAX>
__global__ void setup_topk_runtime_args(int    batch_size,
                                        uint   top_k,
                                        uint*  top_ks,
                                        int    top_ks_size,
                                        float  top_p,
                                        float* top_ps,
                                        int    top_ps_size,
                                        bool*  skip_decode)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = index; i < batch_size; i += gridDim.x * blockDim.x) {
        uint  k = top_ks_size > 1 ? top_ks[i] : top_k;
        float p = top_ps_size > 1 ? top_ps[i] : top_p;
        if (k == 0 && p == 0.0f) {
            // FT's topp implementation does not support topp = 0.0f, but it equivalent to greedy search.
            // So, we set the topk = 1 as an alternative solution.
            k = 1;
        }
        if (k > 0 && p == 0.0f) {
            // for compatibility <= FT5.0.
            // This case corresponds to the old topk sampling, which is equivalent to
            // the old topk_topp sampling with topp=1.0f. TopKSamplingLayer and
            // TopKTopPSamplingLayer are now merged by TopKSamplingLayer. Thus, we
            // replace the case topk>0 and topp=0.0f by topk>0 and topp=1.0f for the
            // compatibility.
            p = 1.0f;
        }
        // Clip k value. A topk sampling kernel supports up to TOP_K_MAX=64.
        top_ks[i] = k > TOP_K_MAX ? TOP_K_MAX : k;
        if (k > TOP_K_MAX) {
            printf("[WARNING] topk (%d) is larger than max supported number (%d) for token %d"
                   " clip to max supported number %d. \n",
                   k,
                   TOP_K_MAX,
                   i,
                   top_ks[i]);
        }
        // Clip p value if it is out of range. range = [0.0, 1.0].
        top_ps[i] = p < 0.0f ? 0.0f : (p > 1.0f ? 1.0f : p);
        if (p < 0.0f || p > 1.0f) {
            printf("[WARNING] topp (%f) is out of range ([0.0, 1.0f]) for token %d"
                   " clip to closest number %f.\n",
                   p,
                   i,
                   top_ps[i]);
        }
        skip_decode[i] = k == 0;
    }
}

void invokeSetupTopKRuntimeArgs(int    batch_size,
                                uint   top_k,
                                uint*  top_ks,
                                int    top_ks_size,
                                float  top_p,
                                float* top_ps,
                                int    top_ps_size,
                                bool*  skip_decode,
                                cudaStream_t stream)
{
    dim3 block(std::min((int)batch_size, 256));
    dim3 grid(div_up((int)batch_size, (int)block.x));
    // support top_k up to 1024.
    setup_topk_runtime_args<1024><<<grid, block, 0, stream>>>(batch_size,
                                top_k,
                                top_ks,
                                top_ks_size,
                                top_p,
                                top_ps,
                                top_ps_size,
                                skip_decode);

}

}  // namespace fastertransformer
