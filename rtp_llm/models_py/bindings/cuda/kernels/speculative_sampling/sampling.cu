#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/sampling.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/vec_dtypes.cuh"
#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/util.cuh"

#include <cstdint>
#include <numeric>
#include <cuda/std/limits>
#include <cub/block/block_adjacent_difference.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>

namespace rtp_llm {
using namespace cub;

template<typename BiasT>
__global__ void dsparkCombineLogitsKernel(const float* base,
                                          const BiasT* bias,
                                          const float* temperature,
                                          float*       output,
                                          int64_t      vocab,
                                          int64_t      base_row_stride) {
    const int64_t row = blockIdx.y;
    const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col < vocab) {
        // Do not replace division by reciprocal multiplication or remove the
        // intermediate FP32 addition rounding. No softmax or RNG changes here.
        const float base_value = base[row * base_row_stride + col];
        const float bias_value = static_cast<float>(bias[row * vocab + col]);
        const float temp       = temperature[row];
        float       sum, result;
        // This target is compiled with ftz=true. Explicit PTX without .ftz
        // preserves Torch's subnormal input/output behavior for this operation.
        asm("add.rn.f32 %0, %1, %2;" : "=f"(sum) : "f"(base_value), "f"(bias_value));
        asm("div.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(sum), "f"(temp));
        output[row * vocab + col] = result;
    }
}

template<typename BiasT>
cudaError_t invokeDSparkCombineLogits(const float* base,
                                      const BiasT* bias,
                                      const float* temperature,
                                      float*       output,
                                      int64_t      batch,
                                      int64_t      vocab,
                                      int64_t      base_row_stride,
                                      cudaStream_t stream) {
    if (batch == 0 || vocab == 0) {
        return cudaSuccess;
    }
    dsparkCombineLogitsKernel<<<dim3((vocab + 255) / 256, batch), 256, 0, stream>>>(
        base, bias, temperature, output, vocab, base_row_stride);
    return cudaGetLastError();
}

template cudaError_t invokeDSparkCombineLogits<float>(
    const float*, const float*, const float*, float*, int64_t, int64_t, int64_t, cudaStream_t);
template cudaError_t invokeDSparkCombineLogits<__nv_bfloat16>(
    const float*, const __nv_bfloat16*, const float*, float*, int64_t, int64_t, int64_t, cudaStream_t);

template<int BLOCK_THREADS>
__global__ void dsparkConfidenceKernel(const __nv_bfloat16* hidden,
                                       const int32_t*       anchors,
                                       const int32_t*       sampled_tokens,
                                       const __nv_bfloat16* markov_w1,
                                       const __nv_bfloat16* confidence_w,
                                       const __nv_bfloat16* confidence_b,
                                       float*               output,
                                       int64_t              gamma,
                                       int64_t              hidden_dim,
                                       int64_t              markov_rank) {
    using Reduce = cub::BlockReduce<float, BLOCK_THREADS>;
    __shared__ typename Reduce::TempStorage reduce_storage;

    const int64_t row      = static_cast<int64_t>(blockIdx.x);
    const int64_t batch_id = row / gamma;
    const int64_t position = row - batch_id * gamma;
    const int32_t previous = position == 0 ? anchors[batch_id] : sampled_tokens[row - 1];

    float partial = 0.0f;
    for (int64_t column = threadIdx.x; column < hidden_dim; column += BLOCK_THREADS) {
        partial += static_cast<float>(hidden[row * hidden_dim + column]) * static_cast<float>(confidence_w[column]);
    }
    const int64_t markov_offset = static_cast<int64_t>(previous) * markov_rank;
    for (int64_t column = threadIdx.x; column < markov_rank; column += BLOCK_THREADS) {
        partial += static_cast<float>(markov_w1[markov_offset + column])
                   * static_cast<float>(confidence_w[hidden_dim + column]);
    }
    const float sum = Reduce(reduce_storage).Sum(partial);
    if (threadIdx.x == 0) {
        // The BF16 confidence projection returns BF16 raw logits before the
        // reference head promotes them to FP32 for sigmoid/STS calibration.
        const float logit = __bfloat162float(__float2bfloat16_rn(sum + static_cast<float>(confidence_b[0])));
        output[row]       = 1.0f / (1.0f + expf(-logit));
    }
}

cudaError_t invokeDSparkConfidence(const __nv_bfloat16* hidden,
                                   const int32_t*       anchors,
                                   const int32_t*       sampled_tokens,
                                   const __nv_bfloat16* markov_w1,
                                   const __nv_bfloat16* confidence_w,
                                   const __nv_bfloat16* confidence_b,
                                   float*               output,
                                   int64_t              batch,
                                   int64_t              gamma,
                                   int64_t              hidden_dim,
                                   int64_t              markov_rank,
                                   cudaStream_t         stream) {
    if (batch == 0 || gamma == 0) {
        return cudaSuccess;
    }
    constexpr int threads = 256;
    dsparkConfidenceKernel<threads><<<batch * gamma, threads, 0, stream>>>(
        hidden, anchors, sampled_tokens, markov_w1, confidence_w, confidence_b, output, gamma, hidden_dim, markov_rank);
    return cudaGetLastError();
}

namespace {

constexpr int64_t kMaxDSparkPlanCandidates = 1024;
constexpr int64_t kMaxDSparkPlanBatch      = 256;

__global__ void dsparkVerifyPlanKernel(const float* confidence,
                                       int32_t*     verify_lengths,
                                       int32_t*     compact_to_dense,
                                       int64_t      batch,
                                       int64_t      gamma,
                                       int64_t      extra_budget) {
    __shared__ float   survival[kMaxDSparkPlanCandidates];
    __shared__ uint8_t selected[kMaxDSparkPlanCandidates];

    const int64_t candidate_count = batch * gamma;
    for (int64_t candidate = threadIdx.x; candidate < candidate_count; candidate += blockDim.x) {
        const int64_t request  = candidate / gamma;
        const int64_t position = candidate - request * gamma;
        float         value    = 1.0f;
        const int64_t base     = request * gamma;
        for (int64_t j = 0; j <= position; ++j) {
            const float conditional = confidence[base + j];
            value *= isfinite(conditional) ? fminf(fmaxf(conditional, 0.0f), 1.0f) : 0.0f;
        }
        survival[candidate] = value;
        selected[candidate] = 0;
    }
    __syncthreads();

    // Current production batches are small (B<=28, gamma=7).  Keeping the
    // global selection in one CTA avoids a host sync and makes ties stable.
    if (threadIdx.x == 0) {
        for (int64_t request = 0; request < batch; ++request) {
            verify_lengths[request] = 1;
        }
        for (int64_t picked = 0; picked < extra_budget; ++picked) {
            int64_t best       = -1;
            float   best_value = -1.0f;
            for (int64_t candidate = 0; candidate < candidate_count; ++candidate) {
                if (selected[candidate]) {
                    continue;
                }
                const float   value         = survival[candidate];
                const int64_t request       = candidate / gamma;
                const int64_t position      = candidate - request * gamma;
                const int64_t best_request  = best < 0 ? batch : best / gamma;
                const int64_t best_position = best < 0 ? gamma : best - best_request * gamma;
                // Match the reference scheduler's stable ordering: survival
                // first, then the earlier prefix position, then request id.
                // Request-major tie breaking can spend the whole budget on
                // one request when confidence saturates at exactly one.
                if (value > best_value
                    || (value == best_value
                        && (position < best_position || (position == best_position && request < best_request)))) {
                    best       = candidate;
                    best_value = value;
                }
            }
            if (best < 0) {
                break;
            }
            selected[best] = 1;
            ++verify_lengths[best / gamma];
        }

        int64_t compact_row = 0;
        for (int64_t request = 0; request < batch; ++request) {
            const int64_t rows       = verify_lengths[request];
            const int64_t dense_base = request * (gamma + 1);
            for (int64_t row = 0; row < rows; ++row) {
                compact_to_dense[compact_row++] = static_cast<int32_t>(dense_base + row);
            }
        }
    }
}

}  // namespace

cudaError_t invokeDSparkVerifyPlan(const float* confidence,
                                   int32_t*     verify_lengths,
                                   int32_t*     compact_to_dense,
                                   int64_t      batch,
                                   int64_t      gamma,
                                   int64_t      extra_budget,
                                   cudaStream_t stream) {
    if (batch == 0) {
        return cudaSuccess;
    }
    if (batch < 0 || batch > kMaxDSparkPlanBatch || gamma <= 0 || batch * gamma > kMaxDSparkPlanCandidates
        || extra_budget < 0 || extra_budget > batch * gamma) {
        return cudaErrorInvalidValue;
    }
    dsparkVerifyPlanKernel<<<1, 256, 0, stream>>>(
        confidence, verify_lengths, compact_to_dense, batch, gamma, extra_budget);
    return cudaGetLastError();
}

constexpr BlockScanAlgorithm   SCAN_ALGO   = BLOCK_SCAN_WARP_SCANS;
constexpr BlockReduceAlgorithm REDUCE_ALGO = BLOCK_REDUCE_WARP_REDUCTIONS;

#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120100)
#define FLASHINFER_CUB_SUBTRACTLEFT_DEFINED
#endif

template<typename T>
struct Pair {
    T   value;
    int count;

    __device__ Pair operator+(const Pair& other) const {
        return {value + other.value, count + other.count};
    }
    __device__ Pair& operator+=(const Pair& other) {
        value += other.value;
        count += other.count;
        return *this;
    }
};

struct BoolDiffOp {
    __device__ __forceinline__ bool operator()(const bool& lhs, const bool& rhs) const {
        return lhs != rhs;
    }
};

template<typename T, uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM, BlockReduceAlgorithm REDUCE_ALGORITHM>
struct SamplingTempStorage {
    union {
        T                                                                     deterministic_scan[BLOCK_THREADS / 32];
        typename BlockScan<T, BLOCK_THREADS, SCAN_ALGORITHM>::TempStorage     scan;
        typename BlockReduce<T, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce;
        typename BlockReduce<Pair<T>, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce_pair;
        typename BlockAdjacentDifference<bool, BLOCK_THREADS>::TempStorage          adj_diff;
    } block_prim;
    struct {
        int32_t sampled_id;
        union {
            T       value;
            Pair<T> pair;
            T       max_p;
        } block_aggregate;
    };
};

/*!
 * \brief Deterministic inclusive scan implementation, use Belloch scan algorithm.
 * \note This implementation is slower than the cub::BlockScan, but it is deterministic.
 */
template<uint32_t             VEC_SIZE,
         uint32_t             BLOCK_THREADS,
         BlockScanAlgorithm   SCAN_ALGORITHM,
         BlockReduceAlgorithm REDUCE_ALGORITHM,
         typename T>
__device__ __forceinline__ void
DeterministicInclusiveSum(const T*                                                                 in_data,
                          T*                                                                       out_data,
                          SamplingTempStorage<T, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>* temp_storage) {
    T* smem_prefix_sum = temp_storage->block_prim.deterministic_scan;
    T  thread_data[VEC_SIZE];
    T  thread_sum = 0;
#pragma unroll
    for (uint32_t i = 0; i < VEC_SIZE; ++i) {
        thread_sum += in_data[i];
        thread_data[i] = thread_sum;
    }

    T thread_exclusive_prefix_sum = thread_sum;

#pragma unroll
    for (uint32_t offset = 1; offset < 32; offset *= 2) {
        T tmp = __shfl_up_sync(0xffffffff, thread_exclusive_prefix_sum, offset);
        if ((threadIdx.x + 1) % (offset * 2) == 0) {
            thread_exclusive_prefix_sum += tmp;
        }
    }

    T warp_sum = __shfl_sync(0xffffffff, thread_exclusive_prefix_sum, threadIdx.x | 0xffffffff);
    if (threadIdx.x % 32 == 31) {
        thread_exclusive_prefix_sum = 0;
    }

#pragma unroll
    for (uint32_t offset = 16; offset >= 1; offset /= 2) {
        T tmp = __shfl_xor_sync(0xffffffff, thread_exclusive_prefix_sum, offset);
        if ((threadIdx.x + 1) % (offset * 2) == 0) {
            thread_exclusive_prefix_sum = tmp + thread_exclusive_prefix_sum;
        }
        if ((threadIdx.x + 1) % (offset * 2) == offset) {
            thread_exclusive_prefix_sum = tmp;
        }
    }

    smem_prefix_sum[threadIdx.x / 32] = warp_sum;
    __syncthreads();

    if (threadIdx.x < 32) {
        T warp_exclusive_prefix_sum = (threadIdx.x < BLOCK_THREADS / 32) ? smem_prefix_sum[threadIdx.x] : 0;

#pragma unroll
        for (uint32_t offset = 1; offset < 32; offset *= 2) {
            T tmp = __shfl_up_sync(0xffffffff, warp_exclusive_prefix_sum, offset);
            if ((threadIdx.x + 1) % (offset * 2) == 0) {
                warp_exclusive_prefix_sum += tmp;
            }
        }

        if (threadIdx.x % 32 == 31) {
            warp_exclusive_prefix_sum = 0;
        }

#pragma unroll
        for (uint32_t offset = 16; offset >= 1; offset /= 2) {
            T tmp = __shfl_xor_sync(0xffffffff, warp_exclusive_prefix_sum, offset);
            if ((threadIdx.x + 1) % (offset * 2) == 0) {
                warp_exclusive_prefix_sum = tmp + warp_exclusive_prefix_sum;
            }
            if ((threadIdx.x + 1) % (offset * 2) == offset) {
                warp_exclusive_prefix_sum = tmp;
            }
        }
        if (threadIdx.x < BLOCK_THREADS / 32) {
            smem_prefix_sum[threadIdx.x] = warp_exclusive_prefix_sum;
        }
    }
    __syncthreads();

#pragma unroll
    for (uint32_t i = 0; i < VEC_SIZE; ++i) {
        out_data[i] = smem_prefix_sum[threadIdx.x / 32] + thread_exclusive_prefix_sum + thread_data[i];
    }
}

template<uint32_t             VEC_SIZE,
         uint32_t             BLOCK_THREADS,
         BlockScanAlgorithm   SCAN_ALGORITHM,
         BlockReduceAlgorithm REDUCE_ALGORITHM,
         bool                 DETERMINISTIC,
         typename T,
         typename Predicate>
__device__ __forceinline__ void
DeviceSamplingFromProb(uint32_t                                                                 i,
                       uint32_t                                                                 d,
                       Predicate                                                                pred,
                       T                                                                        u,
                       flashinfer::vec_t<T, VEC_SIZE>                                           prob_vec,
                       T&                                                                       aggregate,
                       SamplingTempStorage<T, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>* temp_storage) {
    const uint32_t tx = threadIdx.x;
    T              prob_greater_than_threshold[VEC_SIZE];
    T              inclusive_cdf[VEC_SIZE];
    bool           greater_than_u[VEC_SIZE], valid[VEC_SIZE];
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        prob_greater_than_threshold[j] = pred(prob_vec[j]) ? prob_vec[j] : T(0);
        valid[j]                       = pred(prob_vec[j]) && (i * BLOCK_THREADS + tx) * VEC_SIZE < d;
    }
    T aggregate_local = BlockReduce<T, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage->block_prim.reduce)
                            .Sum<VEC_SIZE>(prob_greater_than_threshold);
    if (tx == 0) {
        temp_storage->block_aggregate.value = aggregate_local;
    }
    __syncthreads();
    aggregate_local = temp_storage->block_aggregate.value;

    if (aggregate + aggregate_local > u) {
        if constexpr (DETERMINISTIC) {
            DeterministicInclusiveSum<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM, T>(
                prob_greater_than_threshold, inclusive_cdf, temp_storage);
        } else {
            BlockScan<T, BLOCK_THREADS, SCAN_ALGORITHM>(temp_storage->block_prim.scan)
                .InclusiveSum<VEC_SIZE>(prob_greater_than_threshold, inclusive_cdf);

            __syncthreads();
        }

#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
            greater_than_u[j] = (inclusive_cdf[j] + aggregate > u) && valid[j];
        }

        bool greater_than_u_diff[VEC_SIZE];
#ifdef FLASHINFER_CUB_SUBTRACTLEFT_DEFINED
        BlockAdjacentDifference<bool, BLOCK_THREADS>(temp_storage->block_prim.adj_diff)
            .SubtractLeft<VEC_SIZE>(greater_than_u, greater_than_u_diff, BoolDiffOp());
#else
        BlockAdjacentDifference<bool, BLOCK_THREADS>(temp_storage->block_prim.adj_diff)
            .FlagHeads<VEC_SIZE>(greater_than_u_diff, greater_than_u, BoolDiffOp(), 0);
#endif
        __syncthreads();

#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
            if (greater_than_u_diff[j]) {
                atomicMin(&(temp_storage->sampled_id), (i * BLOCK_THREADS + tx) * VEC_SIZE + j);
            }
        }
        __syncthreads();
    }
    aggregate += aggregate_local;
}

template<uint32_t             BLOCK_THREADS,
         BlockScanAlgorithm   SCAN_ALGORITHM,
         BlockReduceAlgorithm REDUCE_ALGORITHM,
         uint32_t             VEC_SIZE,
         bool                 DETERMINISTIC,
         typename DType,
         typename IdType>
__global__ void rejection_sampling_kernel(DType*     draft_probs,
                                          IdType*    draft_token_ids,
                                          DType*     uniform_samples,
                                          DType*     target_probs,
                                          IdType*    target_token_ids,
                                          int        target_token_stride,
                                          IdType*    output_token_ids,
                                          IdType*    output_accepted_token_num,
                                          bool*      do_sample,
                                          bool       deterministic_draft,
                                          int        batch_size,
                                          int        num_speculative_tokens,
                                          int        target_vocab_size,
                                          bool       sampled_draft,
                                          bool*      success,
                                          const int* active_verify_lengths) {
    const uint32_t bx = blockIdx.x, tx = threadIdx.x;
    const uint32_t row_idx = bx;

    extern __shared__ __align__(alignof(SamplingTempStorage<DType, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
        uint8_t       smem_sampling[];
    auto&             temp_storage =
        reinterpret_cast<SamplingTempStorage<DType, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(smem_sampling);

    if (row_idx >= batch_size) {
        return;
    }
    const int active_rows = active_verify_lengths ? active_verify_lengths[row_idx] : num_speculative_tokens + 1;

    if (deterministic_draft) {
        if (tx == 0) {
            int pos = num_speculative_tokens;
            for (int i = 0; i < num_speculative_tokens; ++i) {
                IdType draft_id  = draft_token_ids[row_idx * num_speculative_tokens + i];
                IdType target_id = target_token_ids[(row_idx * (num_speculative_tokens + 1) + i) * target_token_stride
                                                    + target_token_stride - 1];
                if (success && i < active_rows
                    && (draft_id < 0 || draft_id >= target_vocab_size || target_id < 0
                        || target_id >= target_vocab_size)) {
                    success[row_idx]                   = false;
                    output_accepted_token_num[row_idx] = 1;
                    return;
                }
                if (target_id == draft_id) {
                    output_token_ids[row_idx * (num_speculative_tokens + 1) + i] = draft_id;
                } else {
                    pos                                                          = i;
                    output_token_ids[row_idx * (num_speculative_tokens + 1) + i] = target_id;
                    for (int j = i + 1; j < num_speculative_tokens + 1; ++j) {
                        output_token_ids[row_idx * (num_speculative_tokens + 1) + j] = -1;
                    }
                    break;
                }
            }
            output_accepted_token_num[row_idx] = pos + 1;
            if (pos == num_speculative_tokens) {
                output_token_ids[row_idx * (num_speculative_tokens + 1) + pos] =
                    target_token_ids[(row_idx * (num_speculative_tokens + 1) + pos) * target_token_stride
                                     + target_token_stride - 1];
            }
        }
        return;
    }

    // Preserve the legacy MTP contract unless the caller supplies an actual
    // sampled proposal distribution. DSpARK opts into exact q/p rejection.
    __shared__ int  s_pos;
    __shared__ bool s_all_same_token;
    __shared__ bool s_greedy_exact_done;

    if (tx == 0) {
        bool all_same_token    = true;
        bool greedy_exact_done = false;
        int  pos               = num_speculative_tokens;
        for (int i = 0; i < num_speculative_tokens; ++i) {
            IdType draft_id  = draft_token_ids[row_idx * num_speculative_tokens + i];
            IdType target_id = target_token_ids[(row_idx * (num_speculative_tokens + 1) + i) * target_token_stride
                                                + target_token_stride - 1];
            if (success
                && (draft_id < 0 || draft_id >= target_vocab_size || target_id < 0 || target_id >= target_vocab_size)) {
                if (i < active_rows) {
                    success[row_idx] = false;
                }
                pos               = i;
                all_same_token    = false;
                greedy_exact_done = true;
                break;
            }

            float q = target_probs[(row_idx * (num_speculative_tokens + 1) + i) * target_vocab_size + draft_id],
                  p = draft_probs[(row_idx * num_speculative_tokens + i) * target_vocab_size + draft_id];
            DType u = uniform_samples[row_idx * (num_speculative_tokens + 1) + i];

            bool same_token = target_id == draft_id;
            // For sampled requests the target draw is independent evidence,
            // not an additional acceptance event. Always test q/p, even when
            // that draw happens to equal the draft token.
            if (!sampled_draft) {
                // Historical MTP proposals are argmax/top-k selections with
                // softmax scores, not draws from those scores. Do not silently
                // reinterpret that existing contract as probability sampling.
                if (same_token || (do_sample[row_idx] && u * p < q)) {
                    output_token_ids[row_idx * (num_speculative_tokens + 1) + i] = draft_id;
                    all_same_token                                               = all_same_token && same_token;
                } else {
                    pos = i;
                    break;
                }
            } else if (!do_sample[row_idx] && same_token) {
                output_token_ids[row_idx * (num_speculative_tokens + 1) + i] = draft_id;
            } else if (!do_sample[row_idx]) {
                // Greedy target decoding verifies by exact token match.  On
                // the first mismatch emit the target token and terminate the
                // speculative block; q-p rejection sampling is only valid for
                // stochastic requests.
                pos                                                          = i;
                output_token_ids[row_idx * (num_speculative_tokens + 1) + i] = target_id;
                for (int j = i + 1; j < num_speculative_tokens + 1; ++j) {
                    output_token_ids[row_idx * (num_speculative_tokens + 1) + j] = -1;
                }
                all_same_token    = false;
                greedy_exact_done = true;
                break;
            } else if (u * p < q) {
                output_token_ids[row_idx * (num_speculative_tokens + 1) + i] = draft_id;
                all_same_token                                               = all_same_token && same_token;
            } else {
                pos            = i;
                all_same_token = false;
                break;
            }
        }

        output_accepted_token_num[row_idx] = pos + 1;

        if (all_same_token) {
            IdType bonus_token_id =
                target_token_ids[(row_idx * (num_speculative_tokens + 1) + pos) * target_token_stride
                                 + target_token_stride - 1];
            output_token_ids[row_idx * (num_speculative_tokens + 1) + pos] = bonus_token_id;
        }

        s_pos               = pos;
        s_all_same_token    = all_same_token;
        s_greedy_exact_done = greedy_exact_done;
    }
    __syncthreads();

    if (s_all_same_token || s_greedy_exact_done) {
        return;
    }
    int pos = s_pos;

    // sample from relu(target_probs - draft_probs)
    DType                              sum_relu_q_minus_p(0);
    flashinfer::vec_t<DType, VEC_SIZE> q_vec, p_vec;
    DType                              relu_q_minus_p[VEC_SIZE];
    for (uint32_t i = 0; i < flashinfer::ceil_div(target_vocab_size, BLOCK_THREADS * VEC_SIZE); ++i) {
        q_vec.fill(DType(0));
        p_vec.fill(DType(0));
        if ((i * BLOCK_THREADS + tx) * VEC_SIZE < target_vocab_size) {
            q_vec.load(target_probs + (row_idx * (num_speculative_tokens + 1) + pos) * target_vocab_size
                       + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
            if (pos != num_speculative_tokens) {
                // there is no draft_probs for the bonus token
                p_vec.load(draft_probs + (row_idx * num_speculative_tokens + pos) * target_vocab_size
                           + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
            }
        }
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
            relu_q_minus_p[j] = max(q_vec[j] - p_vec[j], DType(0));
        }
        sum_relu_q_minus_p += BlockReduce<DType, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                                  .Sum<VEC_SIZE>(relu_q_minus_p);
        __syncthreads();
    }
    if (tx == 0) {
        temp_storage.block_aggregate.value = sum_relu_q_minus_p;
    }
    // init the first rejected token to (d - 1)
    temp_storage.sampled_id = success ? target_vocab_size : target_vocab_size - 1;
    __syncthreads();
    sum_relu_q_minus_p = temp_storage.block_aggregate.value;
    if (success && pos < active_rows && (!isfinite(sum_relu_q_minus_p) || sum_relu_q_minus_p <= DType(0))) {
        if (tx == 0) {
            success[row_idx] = false;
        }
        return;
    }
    DType u = uniform_samples[row_idx * (num_speculative_tokens + 1) + min(pos + 1, num_speculative_tokens)]
              * sum_relu_q_minus_p;

    DType aggregate_relu_q_minus_p(0);
    for (uint32_t i = 0; i < flashinfer::ceil_div(target_vocab_size, BLOCK_THREADS * VEC_SIZE); ++i) {
        q_vec.fill(DType(0));
        p_vec.fill(DType(0));
        if ((i * BLOCK_THREADS + tx) * VEC_SIZE < target_vocab_size) {
            q_vec.load(target_probs + (row_idx * (num_speculative_tokens + 1) + pos) * target_vocab_size
                       + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
            if (pos != num_speculative_tokens) {
                // there is no draft_probs for the bonus token
                p_vec.load(draft_probs + (row_idx * num_speculative_tokens + pos) * target_vocab_size
                           + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
            }
        }

        flashinfer::vec_t<DType, VEC_SIZE> relu_q_minus_p_vec;
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
            relu_q_minus_p_vec[j] = max(q_vec[j] - p_vec[j], DType(0));
        }

        DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM, DETERMINISTIC, DType>(
            i,
            target_vocab_size,
            [&](DType x) { return x > 0; },
            u,
            relu_q_minus_p_vec,
            aggregate_relu_q_minus_p,
            &temp_storage);
        if (aggregate_relu_q_minus_p > u) {
            break;
        }
    }
    __syncthreads();
    if (tx == 0) {
        // set the first rejected token
        if (success && temp_storage.sampled_id >= target_vocab_size) {
            if (pos < active_rows) {
                success[row_idx] = false;
            }
            output_token_ids[row_idx * (num_speculative_tokens + 1) + pos] = 0;
        } else {
            output_token_ids[row_idx * (num_speculative_tokens + 1) + pos] = temp_storage.sampled_id;
        }
        // pad remaining tokens with -1
        for (int p = pos + 1; p < num_speculative_tokens + 1; ++p) {
            output_token_ids[row_idx * (num_speculative_tokens + 1) + p] = -1;
        }
    }
}

// Each CTA owns one (request, token row, vocab tile) scratch record. No CTA
// here modifies success or output tokens, and every record is overwritten.
template<int THREADS, typename DType, typename IdType>
__global__ void validateRejectionProbabilityTiles(const DType*  draft_probs,
                                                  const DType*  target_probs,
                                                  const int*    active_verify_lengths,
                                                  const IdType* accepted_lengths,
                                                  const bool*   success,
                                                  float*        workspace,
                                                  int           steps,
                                                  int           vocab,
                                                  int           tiles) {
    const int     batch = blockIdx.x, row = blockIdx.y, tile = blockIdx.z;
    const int     cap    = active_verify_lengths ? active_verify_lengths[batch] : steps + 1;
    const int     rows   = min(static_cast<int>(accepted_lengths[batch]), cap);
    const int64_t record = (static_cast<int64_t>(batch) * (steps + 1) + row) * tiles + tile;
    if (!success[batch] || cap < 1 || cap > steps + 1 || row >= rows) {
        if (threadIdx.x == 0) {
            for (int field = 0; field < 4; ++field) {
                workspace[record * 4 + field] = 0.0f;
            }
        }
        return;
    }
    using Reduce = cub::BlockReduce<DType, THREADS>;
    __shared__ typename Reduce::TempStorage storage;
    for (int plane = 0; plane < 2; ++plane) {
        DType mass               = 0;
        bool  finite_nonnegative = true;
        if (plane == 0 || row < steps) {
            const DType* probabilities = plane == 0 ?
                                             target_probs + (static_cast<int64_t>(batch) * (steps + 1) + row) * vocab :
                                             draft_probs + (static_cast<int64_t>(batch) * steps + row) * vocab;
            const int    start         = tile * kRejectionValidationTileSize;
            const int    end           = min(start + kRejectionValidationTileSize, vocab);
            for (int column = start + threadIdx.x; column < end; column += THREADS) {
                const DType probability = probabilities[column];
                finite_nonnegative      = finite_nonnegative && isfinite(probability) && probability >= DType(0);
                mass += probability;
            }
        }
        const int   invalid = __syncthreads_count(!finite_nonnegative);
        const DType total   = Reduce(storage).Sum(mass);
        if (threadIdx.x == 0) {
            workspace[record * 4 + plane * 2]     = total;
            workspace[record * 4 + plane * 2 + 1] = invalid != 0 ? 1.0f : 0.0f;
        }
        __syncthreads();
    }
}

// Reduce only the small scratch prefix the rejection result can commit.
// This is the sole validation writer of each request's status/output shape.
template<int THREADS, typename IdType>
__global__ void finishRejectionValidation(const float* workspace,
                                          const bool*  target_success,
                                          const int*   active_verify_lengths,
                                          IdType*      output_tokens,
                                          IdType*      accepted_lengths,
                                          bool*        success,
                                          int          steps,
                                          int          vocab,
                                          int          tiles) {
    const int batch = blockIdx.x;
    const int cap   = active_verify_lengths ? active_verify_lengths[batch] : steps + 1;
    const int rows  = min(static_cast<int>(accepted_lengths[batch]), cap);
    using Reduce    = cub::BlockReduce<float, THREADS>;
    __shared__ typename Reduce::TempStorage storage;
    __shared__ bool                         valid;
    if (threadIdx.x == 0) {
        valid = success[batch] && cap >= 1 && cap <= steps + 1 && rows >= 1;
    }
    __syncthreads();
    for (int row = 0; row < rows && valid; ++row) {
        for (int plane = 0; plane < (row < steps ? 2 : 1); ++plane) {
            float         mass        = 0;
            bool          tiles_valid = true;
            const int64_t base        = (static_cast<int64_t>(batch) * (steps + 1) + row) * tiles * 4 + plane * 2;
            for (int tile = threadIdx.x; tile < tiles; tile += THREADS) {
                mass += workspace[base + tile * 4];
                tiles_valid = tiles_valid && workspace[base + tile * 4 + 1] == 0.0f;
            }
            const int   invalid = __syncthreads_count(!tiles_valid);
            const float total   = Reduce(storage).Sum(mass);
            if (threadIdx.x == 0) {
                const auto token = output_tokens[batch * (steps + 1) + row];
                valid = valid && invalid == 0 && isfinite(total) && total > 0.0f && token >= 0 && token < vocab
                        && (plane != 0 || !target_success || target_success[batch * (steps + 1) + row]);
            }
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) {
        success[batch] = valid;
        if (!valid) {
            // Keep a safe active shape until dispatch consumes success. Failed
            // requests never commit this placeholder or use it as an anchor.
            accepted_lengths[batch] = 1;
            for (int row = 0; row <= steps; ++row) {
                output_tokens[batch * (steps + 1) + row] = 0;
            }
        }
    }
}

template<typename DType, typename IdType>
cudaError_t invokeRejectionSampling(DType*       draft_probs,
                                    IdType*      draft_token_ids,
                                    DType*       uniform_samples,
                                    DType*       target_probs,
                                    IdType*      target_token_ids,
                                    int          target_token_stride,
                                    IdType*      output_token_ids,
                                    IdType*      output_accepted_token_num,
                                    bool*        do_sample,
                                    bool         deterministic_draft,
                                    int          batch_size,
                                    int          num_speculative_tokens,
                                    int          target_vocab_size,
                                    cudaStream_t stream,
                                    bool         sampled_draft,
                                    bool*        success,
                                    const bool*  target_success,
                                    const int*   active_verify_lengths,
                                    float*       validation_workspace) {
    if (batch_size == 0) {
        return cudaSuccess;
    }
    if (success && !validation_workspace) {
        return cudaErrorInvalidValue;
    }

    constexpr uint32_t BLOCK_THREADS = 1024;
    const uint32_t     vec_size      = std::gcd(16 / sizeof(DType), target_vocab_size);

    const uint32_t smem_size = sizeof(SamplingTempStorage<DType, BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);
    dim3           nblks(batch_size);
    dim3           nthrs(BLOCK_THREADS);

    void* args[] = {&draft_probs,
                    &draft_token_ids,
                    &uniform_samples,
                    &target_probs,
                    &target_token_ids,
                    &target_token_stride,
                    &output_token_ids,
                    &output_accepted_token_num,
                    &do_sample,
                    &deterministic_draft,
                    &batch_size,
                    &num_speculative_tokens,
                    &target_vocab_size,
                    &sampled_draft,
                    &success,
                    &active_verify_lengths};

    DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
        auto kernel = rejection_sampling_kernel<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, VEC_SIZE, false, DType, IdType>;
        FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    });

    if (success) {
        const int tiles = (target_vocab_size + kRejectionValidationTileSize - 1) / kRejectionValidationTileSize;
        validateRejectionProbabilityTiles<256>
            <<<dim3(batch_size, num_speculative_tokens + 1, tiles), 256, 0, stream>>>(draft_probs,
                                                                                      target_probs,
                                                                                      active_verify_lengths,
                                                                                      output_accepted_token_num,
                                                                                      success,
                                                                                      validation_workspace,
                                                                                      num_speculative_tokens,
                                                                                      target_vocab_size,
                                                                                      tiles);
        FLASHINFER_CUDA_CALL(cudaGetLastError());
        finishRejectionValidation<256><<<batch_size, 256, 0, stream>>>(validation_workspace,
                                                                       target_success,
                                                                       active_verify_lengths,
                                                                       output_token_ids,
                                                                       output_accepted_token_num,
                                                                       success,
                                                                       num_speculative_tokens,
                                                                       target_vocab_size,
                                                                       tiles);
        FLASHINFER_CUDA_CALL(cudaGetLastError());
    }

    return cudaSuccess;
}

#define INSTANTIATE_REJECTION_SAMPLING(DType, IdType)                                                                  \
    template cudaError_t invokeRejectionSampling(DType*       draft_probs,                                             \
                                                 IdType*      draft_token_ids,                                         \
                                                 DType*       uniform_samples,                                         \
                                                 DType*       target_probs,                                            \
                                                 IdType*      target_token_ids,                                        \
                                                 int          target_token_stride,                                     \
                                                 IdType*      output_token_ids,                                        \
                                                 IdType*      output_accepted_token_num,                               \
                                                 bool*        do_sample,                                               \
                                                 bool         deterministic_draft,                                     \
                                                 int          batch_size,                                              \
                                                 int          num_speculative_tokens,                                  \
                                                 int          target_vocab_size,                                       \
                                                 cudaStream_t stream,                                                  \
                                                 bool         sampled_draft,                                           \
                                                 bool*        success,                                                 \
                                                 const bool*  target_success,                                          \
                                                 const int*   active_verify_lengths,                                   \
                                                 float*       validation_workspace);

INSTANTIATE_REJECTION_SAMPLING(float, int);
}  // namespace rtp_llm
