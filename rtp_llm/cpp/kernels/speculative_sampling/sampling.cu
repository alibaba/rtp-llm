#include "rtp_llm/cpp/kernels/speculative_sampling/sampling.h"
#include "rtp_llm/cpp/kernels/speculative_sampling/vec_dtypes.cuh"
#include "rtp_llm/cpp/kernels/speculative_sampling/util.cuh"

#include <cstdint>
#include <numeric>
#include <cuda/std/limits>
#include <cub/block/block_adjacent_difference.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>

namespace rtp_llm {
using namespace cub;

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
__global__ void rejection_sampling_kernel(DType*  draft_probs,
                                          IdType* draft_token_ids,
                                          DType*  uniform_samples,
                                          DType*  target_probs,
                                          IdType* target_token_ids,
                                          int     target_token_stride,
                                          IdType* output_token_ids,
                                          IdType* output_accepted_token_num,
                                          bool*   do_sample,
                                          int     batch_size,
                                          int     num_speculative_tokens,
                                          int     target_vocab_size) {
    const uint32_t bx = blockIdx.x, tx = threadIdx.x;
    const uint32_t row_idx = bx;

    extern __shared__ __align__(alignof(SamplingTempStorage<DType, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
        uint8_t       smem_sampling[];
    auto&             temp_storage =
        reinterpret_cast<SamplingTempStorage<DType, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(smem_sampling);

    if (row_idx >= batch_size) {
        return;
    }

    bool all_same_token = true;
    int  pos            = num_speculative_tokens;
    for (int i = 0; i < num_speculative_tokens; ++i) {
        IdType draft_id  = draft_token_ids[row_idx * num_speculative_tokens + i];
        IdType target_id = target_token_ids[(row_idx * (num_speculative_tokens + 1) + i) * target_token_stride
                                            + target_token_stride - 1];

        float q = target_probs[(row_idx * (num_speculative_tokens + 1) + i) * target_vocab_size + draft_id],
              p = draft_probs[(row_idx * num_speculative_tokens + i) * target_vocab_size + draft_id];
        DType u = uniform_samples[row_idx * (num_speculative_tokens + 1) + i];

        bool same_token = target_id == draft_id;
        if (same_token || (do_sample[row_idx] && u * p < q)) {
            output_token_ids[row_idx * (num_speculative_tokens + 1) + i] = draft_id;
            all_same_token                                               = all_same_token && same_token;
        } else {
            pos = i;
            break;
        }
    }

    if (tx == 0) {
        output_accepted_token_num[row_idx] += pos + 1;
    }

    if (all_same_token) {
        IdType bonus_token_id = target_token_ids[(row_idx * (num_speculative_tokens + 1) + pos) * target_token_stride
                                                 + target_token_stride - 1];
        output_token_ids[row_idx * (num_speculative_tokens + 1) + pos] = bonus_token_id;
        return;
    }

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
    temp_storage.sampled_id = target_vocab_size - 1;
    __syncthreads();
    sum_relu_q_minus_p = temp_storage.block_aggregate.value;
    DType u            = uniform_samples[row_idx * (num_speculative_tokens + 1) + min(pos + 1, num_speculative_tokens)]
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
    // set the first rejected token
    output_token_ids[row_idx * (num_speculative_tokens + 1) + pos] = temp_storage.sampled_id;
    // move to the next token
    pos++;

    // pad remaining tokens with -1
    for (; pos < num_speculative_tokens + 1; ++pos) {
        output_token_ids[row_idx * (num_speculative_tokens + 1) + pos] = -1;
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
                                    int          batch_size,
                                    int          num_speculative_tokens,
                                    int          target_vocab_size,
                                    cudaStream_t stream) {
    if (batch_size == 0) {
        return cudaSuccess;
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
                    &batch_size,
                    &num_speculative_tokens,
                    &target_vocab_size};

    DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
        auto kernel = rejection_sampling_kernel<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, VEC_SIZE, false, DType, IdType>;
        FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    });

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
                                                 int          batch_size,                                              \
                                                 int          num_speculative_tokens,                                  \
                                                 int          target_vocab_size,                                       \
                                                 cudaStream_t stream);

INSTANTIATE_REJECTION_SAMPLING(float, int);
}  // namespace rtp_llm