#include <sampling.cuh>
#include <ATen/Utils.h> 
#include <ATen/core/Generator.h>
#include <ATen/hip/HIPGeneratorImpl.h>

#include <ATen/hip/detail/UnpackRaw.cuh>
#include <mutex>

#include "hip/hip_runtime.h"

#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>
#include <hiprand/hiprand_kernel.h>
#include <hip/amd_detail/amd_warp_sync_functions.h>

#include <hipcub/block/block_adjacent_difference.hpp>
#include <hipcub/hipcub.hpp>
#include <limits>
#include <numeric>
#include <tuple>

#include "utils.cuh"
#include "vec_dtypes.cuh"

using namespace hipcub;

#define DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, ...) \
  if (deterministic) {                                            \
    constexpr bool DETERMINISTIC = true;                          \
    __VA_ARGS__                                                   \
  } else {                                                        \
    constexpr bool DETERMINISTIC = false;                         \
    __VA_ARGS__                                                   \
  }

#define DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, ...) \
  if (compute_capacity.first >= 8) {                                           \
    constexpr uint32_t BLOCK_THREADS = 1024;                                   \
    __VA_ARGS__                                                                \
  } else {                                                                     \
    constexpr uint32_t BLOCK_THREADS = 512;                                    \
    __VA_ARGS__                                                                \
  }

#define DISPATCH_SOFTMAX_CACHE_INPUT(cache_input, CACHE_INPUT, ...) \
  if (cache_input) {                                                \
    constexpr bool CACHE_INPUT = true;                              \
    __VA_ARGS__                                                     \
  } else {                                                          \
    constexpr bool CACHE_INPUT = false;                             \
    __VA_ARGS__                                                     \
  }


constexpr BlockScanAlgorithm SCAN_ALGO = BLOCK_SCAN_WARP_SCANS;
constexpr BlockReduceAlgorithm REDUCE_ALGO = BLOCK_REDUCE_WARP_REDUCTIONS;

template <typename T>
struct ValueCount {
  T value;
  int count;

  __device__ ValueCount operator+(const ValueCount& other) const {
    return {value + other.value, count + other.count};
  }
  __device__ ValueCount& operator+=(const ValueCount& other) {
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

struct Float2SoftmaxReduceOp {
  __device__ __forceinline__ float2 operator()(const float2& a, const float2& b) const {
    if (isinf(a.x)) return b;
    if (isinf(b.x)) return a;

    float new_max = max(a.x, b.x);
    float new_denom = a.y * __expf(a.x - new_max) + b.y * __expf(b.x - new_max);
    return make_float2(new_max, new_denom);
  }
};


template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM>
struct SamplingTempStorage {
  union {
    float deterministic_scan[BLOCK_THREADS / 64];
    typename BlockScan<float, BLOCK_THREADS, SCAN_ALGORITHM>::TempStorage scan;
    typename BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce;
    typename BlockReduce<int, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce_int;
    typename BlockReduce<ValueCount<float>, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage
        reduce_value_count;
    typename BlockAdjacentDifference<bool, BLOCK_THREADS>::TempStorage adj_diff;
  } block_prim;
  struct {
    int32_t sampled_id;
    int32_t last_valid_id;
    float max_val;
    union {
      float value;
      ValueCount<float> pair;
    } block_aggregate;
  };
};


#define DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, ...) \
  if (compute_capacity.first >= 8) {                                           \
    constexpr uint32_t BLOCK_THREADS = 1024;                                   \
    __VA_ARGS__                                                                \
  } else {                                                                     \
    constexpr uint32_t BLOCK_THREADS = 512;                                    \
    __VA_ARGS__                                                                \
  }

  template <uint32_t VEC_SIZE, uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM>
__device__ __forceinline__ void DeterministicInclusiveSum(
    const float* in_data, float* out_data,
    SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>* temp_storage) {
  float* smem_prefix_sum = temp_storage->block_prim.deterministic_scan;
  float thread_data[VEC_SIZE];
  float thread_sum = 0;
#pragma unroll
  for (uint32_t i = 0; i < VEC_SIZE; ++i) {
    thread_sum += in_data[i];
    thread_data[i] = thread_sum;
  }

  float thread_exclusive_prefix_sum = thread_sum;

#pragma unroll
  for (uint32_t offset = 1; offset < 64; offset *= 2) {
    float tmp = __shfl_up_sync(0xffffffffffffffff, thread_exclusive_prefix_sum, offset);
    if ((threadIdx.x + 1) % (offset * 2) == 0) {
      thread_exclusive_prefix_sum += tmp;
    }
  }

  float warp_sum = __shfl_sync(0xffffffffffffffff, thread_exclusive_prefix_sum, 0xffffffff);
  if (threadIdx.x % 64 == 63) {
    thread_exclusive_prefix_sum = 0;
  }

#pragma unroll
  for (uint32_t offset = 32; offset >= 1; offset /= 2) {
    float tmp = __shfl_xor_sync(0xffffffffffffffff, thread_exclusive_prefix_sum, offset);
    if ((threadIdx.x + 1) % (offset * 2) == 0) {
      thread_exclusive_prefix_sum = tmp + thread_exclusive_prefix_sum;
    }
    if ((threadIdx.x + 1) % (offset * 2) == offset) {
      thread_exclusive_prefix_sum = tmp;
    }
  }

  smem_prefix_sum[threadIdx.x / 64] = warp_sum; 
  __syncthreads();

  if (threadIdx.x < 64) {
    float warp_exclusive_prefix_sum =
        (threadIdx.x < BLOCK_THREADS / 64) ? smem_prefix_sum[threadIdx.x] : 0;

#pragma unroll
    for (uint32_t offset = 1; offset < 64; offset *= 2) {
      float tmp = __shfl_up_sync(0xffffffffffffffff, warp_exclusive_prefix_sum, offset);
      if ((threadIdx.x + 1) % (offset * 2) == 0) {
        warp_exclusive_prefix_sum += tmp;
      }
    }

    if (threadIdx.x % 64 == 63) {
      warp_exclusive_prefix_sum = 0;
    }

#pragma unroll
    for (uint32_t offset = 32; offset >= 1; offset /= 2) {
      float tmp = __shfl_xor_sync(0xffffffffffffffff, warp_exclusive_prefix_sum, offset);
      if ((threadIdx.x + 1) % (offset * 2) == 0) {
        warp_exclusive_prefix_sum = tmp + warp_exclusive_prefix_sum;
      }
      if ((threadIdx.x + 1) % (offset * 2) == offset) {
        warp_exclusive_prefix_sum = tmp;
      }
    }
    if (threadIdx.x < BLOCK_THREADS / 64) {
      smem_prefix_sum[threadIdx.x] = warp_exclusive_prefix_sum;
    }
  }
  __syncthreads();

#pragma unroll
  for (uint32_t i = 0; i < VEC_SIZE; ++i) {
    out_data[i] = smem_prefix_sum[threadIdx.x / 64] + thread_exclusive_prefix_sum + thread_data[i];
  }
}

template <uint32_t VEC_SIZE, uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, bool DETERMINISTIC, typename Predicate>
__device__ __forceinline__ void DeviceSamplingFromProb(
    uint32_t i, uint32_t d, Predicate pred, float u, vec_t<float, VEC_SIZE> prob_vec,
    float& aggregate,
    SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>* temp_storage) {
  const uint32_t tx = threadIdx.x;
  float prob_greater_than_threshold[VEC_SIZE];
  float inclusive_cdf[VEC_SIZE];
  bool greater_than_u[VEC_SIZE], valid[VEC_SIZE];
#pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; ++j) {
    prob_greater_than_threshold[j] = pred(prob_vec[j]) ? prob_vec[j] : 0;
    valid[j] = pred(prob_vec[j]) && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d;
  }
  float aggregate_local =
      BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage->block_prim.reduce)
          .template Sum<VEC_SIZE>(prob_greater_than_threshold);
  if (tx == 0) {
    temp_storage->block_aggregate.value = aggregate_local;
  }
  __syncthreads();
  aggregate_local = temp_storage->block_aggregate.value;

  if (aggregate + aggregate_local > u) {
    if constexpr (DETERMINISTIC) {
      DeterministicInclusiveSum<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>(
          prob_greater_than_threshold, inclusive_cdf, temp_storage);
    } else {
      BlockScan<float, BLOCK_THREADS, SCAN_ALGORITHM>(temp_storage->block_prim.scan)
          .template InclusiveSum<VEC_SIZE>(prob_greater_than_threshold, inclusive_cdf);

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
        .template FlagHeads<VEC_SIZE>(greater_than_u_diff, greater_than_u, BoolDiffOp(), 0);
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

  // update the last valid index
  int valid_index[VEC_SIZE];
#pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; ++j) {
    if (valid[j]) {
      valid_index[j] = (i * BLOCK_THREADS + tx) * VEC_SIZE + j;
    } else {
      valid_index[j] = -1;
    }
  }
  int max_valid_index =
      BlockReduce<int, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage->block_prim.reduce_int)
          .Reduce(valid_index, hipcub::Max());
  if (tx == 0 && max_valid_index != -1) {
    temp_storage->last_valid_id = max_valid_index;
  }
  __syncthreads();
  aggregate += aggregate_local;
}

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType, typename IdType>
__global__ void ChainSpeculativeSampling(DType* draft_probs, IdType* draft_token_ids,
                                         DType* uniform_samples, DType* target_probs,
                                         IdType* output_token_ids,
                                         IdType* output_accepted_token_num,
                                         IdType* output_emitted_draft_token_num,
                                         uint32_t num_speculative_tokens, uint32_t d) {
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  const uint32_t row_idx = bx;

  extern __shared__ __align__(
      alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
      uint8_t smem_sampling[];
  auto& temp_storage = 
      reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(
          smem_sampling);

  uint32_t pos = num_speculative_tokens;
  for (uint32_t i = 0; i < num_speculative_tokens; ++i) {
    IdType draft_id = draft_token_ids[row_idx * num_speculative_tokens + i];
    float q = target_probs[(row_idx * (num_speculative_tokens + 1) + i) * d + draft_id],
          p = draft_probs[(row_idx * num_speculative_tokens + i) * d + draft_id];
    DType u = uniform_samples[row_idx * (num_speculative_tokens + 1) + i];
    if (u * p < q) {
      // accept the draft models output
      output_token_ids[row_idx * (num_speculative_tokens + 1) + i] = draft_id;
    } else {
      pos = i;
      break;
    }
  }

  if (tx == 0) {
    output_emitted_draft_token_num[row_idx] += pos + 1;
  }

  if (pos == num_speculative_tokens) {
    return;
  }

  // sample from relu(target_probs - draft_probs)
  float sum_relu_q_minus_p = 0;
  vec_t<float, VEC_SIZE> q_vec, p_vec;
  float relu_q_minus_p[VEC_SIZE];
#pragma unroll 2
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    q_vec.fill(0);
    p_vec.fill(0);
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      q_vec.cast_load(target_probs + (row_idx * (num_speculative_tokens + 1) + pos) * d +
                      i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      if (pos != num_speculative_tokens) {
        // there is no draft_probs for the bonus token
        p_vec.cast_load(draft_probs + (row_idx * num_speculative_tokens + pos) * d +
                        i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      }
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      relu_q_minus_p[j] = max(q_vec[j] - p_vec[j], 0);;
    }
    sum_relu_q_minus_p +=
        BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
            .template Sum<VEC_SIZE>(relu_q_minus_p);
    __syncthreads();
  }
  if (tx == 0) {
    temp_storage.block_aggregate.value = sum_relu_q_minus_p;
  }
  // init the first rejected token to d
  temp_storage.sampled_id = d - 1;
  __syncthreads();
  sum_relu_q_minus_p = temp_storage.block_aggregate.value;
  DType u = uniform_samples[row_idx * (num_speculative_tokens + 1) +
                            min(pos + 1, num_speculative_tokens)] * sum_relu_q_minus_p;

  float aggregate_relu_q_minus_p(0);
#pragma unroll 2
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    q_vec.fill(0);
    p_vec.fill(0);
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      q_vec.cast_load(target_probs + (row_idx * (num_speculative_tokens + 1) + pos) * d +
                      i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      if (pos != num_speculative_tokens) {
        // there is no draft_probs for the bonus token
        p_vec.cast_load(draft_probs + (row_idx * num_speculative_tokens + pos) * d +
                        i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      }
    }

    vec_t<float, VEC_SIZE> relu_q_minus_p_vec;
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      relu_q_minus_p_vec[j] = max(q_vec[j] - p_vec[j], 0);
    }

    DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM,
                           DETERMINISTIC>(
        i, d, [&](float x) { return x > 0; }, u, relu_q_minus_p_vec, aggregate_relu_q_minus_p,
        &temp_storage);
    if (aggregate_relu_q_minus_p > u) {
      break;
    }
  }
  __syncthreads();
  int sampled_id = temp_storage.sampled_id;
  // if (sampled_id == d) {
  //   // NOTE(Zihao): this would happen when u is very close to 1
  //   // and the sum of probabilities is smaller than u
  //   // In this case, we use the last valid index as the sampled id
  //   sampled_id = temp_storage.last_valid_id;
  // }
  // set the first rejected token
  output_token_ids[row_idx * (num_speculative_tokens + 1) + pos] = sampled_id;
  // move to the next token
  pos++;

  // pad remaining tokens with -1
  for (; pos < num_speculative_tokens + 1; ++pos) {
    output_token_ids[row_idx * (num_speculative_tokens + 1) + pos] = -1;
  }
}

template <typename DType, typename IdType>
hipError_t ChainSpeculativeSampling(DType* draft_probs, IdType* draft_token_ids,
                                     DType* uniform_samples, DType* target_probs,
                                     IdType* output_token_ids,
                                     IdType* output_accepted_token_num,
                                     IdType* output_emitted_draft_token_num, uint32_t batch_size,
                                     uint32_t num_speculative_tokens, uint32_t d,
                                     bool deterministic, hipStream_t stream = 0) {
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(DType), d);

  const uint32_t smem_size = sizeof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);
  dim3 nblks(batch_size);
  dim3 nthrs(BLOCK_THREADS);
  DISPATCH_ALIGNED_VEC_SIZE(
      vec_size, VEC_SIZE, {DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
        ChainSpeculativeSampling<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, VEC_SIZE,
            DETERMINISTIC, DType, IdType><<<nblks, nthrs, smem_size, stream>>>(
          draft_probs,
          draft_token_ids,
          uniform_samples,
          target_probs,
          output_token_ids,
          output_accepted_token_num,
          output_emitted_draft_token_num,
          num_speculative_tokens,
          d);
      })});
  return hipSuccess;
}

void chain_speculative_sampling(at::Tensor draft_probs, at::Tensor draft_token_ids,
                                at::Tensor uniform_samples, at::Tensor target_probs,
                                at::Tensor output_token_ids,
                                at::Tensor output_accepted_token_num,
                                at::Tensor output_emitted_draft_token_num, bool deterministic,
                                int64_t hip_stream) {
  auto device = draft_probs.device();
  unsigned int batch_size = draft_probs.size(0);
  unsigned int num_speculate_tokens = draft_probs.size(1);
  unsigned int vocab_size = draft_probs.size(2);

  hipStream_t stream = reinterpret_cast<hipStream_t>(hip_stream);
  hipError_t status = ChainSpeculativeSampling<float, int>(
      static_cast<float*>(draft_probs.data_ptr()), static_cast<int*>(draft_token_ids.data_ptr()),
      static_cast<float*>(uniform_samples.data_ptr()), static_cast<float*>(target_probs.data_ptr()),
      static_cast<int*>(output_token_ids.data_ptr()),
      static_cast<int*>(output_accepted_token_num.data_ptr()),
      static_cast<int*>(output_emitted_draft_token_num.data_ptr()), batch_size, 
      num_speculate_tokens, vocab_size, deterministic, stream);

  TORCH_CHECK(status == hipSuccess, "ChainSpeculativeSampling failed with error code " +
                                         std::string(hipGetErrorString(status)));
}
