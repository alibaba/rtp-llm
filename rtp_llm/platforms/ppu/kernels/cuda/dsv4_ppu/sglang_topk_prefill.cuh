#pragma once
// SGLang selection algorithm unchanged; see NOTICE. RTP uses its DeepGEMM JIT ABI.
#define SGL_DEVICE __device__ __forceinline__
#ifndef USE_ROCM
#include <cub/cub.cuh>
#else
#include <hipcub/hipcub.hpp>
#endif

#include <cuda_fp16.h>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace {

#ifndef SGL_TOPK
#define SGL_TOPK 512
#endif

constexpr uint32_t kTopK = SGL_TOPK;
constexpr uint32_t kTopKBlockSize = 512;
constexpr uint32_t kNumBins = 2048;
constexpr uint32_t kNumFinalItems = 2048;
constexpr uint32_t kWarpSize = 32;
static_assert(kTopK <= kNumFinalItems, "top_k_per_row_prefill supports topK <= 2048");

struct TopKPrefillParams {
  const float* __restrict__ scores;
  const int32_t* __restrict__ row_starts;
  const int32_t* __restrict__ row_ends;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  int32_t* __restrict__ raw_indices;
  const int64_t score_stride;
  const int64_t page_table_stride;
  uint32_t page_bits;
};

SGL_DEVICE uint32_t convert_to_uint32(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? bits : (~bits & 0x7fffffffu);
}

template <int step>
SGL_DEVICE uint32_t extract_bin_idx(float x) {
  if constexpr (step == 0) {
    __half hx = __float2half(x);
    uint16_t bits = __half_as_ushort(hx);
    bits = (bits & 0x8000u) ? bits : (~bits & 0x7fffu);
    return bits >> 5;
  } else {
    const uint32_t bits = convert_to_uint32(x);
    if constexpr (step == 1) {
      return bits >> 21;
    } else if constexpr (step == 2) {
      return (bits >> 10) & 0x7ffu;
    } else {
      return bits & 0x3ffu;
    }
  }
}

template <int shift>
SGL_DEVICE bool is_partial_match(float x, uint32_t pattern) {
  if constexpr (shift == 0) {
    return true;
  }
  return ((convert_to_uint32(x) ^ pattern) >> shift) == 0;
}

SGL_DEVICE int32_t page_to_indices(const int32_t* __restrict__ page_table, uint32_t i, uint32_t page_bits) {
  const uint32_t mask = (1u << page_bits) - 1u;
  return (page_table[i >> page_bits] << page_bits) | (i & mask);
}

SGL_DEVICE void transform_indices(
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ selected,
    int32_t* __restrict__ page_indices,
    int32_t* __restrict__ raw_indices,
    const uint32_t valid_count,
    const uint32_t page_bits) {
  for (uint32_t i = threadIdx.x; i < kTopK; i += kTopKBlockSize) {
    if (i < valid_count) {
      const auto raw = static_cast<uint32_t>(selected[i]);
      page_indices[i] = page_to_indices(page_table, raw, page_bits);
      if (raw_indices != nullptr) {
        raw_indices[i] = static_cast<int32_t>(raw);
      }
    } else {
      page_indices[i] = -1;
      if (raw_indices != nullptr) {
        raw_indices[i] = -1;
      }
    }
  }
}

SGL_DEVICE void naive_transform(
    const int32_t* __restrict__ page_table,
    int32_t* __restrict__ page_indices,
    int32_t* __restrict__ raw_indices,
    const uint32_t length,
    const uint32_t page_bits) {
  for (uint32_t i = threadIdx.x; i < kTopK; i += kTopKBlockSize) {
    if (i < length) {
      page_indices[i] = page_to_indices(page_table, i, page_bits);
      if (raw_indices != nullptr) {
        raw_indices[i] = i;
      }
    } else {
      page_indices[i] = -1;
      if (raw_indices != nullptr) {
        raw_indices[i] = -1;
      }
    }
  }
}

template <typename T, typename IdxT, typename Func>
SGL_DEVICE void vectorized_process(size_t thread_rank, size_t num_threads, const T* input, IdxT length, Func func) {
  using WideT = float4;
  if constexpr (sizeof(T) >= sizeof(WideT)) {
    for (IdxT i = thread_rank; i < length; i += num_threads) {
      func(input[i], i);
    }
  } else {
    static_assert(sizeof(WideT) % sizeof(T) == 0);
    constexpr int items_per_scalar = sizeof(WideT) / sizeof(T);
    union {
      WideT scalar;
      T array[items_per_scalar];
    } wide;

    int skip_count = (reinterpret_cast<size_t>(input) % sizeof(WideT))
                         ? ((sizeof(WideT) - reinterpret_cast<size_t>(input) % sizeof(WideT)) / sizeof(T))
                         : 0;
    if (skip_count > length) {
      skip_count = length;
    }

    const WideT* input_cast = reinterpret_cast<const WideT*>(input + skip_count);
    const IdxT length_cast = (length - skip_count) / items_per_scalar;

    for (IdxT i = thread_rank; i < length_cast; i += num_threads) {
      wide.scalar = input_cast[i];
      const IdxT real_i = skip_count + i * items_per_scalar;
#pragma unroll
      for (int j = 0; j < items_per_scalar; ++j) {
        func(wide.array[j], real_i + j);
      }
    }

    static_assert(kWarpSize >= items_per_scalar);
    if (thread_rank < skip_count) {
      func(input[thread_rank], thread_rank);
    }

    const IdxT remain_i = skip_count + length_cast * items_per_scalar + thread_rank;
    if (remain_i < length) {
      func(input[remain_i], remain_i);
    }
  }
}

template <int step, typename SmemFinalType>
SGL_DEVICE bool process_histogram_step(
    const float* __restrict__ scores,
    const int row_len,
    uint32_t& logit_pattern,
    int& threshold_bin_idx,
    int32_t* __restrict__ smem_output,
    int* __restrict__ smem_threshold_bin_idx,
    int* __restrict__ smem_final_dst_idx,
    int* __restrict__ smem_final_bin_size,
    int* __restrict__ smem_found_topk_values,
    SmemFinalType& smem_final) {
#pragma unroll
  for (int idx = threadIdx.x; idx < static_cast<int>(kNumBins); idx += kTopKBlockSize) {
    smem_final.histo.data[idx] = 0;
  }
  __syncthreads();

  constexpr auto pattern_shift = step < 2 ? 0 : step == 2 ? 21 : 10;
  if constexpr (step == 2) {
    logit_pattern = static_cast<uint32_t>(threshold_bin_idx & 0x7ff) << pattern_shift;
  } else if constexpr (step == 3) {
    logit_pattern |= static_cast<uint32_t>(threshold_bin_idx & 0x7ff) << pattern_shift;
  }

  auto distribute_to_bins = [&](float score, int /* idx */) {
    if (is_partial_match<pattern_shift>(score, logit_pattern)) {
      const uint32_t bin_idx = extract_bin_idx<step>(score);
      atomicAdd(&smem_final.histo.data[bin_idx], 1);
    }
  };

  vectorized_process(threadIdx.x, kTopKBlockSize, scores, row_len, distribute_to_bins);
  __syncthreads();

  int last_value = smem_found_topk_values[0];
  for (int round = 0; round < static_cast<int>(kNumBins / kTopKBlockSize); ++round) {
    const int idx = threadIdx.x + kTopKBlockSize * round;
    const int bin_count = smem_final.histo.data[idx];
    __syncthreads();

    int prefix_sum = 0;
    int total_sum = 0;
    using Scan = cub::BlockScan<int, kTopKBlockSize>;
    Scan(smem_final.histo.scan).ExclusiveSum(bin_count, prefix_sum, total_sum);

    prefix_sum += last_value;
    total_sum += last_value;
    smem_final.histo.data[idx] = prefix_sum;
    __syncthreads();

    bool found_threshold = false;
    if (prefix_sum < static_cast<int>(kTopK)) {
      const int next_prefix_sum =
          threadIdx.x == static_cast<int>(kTopKBlockSize) - 1 ? total_sum : smem_final.histo.data[idx + 1];
      if (next_prefix_sum >= static_cast<int>(kTopK)) {
        smem_threshold_bin_idx[0] = idx;
        smem_final_bin_size[0] = next_prefix_sum - prefix_sum;
        found_threshold = true;
      }
    }

    if (__syncthreads_or(found_threshold)) {
      break;
    }
    last_value = total_sum;
  }
  __syncthreads();

  threshold_bin_idx = smem_threshold_bin_idx[0];

  auto process_bins = [&](float score, int idx) {
    if (is_partial_match<pattern_shift>(score, logit_pattern)) {
      const uint32_t bin_idx = extract_bin_idx<step>(score);
      const bool should_write_directly =
          (step == 0 && smem_final_bin_size[0] <= static_cast<int>(kNumFinalItems)) || (step >= 1);
      if (bin_idx < static_cast<uint32_t>(threshold_bin_idx) && should_write_directly) {
        const int dst_idx = atomicAdd(&smem_found_topk_values[0], 1);
        smem_output[dst_idx] = idx;
      }
      if constexpr (step < 3) {
        if (bin_idx == static_cast<uint32_t>(threshold_bin_idx) &&
            smem_final_bin_size[0] <= static_cast<int>(kNumFinalItems)) {
          const int dst_idx = atomicAdd(&smem_final_dst_idx[0], 1);
          smem_final.items.logits[dst_idx] = score;
          smem_final.items.indices[dst_idx] = idx;
        }
      } else {
        if (bin_idx == static_cast<uint32_t>(threshold_bin_idx)) {
          const int dst_idx = atomicAdd(&smem_final.histo.data[bin_idx], 1);
          if (dst_idx < static_cast<int>(kTopK)) {
            smem_output[dst_idx] = idx;
          }
        }
      }
    }
  };

  vectorized_process(threadIdx.x, kTopKBlockSize, scores, row_len, process_bins);
  __syncthreads();
  return smem_final_bin_size[0] > static_cast<int>(kNumFinalItems);
}

template <bool use_radix_sort>
SGL_DEVICE void topk_per_row_vllm(const float* __restrict__ scores, int32_t* __restrict__ output, const int row_len) {
  constexpr int kNumFinalItemsPerThread = kNumFinalItems / kTopKBlockSize;
  using FinalSort = cub::BlockRadixSort<float, kTopKBlockSize, kNumFinalItemsPerThread, int>;
  using FinalSortTempStorage = std::conditional_t<use_radix_sort, typename FinalSort::TempStorage, int>;
  using Scan = cub::BlockScan<int, kTopKBlockSize>;

  struct FinalItems {
    int indices[kNumFinalItems];
    float logits[kNumFinalItems];
  };

  struct Histogram {
    typename Scan::TempStorage scan;
    int data[kNumBins];
  };

  __shared__ union {
    FinalItems items;
    FinalSortTempStorage final_sort;
    Histogram histo;
  } smem_final;

  extern __shared__ int32_t smem_output[];
  __shared__ int smem_threshold_bin_idx[1];
  __shared__ int smem_final_dst_idx[1];
  __shared__ int smem_final_bin_size[1];
  __shared__ int smem_found_topk_values[1];

  if (threadIdx.x == 0) {
    smem_final_dst_idx[0] = 0;
    smem_found_topk_values[0] = 0;
  }
  __syncthreads();

  int threshold_bin_idx = -1;
  uint32_t logit_pattern = 0;

  bool continue_to_next_step = process_histogram_step<0>(
      scores,
      row_len,
      logit_pattern,
      threshold_bin_idx,
      smem_output,
      smem_threshold_bin_idx,
      smem_final_dst_idx,
      smem_final_bin_size,
      smem_found_topk_values,
      smem_final);

  if (continue_to_next_step) {
    continue_to_next_step = process_histogram_step<1>(
        scores,
        row_len,
        logit_pattern,
        threshold_bin_idx,
        smem_output,
        smem_threshold_bin_idx,
        smem_final_dst_idx,
        smem_final_bin_size,
        smem_found_topk_values,
        smem_final);
  }
  if (continue_to_next_step) {
    continue_to_next_step = process_histogram_step<2>(
        scores,
        row_len,
        logit_pattern,
        threshold_bin_idx,
        smem_output,
        smem_threshold_bin_idx,
        smem_final_dst_idx,
        smem_final_bin_size,
        smem_found_topk_values,
        smem_final);
  }
  if (continue_to_next_step) {
    process_histogram_step<3>(
        scores,
        row_len,
        logit_pattern,
        threshold_bin_idx,
        smem_output,
        smem_threshold_bin_idx,
        smem_final_dst_idx,
        smem_final_bin_size,
        smem_found_topk_values,
        smem_final);
  }

  if (!continue_to_next_step) {
    if constexpr (use_radix_sort) {
      float final_logits[kNumFinalItemsPerThread];
      int final_indices[kNumFinalItemsPerThread];

#pragma unroll
      for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
        final_logits[ii] = -FLT_MAX;
        final_indices[ii] = -1;
      }

#pragma unroll
      for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
        const int src_idx = ii * kTopKBlockSize + threadIdx.x;
        if (src_idx < smem_final_dst_idx[0]) {
          final_logits[ii] = smem_final.items.logits[src_idx];
          final_indices[ii] = smem_final.items.indices[src_idx];
        }
      }
      __syncthreads();

      FinalSort(smem_final.final_sort).SortDescendingBlockedToStriped(final_logits, final_indices);
      const int base_idx = smem_found_topk_values[0];

#pragma unroll
      for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
        const int src_idx = ii * kTopKBlockSize + threadIdx.x;
        const int dst_idx = base_idx + src_idx;
        if (dst_idx < static_cast<int>(kTopK)) {
          smem_output[dst_idx] = final_indices[ii];
        }
      }
    } else {
      const int base_idx = smem_found_topk_values[0];
      for (int i = threadIdx.x; i < smem_final_dst_idx[0]; i += kTopKBlockSize) {
        int out_index = 0;
        const float score = smem_final.items.logits[i];
        for (int j = 0; j < smem_final_dst_idx[0]; ++j) {
          const float other_score = smem_final.items.logits[j];
          if (score < other_score || (score == other_score && i < j)) {
            ++out_index;
          }
        }
        if (out_index + base_idx < static_cast<int>(kTopK)) {
          smem_output[out_index + base_idx] = smem_final.items.indices[i];
        }
      }
    }
    __syncthreads();
  }

  for (uint32_t i = threadIdx.x; i < kTopK; i += kTopKBlockSize) {
    output[i] = smem_output[i];
  }
}

__global__ __launch_bounds__(kTopKBlockSize, 1) void topk_prefill_transform_kernel(
    const __grid_constant__ TopKPrefillParams params) {
  const auto row_id = blockIdx.x;
  const auto row_start = params.row_starts[row_id];
  const auto row_end = params.row_ends[row_id];
  const auto length = row_end > row_start ? static_cast<uint32_t>(row_end - row_start) : 0u;
  const auto scores = params.scores + static_cast<int64_t>(row_id) * params.score_stride + row_start;
  const auto page_table = params.page_table + static_cast<int64_t>(row_id) * params.page_table_stride;
  auto page_indices = params.page_indices + static_cast<int64_t>(row_id) * kTopK;
  auto raw_indices =
      params.raw_indices == nullptr ? nullptr : params.raw_indices + static_cast<int64_t>(row_id) * kTopK;

  if (length <= kTopK) {
    naive_transform(page_table, page_indices, raw_indices, length, params.page_bits);
    return;
  }

  __shared__ int32_t s_topk_indices[kTopK];
  if (row_id < 12288) {
    topk_per_row_vllm<false>(scores, s_topk_indices, static_cast<int>(length));
  } else {
    topk_per_row_vllm<true>(scores, s_topk_indices, static_cast<int>(length));
  }
  transform_indices(page_table, s_topk_indices, page_indices, raw_indices, kTopK, params.page_bits);
}


} // namespace
#undef SGL_DEVICE
