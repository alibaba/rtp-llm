#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/dsa/legacy_radix_topk.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cfloat>
#include <cstdint>

namespace sglang {

// `topk` is a *runtime* value (<= kMaxTopK), so one module serves every k. It
// used to be baked in via -DSGL_TOPK, which built a separate module per k --
// and because `kTopK` came from a macro rather than a template parameter, both
// modules exported identically mangled symbols. The function-local static that
// guarded the smem opt-in is emitted as STB_GNU_UNIQUE, which the loader
// merges across every loaded object, so whichever module was used second
// skipped its cudaFuncSetAttribute opt-in and then failed to launch with 64 KB
// of dynamic shared memory ("invalid argument").
constexpr uint32_t kMaxTopK = 2048;
// Fixed, and deliberately not tied to `topk`: run_cumsum() and the histogram
// init below index up to RADIX + 1 == 257 threads, so a block sized after a
// small topk would silently skip part of the histogram.
constexpr uint32_t kTopKBlockSize = 1024;
constexpr uint32_t kSMEM = 16 * 1024 * sizeof(uint32_t);  // 64KB (bytes)

struct TopKParams {
  const float* __restrict__ scores;
  const int32_t* __restrict__ seq_lens;
  const int32_t* __restrict__ page_table;
  int32_t* __restrict__ page_indices;
  int32_t* __restrict__ raw_indices;  // optional: output raw abs position indices before page transform
  const int64_t score_stride;
  const int64_t page_table_stride;
  uint32_t page_bits;
  uint32_t topk;
};

SGL_DEVICE int32_t page_to_indices(const int32_t* __restrict__ page_table, uint32_t i, uint32_t page_bits) {
  if (page_table == nullptr) return static_cast<int32_t>(i);
  const uint32_t mask = (1u << page_bits) - 1u;
  return (page_table[i >> page_bits] << page_bits) | (i & mask);
}

// ============================================================
// v3: RegisterTopK-based implementation (1024 threads, 12-bit histogram)
// ============================================================

namespace v3 {
constexpr uint32_t kBlockSize = kTopKBlockSize;
constexpr uint32_t kNumWarps = kBlockSize / 32;
constexpr uint32_t kHistBits = 12;
constexpr uint32_t kHistBins = 1 << kHistBits;  // 4096
constexpr uint32_t kVecsPerThread = 4;
constexpr uint32_t kMax1PassLength = kVecsPerThread * 4 * kBlockSize;  // 16384
constexpr uint32_t kMax2PassLength = 2 * kMax1PassLength;              // 32768
// 2048 ties occupy 16KB, the same size as the histogram in their shared union.
constexpr uint32_t kMaxTies = kMaxTopK;
}  // namespace v3

using Vec4 = device::AlignedVector<float, 4>;

struct alignas(16) MatchBinV3 {
  uint32_t bin;
  uint32_t above_count;
  uint32_t equal_count;
};

struct alignas(8) TieV3 {
  uint32_t idx;
  float score;
};

template <uint32_t kBits>
SGL_DEVICE uint32_t extract_coarse_bin_v3(float x) {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return key >> (16 - kBits);
}

SGL_DEVICE uint32_t warp_inclusive_sum_v3(uint32_t lane_id, uint32_t val) {
#pragma unroll
  for (uint32_t offset = 1; offset < 32; offset *= 2) {
    uint32_t n = __shfl_up_sync(0xFFFFFFFF, val, offset);
    if (lane_id >= offset) val += n;
  }
  return val;
}

SGL_DEVICE uint32_t extract_exact_bin_v3(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// cp.async PTX helpers
SGL_DEVICE void cp_async_cg_shared_global_16(void* smem_dst, const void* global_src) {
  uint32_t dst;
  asm volatile(
      "{ .reg .u64 u64addr;\n\t"
      "  cvta.to.shared.u64 u64addr, %1;\n\t"
      "  cvt.u32.u64 %0, u64addr; }\n"
      : "=r"(dst)
      : "l"(smem_dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(global_src));
}

SGL_DEVICE void cp_async_commit() {
  asm volatile("cp.async.commit_group;");
}
SGL_DEVICE void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;");
}
SGL_DEVICE void
radix_topk(const float* __restrict__ input, int32_t* __restrict__ output, uint32_t length, uint32_t topk);

struct RegisterTopKSmem {
  using HistVec = device::AlignedVector<uint32_t, v3::kHistBins / v3::kBlockSize>;
  alignas(128) uint32_t counter_gt;
  alignas(128) uint32_t counter_eq;
  MatchBinV3 match;
  uint32_t warp_sum[v3::kNumWarps];
  union {
    uint32_t histogram[v3::kHistBins];
    HistVec histogram_vec[v3::kBlockSize];
    TieV3 tie_buffer[v3::kMaxTies];
  };  // 16KB
  alignas(16) float score_buffer[v3::kMax1PassLength];  // 64KB
};

template <bool kIs2Pass>
[[maybe_unused]]
SGL_DEVICE void register_topk(
    const float* __restrict__ scores,
    int32_t* __restrict__ indices,
    const uint32_t length,
    const uint32_t topk,
    void* _smem) {
  auto* smem = static_cast<RegisterTopKSmem*>(_smem);
  const auto tx = threadIdx.x;
  const auto lane_id = tx % 32;
  const auto warp_id = tx / 32;

  // Initialize shared memory histogram
  {
    typename RegisterTopKSmem::HistVec hist_vec;
    hist_vec.fill(0);
    smem->histogram_vec[tx] = hist_vec;
    if (tx == 0) {
      smem->counter_gt = smem->counter_eq = 0;
    }
    __syncthreads();
  }

  // Load scores into registers
  Vec4 local[v3::kVecsPerThread];
#pragma unroll
  for (uint32_t v = 0; v < v3::kVecsPerThread; ++v) {
    const uint32_t base = (tx + v * v3::kBlockSize) * 4;
    if (base >= length) break;
    local[v].load(scores, tx + v * v3::kBlockSize);
  }

  // Fetch the next chunk of scores via cp.async (replaces TMA)
  if constexpr (kIs2Pass) {
    const uint32_t extra_length = length - v3::kMax1PassLength;
    const uint32_t extra_vec4s = (extra_length + 3) / 4;
    for (uint32_t i = tx; i < extra_vec4s; i += v3::kBlockSize) {
      cp_async_cg_shared_global_16(&smem->score_buffer[i * 4], scores + v3::kMax1PassLength + i * 4);
    }
    cp_async_commit();
  }

  // Accumulate histogram via shared-memory atomics
#pragma unroll
  for (uint32_t v = 0; v < v3::kVecsPerThread; ++v) {
#pragma unroll
    for (uint32_t e = 0; e < 4; ++e) {
      if constexpr (!kIs2Pass) {
        const uint32_t idx = (tx + v * v3::kBlockSize) * 4 + e;
        if (idx >= length) goto LABEL_ACC_FINISH;
      }
      atomicAdd(&smem->histogram[extract_coarse_bin_v3<v3::kHistBits>(local[v][e])], 1);
    }
  }
  if constexpr (kIs2Pass) {
    // Wait for cp.async to complete
    cp_async_wait_all();
    __syncthreads();
    for (uint32_t i = tx; i + v3::kMax1PassLength < length; i += v3::kBlockSize) {
      const auto val = smem->score_buffer[i];  // Stall instruction stall
      atomicAdd(&smem->histogram[extract_coarse_bin_v3<v3::kHistBits>(val)], 1);
    }
  }
[[maybe_unused]] LABEL_ACC_FINISH:
  __syncthreads();

  // Phase 2: Exclusive prefix scan -> find threshold bin
  {
    constexpr uint32_t kItems = v3::kHistBins / v3::kBlockSize;
    uint32_t orig[kItems];
    const auto hist_vec = smem->histogram_vec[tx];
    uint32_t tmp_local_sum = 0;

#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      orig[i] = hist_vec[i];
      tmp_local_sum += orig[i];
    }

    const auto warp_inc = warp_inclusive_sum_v3(lane_id, tmp_local_sum);
    const auto warp_exc = warp_inc - tmp_local_sum;
    if (lane_id == 31) {
      smem->warp_sum[warp_id] = warp_inc;
    }

    __syncthreads();

    const auto tmp = smem->warp_sum[lane_id];
    uint32_t prefix_sum = device::warp::reduce_sum(lane_id < warp_id ? tmp : 0u);
    prefix_sum += warp_exc;
#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      prefix_sum += orig[i];
      const auto above = length - prefix_sum;
      if (above < topk && above + orig[i] >= topk) {
        smem->match = {
            .bin = tx * kItems + i,
            .above_count = above,
            .equal_count = orig[i],
        };
      }
    }
    __syncthreads();
  }

  const auto [thr_bin, num_above, num_equal] = smem->match;
  if (num_equal > v3::kMaxTies) {
    radix_topk(scores, indices, length, topk);
    return;
  }

  // Phase 3: Scatter
  constexpr uint32_t kMaxTolerance = 0;
  const bool need_tiebreak = (num_equal + num_above > topk + kMaxTolerance);
  const auto topk_indices = indices;
  const auto tie_buffer = smem->tie_buffer;

#pragma unroll
  for (uint32_t v = 0; v < v3::kVecsPerThread; ++v) {
#pragma unroll
    for (uint32_t e = 0; e < 4; ++e) {
      const uint32_t idx = (tx + v * v3::kBlockSize) * 4 + e;
      if constexpr (!kIs2Pass) {
        if (idx >= length) goto LABEL_SCATTER_DONE;
      }
      const uint32_t bin = extract_coarse_bin_v3<v3::kHistBits>(local[v][e]);
      if (bin > thr_bin) {
        topk_indices[atomicAdd(&smem->counter_gt, 1)] = idx;
      } else if (bin == thr_bin) {
        const auto pos = atomicAdd(&smem->counter_eq, 1);
        if (need_tiebreak) {
          if (pos < v3::kMaxTies) {
            tie_buffer[pos] = {.idx = idx, .score = local[v][e]};
          }
        } else {
          if (const auto which = pos + num_above; which < topk) {
            topk_indices[which] = idx;
          }
        }
      }
    }
    // prefetch the next scores from smem into registers (for 2-pass)
    if constexpr (kIs2Pass) {
      local[v].load(smem->score_buffer, tx + v * v3::kBlockSize);
    }
  }

  // 16K ~ 32K: process second chunk from registers (now loaded from score_buffer)
  if constexpr (kIs2Pass) {
#pragma unroll
    for (uint32_t v = 0; v < v3::kVecsPerThread; ++v) {
#pragma unroll
      for (uint32_t e = 0; e < 4; ++e) {
        const uint32_t idx = (tx + v * v3::kBlockSize) * 4 + e + v3::kMax1PassLength;
        if (idx >= length) goto LABEL_SCATTER_DONE;
        const uint32_t bin = extract_coarse_bin_v3<v3::kHistBits>(local[v][e]);
        if (bin > thr_bin) {
          topk_indices[atomicAdd(&smem->counter_gt, 1)] = idx;
        } else if (bin == thr_bin) {
          const auto pos = atomicAdd(&smem->counter_eq, 1);
          if (need_tiebreak) {
            if (pos < v3::kMaxTies) {
              tie_buffer[pos] = {.idx = idx, .score = local[v][e]};
            }
          } else {
            if (const auto which = pos + num_above; which < topk) {
              topk_indices[which] = idx;
            }
          }
        }
      }
    }
  }

[[maybe_unused]] LABEL_SCATTER_DONE:
  if (!need_tiebreak) return;

  // Phase 4: Tie-breaking within the threshold bin
  __syncthreads();

  const uint32_t num_ties = min(num_equal, v3::kMaxTies);
  const uint32_t topk_remain = topk - num_above;

  const auto is_greater = [](const TieV3& a, const TieV3& b) {
    if (a.score > b.score) return true;
    if (a.score < b.score) return false;
    // Equal scores or NaNs use the position as a deterministic tie-breaker.
    return a.idx < b.idx;
  };

  if (num_ties <= 32u) {
    if (lane_id >= num_ties || warp_id >= num_ties) return;
    const uint32_t mask = (1ull << num_ties) - 1u;
    const auto tie = tie_buffer[lane_id];
    const auto target_tie = tie_buffer[warp_id];
    const bool pred = is_greater(tie, target_tie);
    const auto rank = static_cast<uint32_t>(__popc(__ballot_sync(mask, pred)));
    if (lane_id == 0 && rank < topk_remain) {
      topk_indices[num_above + rank] = target_tie.idx;
    }
  } else if (num_ties <= 64u) {
    const auto lane_id_1 = lane_id + 32;
    const auto warp_id_1 = warp_id + 32;
    const auto invalid = TieV3{.idx = 0xFFFFFFFF, .score = -FLT_MAX};
    const auto tie_0 = tie_buffer[lane_id];
    const auto tie_1 = lane_id_1 < num_ties ? tie_buffer[lane_id_1] : invalid;
    if (true) {
      const auto target = tie_buffer[warp_id];
      const bool pred_0 = is_greater(tie_0, target);
      const bool pred_1 = is_greater(tie_1, target);
      const auto rank_0 = static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFF, pred_0)));
      const auto rank_1 = static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFF, pred_1)));
      const auto rank = rank_0 + rank_1;
      if (lane_id == 0 && rank < topk_remain) {
        topk_indices[num_above + rank] = target.idx;
      }
    }
    if (warp_id_1 < num_ties) {
      const auto target = tie_buffer[warp_id_1];
      const bool pred_0 = is_greater(tie_0, target);
      const bool pred_1 = is_greater(tie_1, target);
      const auto rank_0 = static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFF, pred_0)));
      const auto rank_1 = static_cast<uint32_t>(__popc(__ballot_sync(0xFFFFFFFF, pred_1)));
      const auto rank = rank_0 + rank_1;
      if (lane_id == 0 && rank < topk_remain) {
        topk_indices[num_above + rank] = target.idx;
      }
    }
  } else {
    [[unlikely]];
    for (auto i = warp_id; i < num_ties; i += v3::kNumWarps) {
      const auto target_tie = tie_buffer[i];
      uint32_t local_rank = 0;
      for (auto j = lane_id; j < num_ties; j += 32) {
        const auto tie = tie_buffer[j];
        if (is_greater(tie, target_tie)) local_rank++;
      }
      const auto rank = device::warp::reduce_sum(local_rank);
      if (lane_id == 0 && rank < topk_remain) {
        topk_indices[num_above + rank] = target_tie.idx;
      }
    }
  }
}

[[maybe_unused]]
SGL_DEVICE void naive_transform(
    const float* __restrict__,  // unused
    const int32_t* __restrict__ page_table,
    int32_t* __restrict__ indices,
    int32_t* __restrict__ raw_indices,  // optional: output raw abs position indices
    const uint32_t length,
    const uint32_t page_bits,
    const uint32_t topk) {
  for (uint32_t i = threadIdx.x; i < topk; i += v3::kBlockSize) {
    indices[i] = i < length ? page_to_indices(page_table, i, page_bits) : -1;
    if (raw_indices != nullptr) {
      raw_indices[i] = i < length ? static_cast<int32_t>(i) : -1;
    }
  }
}

[[maybe_unused]]
SGL_DEVICE void
radix_topk(const float* __restrict__ input, int32_t* __restrict__ output, const uint32_t length, const uint32_t topk) {
  ::sglang::device::legacy_radix_topk::
      select<kTopKBlockSize, kMaxTopK, static_cast<int>(kSMEM / (2 * sizeof(int32_t)))>(
          input, output, 0, static_cast<int>(length), static_cast<int>(topk));
}

template <bool kUsePDL>
__global__ __launch_bounds__(v3::kBlockSize, 2) void topk_512_transform_v3(const __grid_constant__ TopKParams params) {
  alignas(128) extern __shared__ uint8_t _smem[];
  __shared__ int32_t s_topk_indices[kMaxTopK];
  const auto& [scores, seq_lens, page_table, page_indices, raw_indices, score_stride, page_table_stride, page_bits, topk] =
      params;
  const uint32_t work_id = blockIdx.x;

  const auto score_ptr = scores + work_id * score_stride;
  const auto page_ptr = page_table == nullptr ? nullptr : page_table + work_id * page_table_stride;
  const auto indices_ptr = page_indices + work_id * topk;
  const auto raw_indices_ptr = raw_indices != nullptr ? raw_indices + work_id * topk : nullptr;

  device::PDLWaitPrimary<kUsePDL>();
  const uint32_t seq_len = seq_lens[work_id];

  if (seq_len <= topk) {
    // Strided output is needed when topk exceeds the 1024 threads per block.
    naive_transform(score_ptr, page_ptr, indices_ptr, raw_indices_ptr, seq_len, page_bits, topk);
  } else if (seq_len <= v3::kMax1PassLength && seq_len % 4 == 0 && reinterpret_cast<uintptr_t>(score_ptr) % 16 == 0) {
    register_topk<false>(score_ptr, s_topk_indices, seq_len, topk, _smem);
    __syncthreads();
    const auto tx = threadIdx.x;
    for (uint32_t i = tx; i < topk; i += v3::kBlockSize) {
      const auto raw = s_topk_indices[i];
      indices_ptr[i] = raw < 0 ? -1 : page_to_indices(page_ptr, raw, page_bits);
      if (raw_indices_ptr != nullptr) {
        raw_indices_ptr[i] = raw;
      }
    }
  } else if (seq_len <= v3::kMax2PassLength && seq_len % 4 == 0 && reinterpret_cast<uintptr_t>(score_ptr) % 16 == 0) {
    register_topk<true>(score_ptr, s_topk_indices, seq_len, topk, _smem);
    __syncthreads();
    const auto tx = threadIdx.x;
    for (uint32_t i = tx; i < topk; i += v3::kBlockSize) {
      const auto raw = s_topk_indices[i];
      indices_ptr[i] = raw < 0 ? -1 : page_to_indices(page_ptr, raw, page_bits);
      if (raw_indices_ptr != nullptr) {
        raw_indices_ptr[i] = raw;
      }
    }
  } else {
    // Fallback to radix_topk for sequences exceeding 32K
    radix_topk(score_ptr, s_topk_indices, seq_len, topk);
    __syncthreads();
    const auto tx = threadIdx.x;
    for (uint32_t i = tx; i < topk; i += v3::kBlockSize) {
      const auto raw = s_topk_indices[i];
      indices_ptr[i] = raw < 0 ? -1 : page_to_indices(page_ptr, raw, page_bits);
      if (raw_indices_ptr != nullptr) {
        raw_indices_ptr[i] = raw;
      }
    }
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
__global__ void topk_transform_kernel(const __grid_constant__ TopKParams params) {
  const auto &[
    scores, seq_lens, page_table, page_indices, raw_indices, // pointers
    score_stride, page_table_stride, page_bits, topk // sizes
  ] = params;
  const uint32_t work_id = blockIdx.x;

  const auto score_ptr = scores + work_id * score_stride;
  const auto page_ptr = page_table == nullptr ? nullptr : page_table + work_id * page_table_stride;
  const auto indices_ptr = page_indices + work_id * topk;
  const auto raw_indices_ptr = raw_indices != nullptr ? raw_indices + work_id * topk : nullptr;

  device::PDLWaitPrimary<kUsePDL>();
  const uint32_t seq_len = seq_lens[work_id];

  if (seq_len <= topk) {
    naive_transform(score_ptr, page_ptr, indices_ptr, raw_indices_ptr, seq_len, page_bits, topk);
  } else {
    __shared__ int32_t s_topk_indices[kMaxTopK];
    radix_topk(score_ptr, s_topk_indices, seq_len, topk);
    const auto tx = threadIdx.x;
    for (uint32_t i = tx; i < topk; i += v3::kBlockSize) {
      const auto raw = s_topk_indices[i];
      indices_ptr[i] = raw < 0 ? -1 : page_to_indices(page_ptr, raw, page_bits);
      if (raw_indices_ptr != nullptr) {
        raw_indices_ptr[i] = raw;
      }
    }
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct TopKKernel {
  static constexpr auto kernel = topk_512_transform_v3<kUsePDL>;

  static void transform(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::Optional<tvm::ffi::TensorView> page_table,
      const tvm::ffi::TensorView page_indices,
      const uint32_t page_size,
      const tvm::ffi::Optional<tvm::ffi::TensorView> raw_indices) {
    using namespace host;
    auto B = SymbolicSize{"batch_size"};
    auto S = SymbolicSize{"score_stride"};
    auto P = SymbolicSize{"page_table_stride"};
    auto K = SymbolicSize{"topk"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({B, -1})  // strided scores
        .with_strides({S, 1})
        .with_dtype<float>()
        .with_device(device)
        .verify(scores);
    TensorMatcher({B})  // seq_lens, must be contiguous
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(seq_lens);
    const int32_t* page_table_ptr = nullptr;
    int64_t page_table_stride = 0;
    if (page_table.has_value()) {
      TensorMatcher({B, -1})  // strided page table
          .with_strides({P, 1})
          .with_dtype<int32_t>()
          .with_device(device)
          .verify(page_table.value());
      page_table_ptr = static_cast<const int32_t*>(page_table.value().data_ptr());
      page_table_stride = P.unwrap();
    }
    TensorMatcher({B, K})  // output, must be contiguous
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(page_indices);

    int32_t* raw_indices_ptr = nullptr;
    if (raw_indices.has_value()) {
      TensorMatcher({B, K})  // optional raw indices output, must be contiguous
          .with_dtype<int32_t>()
          .with_device(device)
          .verify(raw_indices.value());
      raw_indices_ptr = static_cast<int32_t*>(raw_indices.value().data_ptr());
    }

    RuntimeCheck(std::has_single_bit(page_size), "page_size must be power of 2");
    const auto page_bits = static_cast<uint32_t>(std::countr_zero(page_size));
    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    const auto topk = static_cast<uint32_t>(K.unwrap());
    RuntimeCheck(topk > 0 && topk <= kMaxTopK, "topk must be in (0, 2048]");
    const auto params = TopKParams{
        .scores = static_cast<float*>(scores.data_ptr()),
        .seq_lens = static_cast<int32_t*>(seq_lens.data_ptr()),
        .page_table = page_table_ptr,
        .page_indices = static_cast<int32_t*>(page_indices.data_ptr()),
        .raw_indices = raw_indices_ptr,
        .score_stride = S.unwrap(),
        .page_table_stride = page_table_stride,
        .page_bits = page_bits,
        .topk = topk,
    };
    constexpr auto kSMEM_ = sizeof(RegisterTopKSmem) + 128;
    runtime::set_max_dynamic_smem_per_device<kernel>(device.unwrap(), kSMEM_);
    LaunchKernel(batch_size, v3::kBlockSize, device.unwrap(), kSMEM_).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
