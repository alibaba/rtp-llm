/*
 * Adapted from SGLang e2982985110fb432fd8ddba5f0147a5004fd03ab,
 * python/sglang/jit_kernel/csrc/moe/route_radix.cuh. Apache-2.0.
 * https://github.com/sgl-project/sglang/blob/e2982985110fb432fd8ddba5f0147a5004fd03ab/LICENSE
 *
 * Retains the radix selector and deterministic ordering. Local adaptations:
 * FP32-only C++17 utilities, RTP finite-mask/expf/epsilon contracts, and a
 * normal-stream launch. No external TVM/PDL dependency is required.
 */
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include <type_traits>
#define K31_ROUTER_DEVICE __device__ __forceinline__
namespace rtp_llm::kimi_k3_router_radix {
using fp32_t = float;
using fp32x2_t = float2;
namespace device {
constexpr uint32_t kWarpThreads = 32;
template <typename T> using packed_t = float2;
template <typename Out, typename In> K31_ROUTER_DEVICE Out cast(In value) { return value; }
template <bool PDL> K31_ROUTER_DEVICE void PDLWaitPrimary() { static_assert(!PDL); }
template <bool PDL> K31_ROUTER_DEVICE void PDLTriggerSecondary() { static_assert(!PDL); }
template <typename T, unsigned Count>
struct alignas(sizeof(T) * Count) AlignedVector {
  T items[Count];
  K31_ROUTER_DEVICE T& operator[](unsigned i) { return items[i]; }
  K31_ROUTER_DEVICE const T& operator[](unsigned i) const { return items[i]; }
  K31_ROUTER_DEVICE void fill(T value) {
    #pragma unroll
    for (unsigned i = 0; i < Count; ++i) items[i] = value;
  }
  K31_ROUTER_DEVICE void load(const void* ptr, unsigned idx) {
    *this = reinterpret_cast<const AlignedVector*>(ptr)[idx];
  }
  K31_ROUTER_DEVICE void store(void* ptr, unsigned idx) const {
    reinterpret_cast<AlignedVector*>(ptr)[idx] = *this;
  }
};
namespace warp {
K31_ROUTER_DEVICE uint32_t reduce_sum(uint32_t value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __reduce_add_sync(0xffffffffu, value);
#else
  #pragma unroll
  for (unsigned offset = 16; offset; offset /= 2) value += __shfl_xor_sync(0xffffffffu, value, offset);
  return value;
#endif
}
K31_ROUTER_DEVICE uint32_t inclusive_sum(uint32_t value, uint32_t lane) {
  #pragma unroll
  for (unsigned offset = 1; offset < 32; offset *= 2) {
    const uint32_t prior = __shfl_up_sync(0xffffffffu, value, offset);
    if (lane >= offset) value += prior;
  }
  return value;
}
}
} }
namespace rtp_llm::kimi_k3_router_radix {

namespace moe::radix {

struct RadixSelectBase {
  static constexpr uint32_t kRadixBits = 8;
  static constexpr uint32_t kRadixSize = 1 << kRadixBits;
  static constexpr uint32_t kRadixRounds = 32 / kRadixBits;

  struct alignas(16) MatchBin {
    uint32_t bin;
    uint32_t above_count;  // active elements in bins strictly above `bin`
    uint32_t equal_count;  // active elements in bin `bin`
  };
};

inline constexpr float kNanFloor = -1e30f;

// Monotone unsigned key: larger biased -> larger key. Caller must have floored
// biased-NaN. Canonicalizes -0.0 -> +0.0 so equal values get equal keys.
K31_ROUTER_DEVICE uint32_t biased_to_key(float biased) {
  if (biased == 0.0f) biased = 0.0f;
  uint32_t u = __float_as_uint(biased);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// tl.sigmoid(x) = 1/(1+exp(-x)). Must stay instruction-identical to v1's
// sigmoid_match so both kernels rank (and weight) identically.
K31_ROUTER_DEVICE float sigmoid_match(float x) {
  return 1.0f / (1.0f + expf(-x));
}

K31_ROUTER_DEVICE float nan_floor(float x) {
  return (x == x) ? x : kNanFloor;
}

K31_ROUTER_DEVICE void bar_sync(uint32_t id, uint32_t num_threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(num_threads) : "memory");
}

// Exclusive prefix (block-wide, thread-rank order) of `cnt`. Uses
// smem_warp_sum[kNumWarps]; syncs on entry (so the workspace can be reused
// across calls) and before the cross-warp read.
K31_ROUTER_DEVICE uint32_t block_exclusive_sum(uint32_t cnt, uint32_t lane_id, uint32_t warp_id, uint32_t* smem_warp_sum) {
  const uint32_t inc = device::warp::inclusive_sum(cnt, lane_id);
  if (lane_id == 31) smem_warp_sum[warp_id] = inc;
  __syncthreads();
  // TODO: replace `__reduce_add_sync` with `warp::reduce_sum`
  const auto base = device::warp::reduce_sum(lane_id < warp_id ? smem_warp_sum[lane_id] : 0u);
  return base + inc - cnt;
}

}  // namespace moe::radix

inline constexpr uint32_t kNumExperts_ = 896;
inline constexpr uint32_t kTopK_ = 16;

struct LargeRouterRadixTrait : moe::radix::RadixSelectBase {
  static constexpr uint32_t kNumExperts = kNumExperts_;
  static constexpr uint32_t kTopK = kTopK_;
  static constexpr uint32_t kVecSize = 4;

  static constexpr uint32_t kBlockSize = kNumExperts / kVecSize;  // 224 = 7 warps
  static constexpr uint32_t kNumWarps = kBlockSize / 32;
  struct Smem {
    uint32_t warp_sum[3][kNumWarps];  // cross-warp scan workspace
    MatchBin match[kRadixRounds];
    uint32_t histogram[kRadixSize];
    // winner staging (compaction order = expert-id ascending)
    int32_t wid[kTopK];
    uint32_t wkey[kTopK];
    fp32_t wact[kTopK];
    // sorted staging ((key desc, id asc) order), only used when sorted != 0
    int32_t sid[kTopK];
    fp32_t sact[kTopK];
    fp32_t norm;
  };
};

struct RouteRadixParams {
  const void* __restrict__ scores;  // bf16 or fp32, typed by the kernel template
  const fp32_t* __restrict__ bias;
  fp32_t* __restrict__ out_w;
  int32_t* __restrict__ out_i;
  // Optional trtllm-gen routed-MoE packing: (id << 16) | bf16(weight) bits,
  // bit-identical to the standalone triton pack. nullptr skips the store.
  int32_t* __restrict__ out_packed;
  int M;
  long long scores_stride;
  long long out_w_stride;
  long long out_i_stride;
  long long out_packed_stride;
  float routed_scaling_factor;
  int renormalize;
  int apply_scale;
  int sorted;
};

// Whole-CTA routing body, callable from other kernels (the fused
// route+quant launch runs it on its first M CTAs). Routes row `blockIdx.x`;
// every thread of the 224-wide CTA must enter (block barriers inside).
template <bool kUsePDL, typename TScore>
K31_ROUTER_DEVICE void route_radix_block(const RouteRadixParams& params, typename LargeRouterRadixTrait::Smem& smem) {
  using namespace device;
  using T = LargeRouterRadixTrait;
  constexpr uint32_t kVecSize = T::kVecSize;
  constexpr uint32_t kRadixLanes = T::kRadixSize / 2;  // 128: 2 bins per thread
  enum { BAR_RESERVED = 0, BAR_SUM = 1 };

  const auto bx = blockIdx.x;
  const auto tx = threadIdx.x;
  const auto warp_id = tx / kWarpThreads;
  const auto lane_id = tx % kWarpThreads;
  // grid.x == M exactly; no row guard (an early return would deadlock the
  // block-wide barriers below).

  // ---- Load + key transform: thread tx owns experts [4*tx, 4*tx+4) ----
  uint32_t keys[kVecSize];
  float act[kVecSize];  // raw sigmoid (weight source) -- never NaN-sanitized
  {
    const auto scores = static_cast<const TScore*>(params.scores) + bx * params.scores_stride;
    AlignedVector<fp32x2_t, kVecSize / 2> bias_vec;
    // bf16: 2x bf16x2 (8B row loads); fp32: 2x fp32x2 (16B row loads). The
    // radix math below is fp32 either way -- only the load width differs.
    AlignedVector<packed_t<TScore>, kVecSize / 2> scores_vec;

    // Bias may be produced by a preceding cast or fill kernel (the caller
    // does not guarantee a frozen weight), so wait before loading either input.
    PDLWaitPrimary<kUsePDL>();
    bias_vec.load(params.bias, tx);
    scores_vec.load(scores, tx);

#pragma unroll
    for (uint32_t i = 0; i < kVecSize / 2; ++i) {
      fp32x2_t xy;
      if constexpr (std::is_same_v<TScore, fp32_t>) {
        xy = scores_vec[i];
      } else {
        xy = cast<fp32x2_t>(scores_vec[i]);
      }
      const auto [x, y] = xy;
      const bool vx = isfinite(x) && isfinite(bias_vec[i].x);
      const bool vy = isfinite(y) && isfinite(bias_vec[i].y);
      const float sx = vx ? moe::radix::sigmoid_match(x) : 0.0f;
      const float sy = vy ? moe::radix::sigmoid_match(y) : 0.0f;
      keys[2 * i + 0] = moe::radix::biased_to_key(vx ? sx + bias_vec[i].x : -INFINITY);
      keys[2 * i + 1] = moe::radix::biased_to_key(vy ? sy + bias_vec[i].y : -INFINITY);
      act[2 * i + 0] = sx;
      act[2 * i + 1] = sy;
    }
  }

  // ---- Radix narrowing, MSB -> LSB ----
  // Invariants entering round r:
  //   active[i]      <=> key's top 8r bits == threshold's top 8r bits
  //   total_active    = size of the active set
  //   topk            = winners still to take from the active set (1..total_active)
  bool active[kVecSize];
#pragma unroll
  for (uint32_t i = 0; i < kVecSize; ++i) {
    active[i] = true;
  }

  uint32_t total_active = T::kNumExperts;
  uint32_t topk = T::kTopK;
  uint32_t threshold = 0;      // assembled split-key prefix (unexamined low bits zero)
  uint32_t examined_mask = 0;  // bits of `threshold` that have been fixed
  bool take_all_equals = false;

  {
    AlignedVector<uint32_t, 2> zero;
    zero.fill(0);
    if (tx < kRadixLanes) zero.store(smem.histogram, tx);

#pragma unroll
    for (uint32_t round = 0; round < T::kRadixRounds; ++round) {
      __syncthreads();  // histogram zeroed & previous match consumed
      const uint32_t shift = 24 - round * 8;
      uint32_t bin[kVecSize];
#pragma unroll
      for (uint32_t i = 0; i < kVecSize; ++i) {
        bin[i] = (keys[i] >> shift) & 0xff;
      }
#pragma unroll
      for (uint32_t i = 0; i < kVecSize; ++i) {
        if (active[i]) atomicAdd(&smem.histogram[bin[i]], 1);
      }
      __syncthreads();

      // Split-bin search on 128 threads: thread t owns bins {2t, 2t+1}.
      // The split bin b is the unique bin with above(b) < topk <= above(b) + hist[b].
      if (tx < kRadixLanes) {
        AlignedVector<uint32_t, 2> hist;
        hist.load(smem.histogram, tx);
        const auto local_val = hist[0] + hist[1];
        const auto warp_inc = device::warp::inclusive_sum(local_val, lane_id);
        if (lane_id == kWarpThreads - 1) smem.warp_sum[0][warp_id] = warp_inc;
        moe::radix::bar_sync(BAR_SUM, kRadixLanes);
        const auto inter = device::warp::reduce_sum(lane_id < warp_id ? smem.warp_sum[0][lane_id] : 0u);
        const auto prefix = inter + warp_inc;        // active elements in bins [0, 2t+1]
        const auto above_r = total_active - prefix;  // in bins > 2t+1
        const auto above_m = above_r + hist[1];      // in bins > 2t
        const auto above_l = above_m + hist[0];      // in bins >= 2t
        if (above_r < topk && above_m >= topk) {
          smem.match[round] = {tx * 2 + 1, above_r, hist[1]};
        } else if (above_m < topk && above_l >= topk) {
          smem.match[round] = {tx * 2 + 0, above_m, hist[0]};
        }
      }
      __syncthreads();

      const auto [threshold_bin, above_count, equal_count] = smem.match[round];
      threshold |= threshold_bin << shift;
      examined_mask |= 0xffu << shift;
#pragma unroll
      for (uint32_t i = 0; i < kVecSize; ++i) {
        active[i] &= (bin[i] == threshold_bin);
      }
      total_active = equal_count;
      topk -= above_count;  // split condition guarantees 1 <= topk <= equal_count
      if (topk == equal_count) {
        // The remaining quota exactly covers the equal set: every active
        // element wins, no deeper narrowing or tie-break needed. At the last
        // round this is the no-full-key-tie case (the typical one).
        take_all_equals = true;
        break;
      }
      // Re-zero for the next round (synced by the loop-top barrier). Reaching
      // round 3 with topk < equal_count means a full-key tie: resolved below
      // by the smallest-id rank among `active`.
      if (round + 1 < T::kRadixRounds && tx < kRadixLanes) zero.store(smem.histogram, tx);
    }
  }

  // ---- Epilogue: collect the K winners ----
  // Strict winners: examined bits compare above the split prefix (these were
  // peeled off `active` in earlier rounds). Equal set (== `active`): take all
  // (take_all_equals) or the `topk` smallest ids (full-key tie-break).
  bool selected[kVecSize];
  if (take_all_equals) {
#pragma unroll
    for (uint32_t i = 0; i < kVecSize; ++i) {
      selected[i] = active[i] || (keys[i] & examined_mask) > threshold;
    }
  } else {  // deterministic tie-break
    uint32_t cnt = 0;
#pragma unroll
    for (uint32_t i = 0; i < kVecSize; ++i) {
      cnt += active[i] ? 1 : 0;
    }
    uint32_t rank = moe::radix::block_exclusive_sum(cnt, lane_id, warp_id, smem.warp_sum[1]);
#pragma unroll
    for (uint32_t i = 0; i < kVecSize; ++i) {
      const bool eq_win = active[i] && rank < topk;
      if (active[i]) ++rank;
      selected[i] = eq_win || (keys[i] & examined_mask) > threshold;
    }
  }

  // Compaction slots in expert-id order (deterministic).
  uint32_t selected_cnt = 0;
#pragma unroll
  for (uint32_t i = 0; i < kVecSize; ++i) {
    selected_cnt += selected[i] ? 1 : 0;
  }
  uint32_t slot = moe::radix::block_exclusive_sum(selected_cnt, lane_id, warp_id, smem.warp_sum[2]);
#pragma unroll
  for (uint32_t i = 0; i < kVecSize; ++i) {
    if (selected[i] && slot < T::kTopK) {
      smem.wid[slot] = (int32_t)(tx * kVecSize + i);
      smem.wkey[slot] = keys[i];
      smem.wact[slot] = act[i];
      ++slot;
    }
  }
  __syncthreads();

  static_assert(T::kTopK <= kWarpThreads);
  if (tx < T::kTopK) {
    uint32_t rank = tx;
    auto w = smem.wact[tx];
    const auto id = smem.wid[tx];
    if (params.sorted) {
      const uint32_t ka = smem.wkey[tx];
      const int32_t ia = id;
      rank = 0;
#pragma unroll
      for (uint32_t b = 0; b < T::kTopK; ++b) {
        if (smem.wkey[b] > ka || (smem.wkey[b] == ka && smem.wid[b] < ia)) ++rank;
      }
    }
    PDLTriggerSecondary<kUsePDL>();
    float sum = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < T::kTopK; ++i) {
      sum += smem.wact[i];
    }
    const auto norm = sum + 1e-20f;
    if (params.renormalize) w = w / norm;
    if (params.apply_scale) w = w * params.routed_scaling_factor;
    params.out_w[bx * params.out_w_stride + rank] = w;
    params.out_i[bx * params.out_i_stride + rank] = id;
    if (params.out_packed != nullptr) {
      // (id << 16) | bf16(w) bits -- RN float->bf16 matches the triton pack.
      const auto bits = static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(w)));
      params.out_packed[bx * params.out_packed_stride + rank] =
          static_cast<int32_t>((static_cast<uint32_t>(id) << 16) | bits);
    }
  }
}

template <bool kUsePDL, typename TScore>
__global__ __launch_bounds__(LargeRouterRadixTrait::kBlockSize)  //
    void route_radix_kernel(const __grid_constant__ RouteRadixParams params) {
  __shared__ typename LargeRouterRadixTrait::Smem smem;
  route_radix_block<kUsePDL, TScore>(params, smem);
}


}  // namespace rtp_llm::kimi_k3_router_radix

#undef K31_ROUTER_DEVICE
