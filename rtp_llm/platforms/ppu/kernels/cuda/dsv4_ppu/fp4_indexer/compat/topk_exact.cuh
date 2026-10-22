#pragma once

#include <cstdint>

namespace rtp_topk {

__device__ __forceinline__ uint32_t ordered_key(float score) {
  const uint32_t bits = score == 0.0f ? 0u : __float_as_uint(score);
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

// Overflow-only fallback. Rescan the original scores at each radix byte;
// candidate capacity never limits selection. Equal scores prefer lower indices.
template <uint32_t K, typename Score>
__device__ __forceinline__ void exact_topk(
    const Score* scores, int32_t* output, uint32_t length, void* workspace) {
  auto* histogram = static_cast<uint32_t*>(workspace);
  auto* warp_counts = histogram + 256;
  auto* state = warp_counts + 32;
  const uint32_t tx = threadIdx.x;
  const uint32_t lane = tx % 32;
  const uint32_t warp = tx / 32;
  __syncthreads();
  if (tx == 0) {
    state[0] = 0;  // Selected prefix.
    state[1] = 0;  // Prefix mask.
    state[2] = K;  // Rank within the selected prefix.
    state[3] = 0;  // Number of greater values emitted.
    state[4] = 0;  // Equal values in preceding index tiles.
  }
  __syncthreads();
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (tx < 256) histogram[tx] = 0;
    __syncthreads();
    const uint32_t prefix = state[0], mask = state[1];
    for (uint32_t i = tx; i < length; i += blockDim.x) {
      const uint32_t key = ordered_key(static_cast<float>(scores[i]));
      if ((key & mask) == prefix) atomicAdd(histogram + ((key >> shift) & 255u), 1u);
    }
    __syncthreads();
    if (tx == 0) {
      uint32_t rank = state[2];
      for (int bin = 255; bin >= 0; --bin) {
        if (rank <= histogram[bin]) {
          state[0] |= static_cast<uint32_t>(bin) << shift;
          state[1] |= 255u << shift;
          state[2] = rank;
          break;
        }
        rank -= histogram[bin];
      }
    }
    __syncthreads();
  }
  const uint32_t threshold = state[0], equal_needed = state[2];
  for (uint32_t base = 0; base < length; base += blockDim.x) {
    const uint32_t i = base + tx;
    const bool valid = i < length;
    const uint32_t key = valid ? ordered_key(static_cast<float>(scores[i])) : 0u;
    if (valid && key > threshold) output[atomicAdd(state + 3, 1u)] = i;
    const bool equal = valid && key == threshold;
    const uint32_t votes = __ballot_sync(0xffffffffu, equal);
    if (lane == 0) warp_counts[warp] = __popc(votes);
    __syncthreads();
    uint32_t rank = state[4] + __popc(votes & ((1u << lane) - 1u));
    for (uint32_t w = 0; w < warp; ++w) rank += warp_counts[w];
    if (equal && rank < equal_needed) output[K - equal_needed + rank] = i;
    __syncthreads();
    if (tx == 0) {
      for (uint32_t w = 0; w < blockDim.x / 32; ++w) state[4] += warp_counts[w];
    }
    __syncthreads();
  }
}

}  // namespace rtp_topk
