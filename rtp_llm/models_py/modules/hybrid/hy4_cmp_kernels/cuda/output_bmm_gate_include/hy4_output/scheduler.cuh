#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/scheduler/gemm.cuh>

#include <cstdint>

namespace hy4_output {

enum class Bf16SchedulerKind {
  kDeepGemm,
  kRuntimeBatched,
  kFixedSplitK,
};

enum class PdlTriggerPoint {
  kNone,
  kBeforeWait,
  kAfterWait,
};

// Persistent scheduler for uniform batched GEMMs. The launch grid controls the
// worker budget without changing the logical task set.
template <std::uint32_t kBlockM, std::uint32_t kBlockN,
          std::uint32_t kNumGroups, bool kIsMulticastOnA>
struct RuntimeBatchedScheduler {
  int current_iter = -1;
  std::uint32_t current_group_idx = 0;
  std::uint32_t current_shape_k;
  std::uint32_t num_m_blocks;
  std::uint32_t num_n_blocks;
  std::uint32_t num_blocks_per_group;

  CUTLASS_DEVICE explicit RuntimeBatchedScheduler(
      const std::uint32_t& shape_m, const std::uint32_t& shape_n,
      const std::uint32_t& shape_k, int* = nullptr)
      : current_shape_k(shape_k),
        num_m_blocks(::deep_gemm::math::ceil_div(shape_m, kBlockM)),
        num_n_blocks(::deep_gemm::math::ceil_div(shape_n, kBlockN)),
        num_blocks_per_group(num_m_blocks * num_n_blocks) {}

  CUTLASS_DEVICE bool get_next_block(std::uint32_t& m_block_idx,
                                     std::uint32_t& n_block_idx) {
    const std::uint32_t linear_task =
        static_cast<std::uint32_t>(++current_iter) * gridDim.x + blockIdx.x;
    if (linear_task >= kNumGroups * num_blocks_per_group) {
      return false;
    }

    current_group_idx = linear_task / num_blocks_per_group;
    const std::uint32_t block_in_group =
        linear_task - current_group_idx * num_blocks_per_group;
    if constexpr (kIsMulticastOnA) {
      m_block_idx = block_in_group / num_n_blocks;
      n_block_idx = block_in_group % num_n_blocks;
    } else {
      m_block_idx = block_in_group % num_m_blocks;
      n_block_idx = block_in_group / num_m_blocks;
    }
    return true;
  }

  CUTLASS_DEVICE std::uint32_t get_aligned_effective_m_in_block(
      const std::uint32_t&) const {
    return kBlockM;
  }

  template <bool kWithGroupOffset,
            ::deep_gemm::sched::IndexType kIndexType =
                ::deep_gemm::sched::IndexType::MN>
  CUTLASS_DEVICE std::uint32_t get_global_idx(
      const std::uint32_t shape_dim, const std::uint32_t block_size,
      const std::uint32_t& block_idx, const std::uint32_t& = 0) const {
    (void)kWithGroupOffset;
    const std::uint32_t offset =
        kIndexType == ::deep_gemm::sched::IndexType::SF_K
            ? current_group_idx * shape_dim
            : 0;
    return offset + block_idx * block_size;
  }
};

// O(1) persistent mapping for one GEMM split uniformly along K. Every split
// accumulates into the same output tile through the DeepGEMM TMA epilogue.
template <std::uint32_t kBlockM, std::uint32_t kBlockN,
          std::uint32_t kNumSplits>
struct FixedSplitKScheduler {
  static_assert(kNumSplits > 0);

  int current_iter = -1;
  std::uint32_t current_group_idx = 0;
  std::uint32_t current_shape_k;
  std::uint32_t num_m_blocks;
  std::uint32_t num_n_blocks;
  std::uint32_t num_blocks_per_split;

  CUTLASS_DEVICE explicit FixedSplitKScheduler(
      const std::uint32_t& shape_m, const std::uint32_t& shape_n,
      const std::uint32_t& shape_k, int* = nullptr)
      : current_shape_k(shape_k / kNumSplits),
        num_m_blocks(::deep_gemm::math::ceil_div(shape_m, kBlockM)),
        num_n_blocks(::deep_gemm::math::ceil_div(shape_n, kBlockN)),
        num_blocks_per_split(num_m_blocks * num_n_blocks) {}

  CUTLASS_DEVICE bool get_next_block(std::uint32_t& m_block_idx,
                                     std::uint32_t& n_block_idx) {
    const std::uint32_t linear_task =
        static_cast<std::uint32_t>(++current_iter) * gridDim.x + blockIdx.x;
    if (linear_task >= kNumSplits * num_blocks_per_split) {
      return false;
    }

    current_group_idx = linear_task / num_blocks_per_split;
    const std::uint32_t block_in_split =
        linear_task - current_group_idx * num_blocks_per_split;
    m_block_idx = block_in_split % num_m_blocks;
    n_block_idx = block_in_split / num_m_blocks;
    return true;
  }

  CUTLASS_DEVICE std::uint32_t get_aligned_effective_m_in_block(
      const std::uint32_t&) const {
    return kBlockM;
  }

  template <bool,
            ::deep_gemm::sched::IndexType kIndexType =
                ::deep_gemm::sched::IndexType::MN>
  CUTLASS_DEVICE std::uint32_t get_global_idx(
      const std::uint32_t, const std::uint32_t block_size,
      const std::uint32_t& block_idx, const std::uint32_t& = 0) const {
    const std::uint32_t split_offset =
        kIndexType == ::deep_gemm::sched::IndexType::K
            ? current_group_idx * current_shape_k
            : 0;
    return split_offset + block_idx * block_size;
  }
};

}  // namespace hy4_output
