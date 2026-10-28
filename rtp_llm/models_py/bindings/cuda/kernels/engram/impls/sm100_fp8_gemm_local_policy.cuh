#pragma once

#include <rtp_llm/models_py/bindings/cuda/kernels/engram/scheduler/gemm.cuh>

namespace rtp_llm::engram {

// Preserve ordinary FP8/FP4 scheduling, grouped offsets, and local stores.
struct SM100FP8GemmLocalExecutionPolicy {
    using Arguments = cute::TmaDescriptor;

    static constexpr uint32_t kSFAExtraElems = 0;

    CUTLASS_DEVICE static uint32_t get_sfa_shift(const uint32_t) {
        return 0;
    }

    template<GemmType kGemmType,
             uint32_t BLOCK_M,
             uint32_t BLOCK_N,
             uint32_t kNumGroups,
             uint32_t kNumMulticast,
             bool     kIsMulticastOnA,
             uint32_t kNumSMs,
             bool     kEnsureZeroPadding,
             uint32_t kKAlignment,
             uint32_t kSFAlignment>
    using Scheduler = sched::Scheduler<kGemmType,
                                       BLOCK_M,
                                       BLOCK_N,
                                       kNumGroups,
                                       kNumMulticast,
                                       kIsMulticastOnA,
                                       kNumSMs,
                                       kEnsureZeroPadding,
                                       kKAlignment,
                                       kSFAlignment>;

    template<GemmType, bool, typename, typename, typename, cute::UMMA::Major, uint32_t, bool>
    CUTLASS_DEVICE static void validate() {}

    template<bool kWithGroupOffset, uint32_t BLOCK_M, typename SchedulerT>
    CUTLASS_DEVICE static uint32_t
    get_global_m_idx(SchedulerT& scheduler, const uint32_t shape_m, const uint32_t m_block_idx) {
        return scheduler.template get_global_idx<kWithGroupOffset, sched::IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
    }

    template<uint32_t BLOCK_M, typename SchedulerT>
    CUTLASS_DEVICE static uint32_t get_sfa_m_idx(const SchedulerT&, const uint32_t, const uint32_t m_block_idx) {
        // Grouped SFA uses the original block index, without the A group offset.
        return m_block_idx * BLOCK_M;
    }

    CUTLASS_DEVICE static void prefetch_output(const Arguments& tensor_map) {
        cute::prefetch_tma_descriptor(&tensor_map);
    }

    CUTLASS_DEVICE static uint32_t get_store_m_idx(const uint32_t base_m_idx, const uint32_t, const Arguments&) {
        return base_m_idx;
    }

    CUTLASS_DEVICE static const cute::TmaDescriptor&
    get_store_tensor_map(const Arguments& tensor_map, const uint32_t, const uint32_t) {
        return tensor_map;
    }

    CUTLASS_DEVICE static void drain_epilogue(const uint32_t) {}
};

}  // namespace rtp_llm::engram
