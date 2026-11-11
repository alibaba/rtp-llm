#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
namespace rtp_llm {
class PerRankBlockTransferEngine;
}
namespace rtp_llm::benchmark {
class BenchmarkJsonWriter;
struct BenchmarkDeviceHostCopyStats {
    std::atomic<size_t>  cuda_3d_batch{0};
    std::atomic<size_t>  staged_sm{0};
    std::atomic<size_t>  cuda_batch{0};
    std::atomic<size_t>  generic{0};
    std::atomic<int64_t> lowest_api_ns{0};
    std::atomic<size_t>  lowest_api_calls{0};

    void reset() {
        cuda_3d_batch.store(0, std::memory_order_relaxed);
        staged_sm.store(0, std::memory_order_relaxed);
        cuda_batch.store(0, std::memory_order_relaxed);
        generic.store(0, std::memory_order_relaxed);
        lowest_api_ns.store(0, std::memory_order_relaxed);
        lowest_api_calls.store(0, std::memory_order_relaxed);
    }
};

void installStrategyRecorders(PerRankBlockTransferEngine&                          engine,
                              const std::shared_ptr<BenchmarkDeviceHostCopyStats>& stats);
void recordLegacyStaging(PerRankBlockTransferEngine& engine, BenchmarkJsonWriter& writer, bool is_swa);
}  // namespace rtp_llm::benchmark
