#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include <gtest/gtest.h>

#if USING_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace rtp_llm {

// ExecOps exposes wrappers for these platform operations; this focused test does not call them.
GreedyOutput sampleGreedy(const GreedyParams&) {
    return {};
}

BeamSearchOutput sampleBeamSearch(BeamSearchParams) {
    return {};
}

void chainSpeculativeSampling(const SpeculativeSamplingParams&) {}

void multiMergeCopy(const MultiMergeCopyParams&) {}

void runtimeBatchCopy(const BatchCopyParams&) {}

}  // namespace rtp_llm

TEST(ExecOpsMemoryTraceTest, ReportsPeakAndResets) {
#if USING_CUDA
    EXPECT_NO_THROW(rtp_llm::getGpuExecStatus());

    auto baseline_allocation = torch::empty({1}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    rtp_llm::runtimeSyncAndCheck();
    c10::cuda::CUDACachingAllocator::emptyCache();

    rtp_llm::setTraceMemory(true);
    EXPECT_TRUE(rtp_llm::isTraceMemoryEnabled());
    auto allocation = torch::empty(
        {32 * 1024 * 1024}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    rtp_llm::runtimeSyncAndCheck();

    EXPECT_GE(rtp_llm::getGpuExecStatus().device_memory_status.max_consumed_bytes, allocation.nbytes());

    rtp_llm::setTraceMemory(true);
    EXPECT_GE(rtp_llm::getGpuExecStatus().device_memory_status.max_consumed_bytes, allocation.nbytes());

    rtp_llm::setTraceMemory(false);
    EXPECT_FALSE(rtp_llm::isTraceMemoryEnabled());
    EXPECT_EQ(rtp_llm::getGpuExecStatus().device_memory_status.max_consumed_bytes, 0u);

    allocation = torch::Tensor();
    rtp_llm::runtimeSyncAndCheck();

    rtp_llm::setTraceMemory(true);
    allocation = torch::empty(
        {32 * 1024 * 1024}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    rtp_llm::runtimeSyncAndCheck();
    EXPECT_GE(rtp_llm::getGpuExecStatus().device_memory_status.max_consumed_bytes, allocation.nbytes());
    rtp_llm::setTraceMemory(false);
    EXPECT_FALSE(rtp_llm::isTraceMemoryEnabled());

    allocation = torch::Tensor();
    rtp_llm::runtimeSyncAndCheck();
    c10::cuda::CUDACachingAllocator::emptyCache();

    rtp_llm::setTraceMemory(true);
    EXPECT_EQ(rtp_llm::getGpuExecStatus().device_memory_status.max_consumed_bytes, 0u);
    rtp_llm::setTraceMemory(false);
    EXPECT_FALSE(rtp_llm::isTraceMemoryEnabled());

    baseline_allocation = torch::Tensor();
#else
    GTEST_SKIP() << "memory peak tracing is only used by the CUDA warmup path";
#endif
}
