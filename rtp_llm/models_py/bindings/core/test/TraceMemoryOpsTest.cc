#include <gtest/gtest.h>

#if USING_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif

#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {
namespace {

class TraceMemoryOpsTest: public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        initRuntime(0, false, false, MlaOpsType::AUTO);
        auto init = torch::empty({1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        (void)init;
    }
};

TEST_F(TraceMemoryOpsTest, TraceWindowActivatesThenCloses) {
    setTraceMemory(true);
    EXPECT_TRUE(isTraceMemory());
    setTraceMemory(false);
    EXPECT_FALSE(isTraceMemory());

    const auto memory = getGpuExecStatus().device_memory_status;
    EXPECT_EQ(memory.torch_allocated_peak_bytes, 0u);
    EXPECT_GT(memory.total_bytes, 0u);
}

#if USING_CUDA
TEST_F(TraceMemoryOpsTest, TracksAllocatedPeakAndCurrent) {
    constexpr size_t kProbeBytes = 256ULL * 1024 * 1024;
    setTraceMemory(true);
    const auto baseline = getGpuExecStatus().device_memory_status;
    {
        auto held = torch::empty({static_cast<int64_t>(kProbeBytes)},
                                 torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        const auto memory = getGpuExecStatus().device_memory_status;
        EXPECT_GE(memory.torch_allocated_peak_bytes, baseline.allocated_bytes + kProbeBytes);
    }
    c10::cuda::CUDACachingAllocator::emptyCache();

    const auto memory = getGpuExecStatus().device_memory_status;
    EXPECT_GE(memory.torch_allocated_peak_bytes, baseline.allocated_bytes + kProbeBytes);
    EXPECT_LT(memory.allocated_bytes, baseline.allocated_bytes + kProbeBytes);
    setTraceMemory(false);
}
#endif

}  // namespace
}  // namespace rtp_llm
