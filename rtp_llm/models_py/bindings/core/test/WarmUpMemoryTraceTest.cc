#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include <gtest/gtest.h>

using namespace rtp_llm;

// Covers the warmup peak-memory path (used by prefillWarmUp/decodeWarmUp to size the KV cache):
// setTraceMemory(true) snapshots a baseline, and getGpuExecStatus().max_consumed_bytes must then
// report the forward's transient reserved-memory growth. Kept in its own target so it stays
// isolated from the rest of the exec-ops suite.
class WarmUpMemoryTraceTest: public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        initRuntime(/*device_id=*/0,
                    /*trace_memory=*/false,
                    /*enable_comm_overlap=*/false,
                    MlaOpsType::AUTO);
    }
};

// Before the fix max_consumed_bytes was always 0; it must now reflect the forward's transient
// reserved-memory growth between setTraceMemory(true) and getGpuExecStatus().
TEST_F(WarmUpMemoryTraceTest, testTraceMemoryReportsPeak) {
    // Force CUDA context + caching-allocator device pool initialization before tracing. In
    // production setTraceMemory(true) runs inside warmUp, after the model weights are already
    // resident, so the pool exists. In a cold single-test process nothing has allocated yet, and
    // the first emptyCache/resetPeakStats throws "Invalid device argument" — so prime it here.
    { auto init = torch::empty({1}, torch::TensorOptions(torch::kByte).device(torch::kCUDA)); }
    runtimeSyncAndCheck();

    // Gate off by default: no peak is reported.
    EXPECT_EQ(getGpuExecStatus().device_memory_status.max_consumed_bytes, 0u);

    constexpr int64_t kProbeBytes = 64 * 1024 * 1024;  // 64 MiB

    setTraceMemory(true);  // emptyCache + resetPeakStats + snapshot baseline
    // Allocate after the baseline so it counts as transient growth; keep it alive across the query.
    auto probe = torch::empty({kProbeBytes}, torch::TensorOptions(torch::kByte).device(torch::kCUDA));
    runtimeSyncAndCheck();
    const auto peak = getGpuExecStatus().device_memory_status.max_consumed_bytes;
    setTraceMemory(false);

    // Allocator rounds up, so the reserved delta must be at least the probe, and within a sane
    // bound (not the whole device) — i.e. a reasonable, non-zero value.
    EXPECT_GE(peak, static_cast<size_t>(kProbeBytes));
    EXPECT_LT(peak, static_cast<size_t>(kProbeBytes) * 16);

    // Gate turns back off after setTraceMemory(false).
    EXPECT_EQ(getGpuExecStatus().device_memory_status.max_consumed_bytes, 0u);

    ASSERT_TRUE(probe.defined());
}
