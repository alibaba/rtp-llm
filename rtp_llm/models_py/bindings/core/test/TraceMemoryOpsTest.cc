#include <gtest/gtest.h>

#include <string>

#if USING_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif

#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/testing/TestLogCapture.h"

namespace rtp_llm {
namespace {

#if USING_CUDA
constexpr size_t kMiB = 1024 * 1024;
// Large enough to dominate allocator rounding and any neighbour noise, small enough to fit
// alongside whatever else shares the test GPU.
constexpr size_t kProbeBytes = 256 * kMiB;
// non_torch growth is read from device-global cudaMemGetInfo, so it is not exclusively ours.
constexpr size_t kNeighbourNoiseTolerance = 256 * kMiB;
#endif

// Covers the production free functions linked from ExecOps.cc -- setTraceMemory /
// getTraceMemoryState / isTraceMemory / finishTraceMemory / getGpuExecStatus -- which
// MemoryDiagnosticsTest deliberately does not: that suite exercises the TraceMemoryState
// class in isolation, without the CUDA baselines these entry points manage. This target is
// NOT tagged manual (unlike exec_ops_test) so the wildcard gates run it; it needs a GPU
// because setTraceMemory(true) touches the CUDA caching allocator.
//
// All cases share process-global trace state, so each one starts by forcing a known phase.

int phase() {
    return getTraceMemoryState();
}

// setTraceMemory(true) resets and reads the CUDA caching allocator's per-device stats, so two
// things must already hold or those calls throw c10's "Invalid device argument": the runtime
// must have selected a device (initRuntime, same fixture pattern as ExecOpsTest) AND the
// caching allocator must be initialized for it. The allocator initializes lazily on the first
// CUDA allocation -- in production the model weights do that long before warmup -- so the
// fixture allocates a throwaway tensor to reach the same state. trace_memory=false because
// these cases drive the trace phase themselves. Torch maps ROCm devices onto kCUDA, so the
// same fixture initializes both platforms.
class TraceMemoryOpsTest: public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        initRuntime(/*device_id=*/0,
                    /*trace_memory=*/false,
                    /*enable_comm_overlap=*/false,
                    MlaOpsType::AUTO);
        auto allocator_init = torch::empty({1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        (void)allocator_init;
    }
};

TEST_F(TraceMemoryOpsTest, StartupLifecycleActivatesThenFinishes) {
    finishTraceMemory();  // normalize whatever earlier tests left behind

    setTraceMemory(true);
    EXPECT_EQ(phase(), static_cast<int>(TraceMemoryPhase::Active));
    EXPECT_TRUE(isTraceMemory());

    setTraceMemory(false);
    EXPECT_EQ(phase(), static_cast<int>(TraceMemoryPhase::Finished));
    EXPECT_FALSE(isTraceMemory());

    // Outside the window getGpuExecStatus skips the entire breakdown block (it is guarded by
    // isTraceMemory()), so these fields read their zero defaults. What this pins is the
    // gating, NOT the baseline reset: a stale baseline is unobservable from outside the
    // window by construction. The reset itself is covered by ActiveWindowZeroesDeltas below,
    // where the fields are live.
    const auto mem = getGpuExecStatus().device_memory_status;
    EXPECT_EQ(mem.max_consumed_bytes, 0u);
    EXPECT_EQ(mem.torch_current_increase_bytes, 0u);
    EXPECT_EQ(mem.non_torch_increase_bytes, 0u);
    EXPECT_GT(mem.total_bytes, 0u);
}

#if USING_CUDA
TEST_F(TraceMemoryOpsTest, ActiveWindowZeroesDeltas) {
    setTraceMemory(true);
    // Sampled immediately after setTraceMemory(true) snapshotted the baselines and reset the
    // peak stats, with no allocation in between, so the two torch deltas must be exactly 0 --
    // they come from this process's own allocator. Here the fields are live, so a baseline that
    // was not re-snapshotted (e.g. carried over from a previous lifecycle) surfaces as a
    // non-zero delta and fails this case -- the check the post-close assertions cannot make.
    const auto mem = getGpuExecStatus().device_memory_status;
    EXPECT_EQ(mem.max_consumed_bytes, 0u);
    EXPECT_EQ(mem.torch_current_increase_bytes, 0u);
    // non_torch is derived from device-global cudaMemGetInfo, so a neighbour process on the
    // same GPU can move it between the baseline snapshot and this sample. Bound it instead of
    // demanding exact zero; the tolerance stands for "plausible neighbour noise", not for a
    // quantity this test measures.
    EXPECT_LT(mem.non_torch_increase_bytes, kNeighbourNoiseTolerance);
    EXPECT_GT(mem.total_bytes, 0u);
    EXPECT_GE(mem.total_bytes, mem.available_bytes);
    setTraceMemory(false);
}

// The zero-delta case above cannot tell max_consumed_bytes (peak) from
// torch_current_increase_bytes (current): both read 0. This one drives a real allocation so the
// two must diverge, which is the basis of the transient/resident split in makeWarmUpResult --
// peak stays at the high-water mark while current falls back once the allocation is released.
TEST_F(TraceMemoryOpsTest, ActiveWindowTracksPeakAndCurrentSeparately) {
    setTraceMemory(true);
    {
        auto held = torch::empty({static_cast<int64_t>(kProbeBytes)},
                                 torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        const auto in_scope = getGpuExecStatus().device_memory_status;
        EXPECT_GE(in_scope.max_consumed_bytes, kProbeBytes);
        EXPECT_GE(in_scope.torch_current_increase_bytes, kProbeBytes);
    }
    c10::cuda::CUDACachingAllocator::emptyCache();

    const auto after_release = getGpuExecStatus().device_memory_status;
    // Peak is a high-water mark: releasing must not lower it.
    EXPECT_GE(after_release.max_consumed_bytes, kProbeBytes);
    // Current fell back -- that difference is exactly the transient share the KV budget has to
    // reserve, because serving allocates it again.
    EXPECT_LT(after_release.torch_current_increase_bytes, kProbeBytes);
    setTraceMemory(false);
}
#endif  // USING_CUDA

TEST_F(TraceMemoryOpsTest, ForcedFinishFromActiveWarnsAndDiscardsTheWindow) {
    finishTraceMemory();
    setTraceMemory(true);
    {
        test::TestLogCapture capture("trace_memory_forced_finish");
        finishTraceMemory();
        EXPECT_NE(capture.content().find("forced Active -> Finished"), std::string::npos);
    }
    EXPECT_EQ(phase(), static_cast<int>(TraceMemoryPhase::Finished));
}

TEST_F(TraceMemoryOpsTest, FinishIsIdempotentOnceFinished) {
    setTraceMemory(false);
    {
        test::TestLogCapture capture("trace_memory_idempotent_finish");
        finishTraceMemory();
        finishTraceMemory();
        // Already-Finished repeats are the every-startup path (NormalEngine calls
        // finishTraceMemory after setTraceMemory(false)); they must stay silent.
        EXPECT_EQ(capture.content().find("finishTraceMemory"), std::string::npos);
    }
    EXPECT_EQ(phase(), static_cast<int>(TraceMemoryPhase::Finished));
}

}  // namespace
}  // namespace rtp_llm
