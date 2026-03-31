#include "rtp_llm/cpp/core/ExecOps.h"
#include "rtp_llm/cpp/core/DistributedComm.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include <gtest/gtest.h>

using namespace rtp_llm;

class ExecOpsTest: public ::testing::Test {
protected:
    static ExecInitParams test_params_;

    static void SetUpTestSuite() {
        // Use initExecCtx with minimal config objects.
        // initExecCtx returns ExecInitParams after performing runtime init.
        ParallelismConfig pc;
        pc.tp_size          = 1;
        pc.local_world_size = 1;
        ModelConfig mc;
        test_params_ = initExecCtx(pc,
                                   mc,
                                   EPLBConfig{},
                                   FMHAConfig{},
                                   DeviceResourceConfig{},
                                   MoeConfig{},
                                   SpeculativeExecutionConfig{},
                                   MiscellaneousConfig{},
                                   ProfilingDebugLoggingConfig{},
                                   HWKernelConfig{},
                                   ConcurrencyConfig{},
                                   FfnDisAggregateConfig{},
                                   RuntimeConfig{},
                                   ModelSpecificConfig{});
    }
};

ExecInitParams ExecOpsTest::test_params_;

TEST_F(ExecOpsTest, testInitExecCtxIdempotent) {
    ASSERT_EQ(test_params_.tp_size, 1);
    ASSERT_EQ(test_params_.dp_size, 1);
}

TEST_F(ExecOpsTest, testGetEnableCommOverlap) {
    // Default DeviceResourceConfig has enable_comm_overlap = some value;
    // just verify the accessor works.
    (void)getEnableCommOverlap();
}

TEST_F(ExecOpsTest, testRuntimeSyncAndCheck) {
    ASSERT_NO_THROW(runtimeSyncAndCheck());
}

TEST_F(ExecOpsTest, testRuntimeCreateEvent) {
    auto event = runtimeCreateEvent();
    ASSERT_NE(event, nullptr);
    ASSERT_NO_THROW(event->synchronize());
}

TEST_F(ExecOpsTest, testCopyD2D) {
    auto       src = torch::randn({16}, torch::kCUDA);
    auto       dst = torch::empty({16}, torch::kCUDA);
    CopyParams params{dst, src};
    ASSERT_NO_THROW(execCopy(params));
    runtimeSyncAndCheck();
    ASSERT_TRUE(torch::equal(src, dst));
}

TEST_F(ExecOpsTest, testCopyH2D) {
    auto       src = torch::randn({16}, torch::kCPU);
    auto       dst = torch::empty({16}, torch::kCUDA);
    CopyParams params{dst, src};
    ASSERT_NO_THROW(execCopy(params));
    runtimeSyncAndCheck();
    ASSERT_TRUE(torch::equal(src, dst.cpu()));
}

TEST_F(ExecOpsTest, testCopyD2H) {
    auto       src = torch::randn({16}, torch::kCUDA);
    auto       dst = torch::empty({16}, torch::kCPU);
    CopyParams params{dst, src};
    ASSERT_NO_THROW(execCopy(params));
    ASSERT_TRUE(torch::equal(src.cpu(), dst));
}

TEST_F(ExecOpsTest, testNoBlockCopy) {
    auto       src = torch::randn({32}, torch::kCUDA);
    auto       dst = torch::empty({32}, torch::kCUDA);
    CopyParams params{dst, src};
    ASSERT_NO_THROW(execNoBlockCopy(params));
    runtimeSyncAndCheck();
    ASSERT_TRUE(torch::equal(src, dst));
}

TEST_F(ExecOpsTest, testBatchCopyD2D) {
    auto src1 = torch::randn({8}, torch::kCUDA);
    auto src2 = torch::randn({16}, torch::kCUDA);
    auto dst1 = torch::empty({8}, torch::kCUDA);
    auto dst2 = torch::empty({16}, torch::kCUDA);

    BatchCopyParams params;
    auto&           d2d = params.copy_buffers[BatchCopyParams::D2D];
    d2d.src_ptr.push_back(src1.data_ptr());
    d2d.dst_ptr.push_back(dst1.data_ptr());
    d2d.sizes.push_back(src1.nbytes());
    d2d.src_ptr.push_back(src2.data_ptr());
    d2d.dst_ptr.push_back(dst2.data_ptr());
    d2d.sizes.push_back(src2.nbytes());

    ASSERT_NO_THROW(execBatchCopy(params));
    runtimeSyncAndCheck();
    ASSERT_TRUE(torch::equal(src1, dst1));
    ASSERT_TRUE(torch::equal(src2, dst2));
}

TEST_F(ExecOpsTest, testBatchCopyH2D) {
    auto src = torch::randn({8}, torch::kCPU);
    auto dst = torch::empty({8}, torch::kCUDA);

    BatchCopyParams params;
    auto&           h2d = params.copy_buffers[BatchCopyParams::H2D];
    h2d.src_ptr.push_back(src.data_ptr());
    h2d.dst_ptr.push_back(dst.data_ptr());
    h2d.sizes.push_back(src.nbytes());

    ASSERT_NO_THROW(execBatchCopy(params));
    runtimeSyncAndCheck();
    ASSERT_TRUE(torch::equal(src, dst.cpu()));
}

TEST_F(ExecOpsTest, testBatchCopyD2H) {
    auto src = torch::randn({8}, torch::kCUDA);
    auto dst = torch::empty({8}, torch::kCPU);

    BatchCopyParams params;
    auto&           d2h = params.copy_buffers[BatchCopyParams::D2H];
    d2h.src_ptr.push_back(src.data_ptr());
    d2h.dst_ptr.push_back(dst.data_ptr());
    d2h.sizes.push_back(src.nbytes());

    ASSERT_NO_THROW(execBatchCopy(params));
    ASSERT_TRUE(torch::equal(src.cpu(), dst));
}

TEST_F(ExecOpsTest, testProcessGroupRegistration) {
    ASSERT_FALSE(hasProcessGroup(ParallelMode::TP));

    clearProcessGroups();
    ASSERT_FALSE(hasProcessGroup(ParallelMode::TP));
}

TEST_F(ExecOpsTest, testGetGpuMemoryStatus) {
    auto status = getGpuMemoryStatus();
    ASSERT_GT(status.free_bytes, 0u);
    ASSERT_GT(status.available_bytes, 0u);
}

TEST_F(ExecOpsTest, testExecMaskLogits) {
    auto logits = torch::randn({2, 8}, torch::kCUDA);
    auto mask   = torch::zeros({2, 8}, torch::TensorOptions(torch::kBool).device(torch::kCUDA));
    mask[0][0]  = true;
    mask[1][3]  = true;

    ASSERT_NO_THROW(execMaskLogits(logits, mask));
    runtimeSyncAndCheck();
}
