#include "rtp_llm/cpp/core/CopyOps.h"
#include "rtp_llm/cpp/runtime/CudaRuntime.h"
#include "rtp_llm/cpp/testing/TestBase.h"

#include <gtest/gtest.h>

namespace rtp_llm {

class RocmRuntimeOpsTest: public DeviceTestBase {};

TEST_F(RocmRuntimeOpsTest, RuntimeCopyD2D) {
    auto src = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f}, torch::kFloat32).to(torch::kCUDA);
    auto dst = torch::zeros_like(src);

    runtimeCopy({dst, src});
    runtimeSyncAndCheck();

    EXPECT_TRUE(torch::equal(dst.cpu(), src.cpu()));
}

TEST_F(RocmRuntimeOpsTest, RuntimeCopyHostRoundTrip) {
    auto host_src = torch::tensor({1, 2, 3, 4}, torch::kUInt8);
    auto device   = torch::zeros({4}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    auto host_dst = torch::zeros_like(host_src);

    runtimeCopy({device, host_src});
    runtimeCopy({host_dst, device});

    EXPECT_TRUE(torch::equal(host_dst, host_src));
}

TEST_F(RocmRuntimeOpsTest, RuntimeBatchCopyD2DH2DAndD2H) {
    auto host_src   = torch::tensor({1, 2, 3, 4}, torch::kUInt8);
    auto device_src = torch::tensor({5, 6, 7, 8}, torch::kUInt8).to(torch::kCUDA);
    auto h2d_dst    = torch::zeros({4}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    auto d2d_dst    = torch::zeros_like(device_src);
    auto d2h_dst    = torch::zeros_like(host_src);

    BatchCopyParams params;
    params.add(h2d_dst.data_ptr(), host_src.data_ptr(), host_src.nbytes(), BatchCopyParams::H2D);
    params.add(d2d_dst.data_ptr(), device_src.data_ptr(), device_src.nbytes(), BatchCopyParams::D2D);
    params.add(d2h_dst.data_ptr(), device_src.data_ptr(), device_src.nbytes(), BatchCopyParams::D2H);
    runtimeBatchCopy(params);
    runtimeSyncAndCheck();

    EXPECT_TRUE(torch::equal(h2d_dst.cpu(), host_src));
    EXPECT_TRUE(torch::equal(d2d_dst.cpu(), device_src.cpu()));
    EXPECT_TRUE(torch::equal(d2h_dst, device_src.cpu()));
}

TEST_F(RocmRuntimeOpsTest, RuntimeNoBlockCopyH2D) {
    auto host_src = torch::tensor({9, 8, 7, 6}, torch::kUInt8);
    auto device   = torch::zeros({4}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

    runtimeNoBlockCopy({device, host_src});

    EXPECT_TRUE(torch::equal(device.cpu(), host_src));
}

TEST_F(RocmRuntimeOpsTest, RuntimeMultiMergeCopyUsesDeviceKernel) {
    auto src0 = torch::tensor({1, 2, 3}, torch::kUInt8).to(torch::kCUDA);
    auto src1 = torch::tensor({4, 5}, torch::kUInt8).to(torch::kCUDA);
    auto src2 = torch::tensor({6, 7, 8, 9}, torch::kUInt8).to(torch::kCUDA);
    auto dst  = torch::zeros({12}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

    MultiMergeCopyParams params{
        dst.data_ptr(), {src0.data_ptr(), src1.data_ptr(), src2.data_ptr()}, {3, 2, 4}, {0, 5, 8}};
    runtimeMultiMergeCopy(params);
    runtimeSyncAndCheck();

    auto expected = torch::tensor({1, 2, 3, 0, 0, 4, 5, 0, 6, 7, 8, 9}, torch::kUInt8);
    EXPECT_TRUE(torch::equal(dst.cpu(), expected));
}

TEST_F(RocmRuntimeOpsTest, RuntimeMaskLogitsIsExplicitlyUnsupported) {
    auto logits = torch::zeros({1, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto mask   = torch::zeros({1, 4}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

    EXPECT_ANY_THROW(runtimeMaskLogits(logits, mask));
}

TEST_F(RocmRuntimeOpsTest, EmptyFusedCopiesAreNoOps) {
    fusedCopy(FusedD2DCopyParams{});
    fusedStridedCopy(FusedStridedCopyParams{});
    runtimeSyncAndCheck();
}

}  // namespace rtp_llm
