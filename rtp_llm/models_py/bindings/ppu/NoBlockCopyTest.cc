#include <gtest/gtest.h>

#include <future>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <torch/torch.h>

#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm {
namespace {

class PpuNoBlockCopyTest: public ::testing::Test {
protected:
    void SetUp() override {
        int count = 0;
        ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
        ASSERT_GE(count, 2) << "This regression requires two reserved PPU devices";
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    }

    torch::Tensor values(int offset) {
        return torch::arange(16384, torch::TensorOptions().dtype(torch::kInt32)) + offset;
    }
};

TEST_F(PpuNoBlockCopyTest, BackgroundThreadUsesPayloadDeviceAndRestoresCurrentDevice) {
    auto expected = values(17);
    auto device = expected.to(torch::Device(torch::kCUDA, 1));
    auto host = torch::zeros_like(expected);
    auto owner = acquireDeviceHostCopyStreams(1);
    auto job = std::async(std::launch::async, [&] {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        execNoBlockCopy(MultiCopyParams{{host}, {device}}, {owner, DeviceHostCopyDirection::D2H});
        int current = -1;
        ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
        EXPECT_EQ(current, 0);
        EXPECT_TRUE(torch::equal(host, expected));
        auto changed = values(41);
        execNoBlockCopy(MultiCopyParams{{device}, {changed}}, {owner, DeviceHostCopyDirection::H2D});
        EXPECT_TRUE(torch::equal(device.cpu(), changed));
    });
    job.get();
}

TEST_F(PpuNoBlockCopyTest, OneThreadCanAlternateDeviceStreamsWithoutExplicitContext) {
    auto expected0 = values(3);
    auto expected1 = values(29);
    auto device0 = expected0.to(torch::Device(torch::kCUDA, 0));
    auto device1 = expected1.to(torch::Device(torch::kCUDA, 1));
    auto job = std::async(std::launch::async, [&] {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        for (int i = 0; i < 8; ++i) {
            auto host0 = torch::zeros_like(expected0);
            auto host1 = torch::zeros_like(expected1);
            execNoBlockCopy(MultiCopyParams{{host1}, {device1}});
            execNoBlockCopy(MultiCopyParams{{host0}, {device0}});
            EXPECT_TRUE(torch::equal(host0, expected0));
            EXPECT_TRUE(torch::equal(host1, expected1));
        }
        int current = -1;
        ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
        EXPECT_EQ(current, 0);
    });
    job.get();
}

TEST_F(PpuNoBlockCopyTest, ConcurrentBackgroundTransfersPreserveIndependentPayloads) {
    std::vector<std::future<void>> jobs;
    for (int worker = 0; worker < 2; ++worker) {
        jobs.emplace_back(std::async(std::launch::async, [&, worker] {
            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
            auto expected = values(100 + worker);
            auto device = expected.to(torch::Device(torch::kCUDA, 1));
            auto owner = acquireDeviceHostCopyStreams(1);
            for (int i = 0; i < 16; ++i) {
                auto host = torch::zeros_like(expected);
                execNoBlockCopy(MultiCopyParams{{host}, {device}}, {owner, DeviceHostCopyDirection::D2H});
                EXPECT_TRUE(torch::equal(host, expected));
                execNoBlockCopy(MultiCopyParams{{device}, {host}}, {owner, DeviceHostCopyDirection::H2D});
            }
        }));
    }
    for (auto& job : jobs) {
        job.get();
    }
}

TEST_F(PpuNoBlockCopyTest, RejectsMismatchedOrMissingContextBeforeSubmittingCopies) {
    auto host = values(7);
    auto device = host.to(torch::Device(torch::kCUDA, 1));
    auto wrong = acquireDeviceHostCopyStreams(0);
    EXPECT_THROW(execNoBlockCopy(MultiCopyParams{{host}, {device}}, {wrong, DeviceHostCopyDirection::D2H}),
                 std::invalid_argument);
    EXPECT_THROW(execNoBlockCopy(MultiCopyParams{{host}, {device}}, {nullptr, DeviceHostCopyDirection::D2H}),
                 std::invalid_argument);
    EXPECT_THROW(acquireDeviceHostCopyStreams(-1), std::invalid_argument);
}

}  // namespace
}  // namespace rtp_llm
