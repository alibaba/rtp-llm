#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm {
namespace {

constexpr size_t kMiB = 1024ULL * 1024ULL;

class NoBlockCopyScratchTest: public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&device_buffer_, 4 * kMiB), cudaSuccess);
        host_buffer_.resize(4 * kMiB, 0x5a);
    }

    void TearDown() override {
        releaseStagedMemoryCopyScratch(scratch_);
        if (device_buffer_ != nullptr) {
            EXPECT_EQ(cudaFree(device_buffer_), cudaSuccess);
        }
    }

    bool copyToDevice(size_t bytes) {
        StagedMemoryCopyParams params;
        params.host_base    = host_buffer_.data();
        params.host_bytes   = bytes;
        params.tiles        = {{device_buffer_, 0, bytes}};
        params.device_index = 0;
        params.direction    = StagedMemoryCopyDirection::H2D;
        return execStagedMemoryCopy(params, &scratch_);
    }

    std::vector<uint8_t>    host_buffer_;
    void*                   device_buffer_ = nullptr;
    StagedMemoryCopyScratch scratch_;
};

TEST_F(NoBlockCopyScratchTest, ReusesPinnedHostAllocationWithinGrowthCapacity) {
    ASSERT_TRUE(copyToDevice(64 * 1024));
    ASSERT_EQ(scratch_.host_capacity, kMiB);
    ASSERT_EQ(scratch_.host_allocation_count, 1);
    void* first_staging = scratch_.host_staging;

    ASSERT_TRUE(copyToDevice(512 * 1024));
    EXPECT_EQ(scratch_.host_staging, first_staging);
    EXPECT_EQ(scratch_.host_capacity, kMiB);
    EXPECT_EQ(scratch_.host_allocation_count, 1);

    ASSERT_TRUE(copyToDevice(kMiB));
    EXPECT_EQ(scratch_.host_staging, first_staging);
    EXPECT_EQ(scratch_.host_allocation_count, 1);

    ASSERT_TRUE(copyToDevice(kMiB + 16));
    ASSERT_NE(scratch_.host_staging, first_staging);
    ASSERT_EQ(scratch_.host_capacity, 2 * kMiB);
    ASSERT_EQ(scratch_.host_allocation_count, 2);
    void* second_staging = scratch_.host_staging;

    ASSERT_TRUE(copyToDevice(1536 * 1024));
    EXPECT_EQ(scratch_.host_staging, second_staging);
    EXPECT_EQ(scratch_.host_capacity, 2 * kMiB);
    EXPECT_EQ(scratch_.host_allocation_count, 2);

    releaseStagedMemoryCopyScratch(scratch_);
    EXPECT_EQ(scratch_.host_staging, nullptr);
    EXPECT_EQ(scratch_.host_capacity, 0);
    EXPECT_EQ(scratch_.host_allocation_count, 0);
}

}  // namespace
}  // namespace rtp_llm
