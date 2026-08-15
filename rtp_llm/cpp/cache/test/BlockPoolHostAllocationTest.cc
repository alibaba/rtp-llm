#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include <torch/torch.h>

#include "rtp_llm/cpp/cache/BlockPool.h"

namespace rtp_llm {
namespace test {

TEST(BlockPoolHostAllocationTest, RequestsOnePinnedCpuAllocationAtExactSize) {
    constexpr int64_t    kPoolBytes = 4096;
    int                  calls      = 0;
    std::vector<int64_t> requested_sizes;
    torch::TensorOptions captured_options;
    auto                 owned_tensor = torch::empty({1}, torch::TensorOptions().dtype(torch::kUInt8));

    BlockPoolConfig config;
    config.total_size_bytes = kPoolBytes;
    BlockPool pool(config, AllocationType::HOST);
    pool.cache_buffer_allocator_ =
        [&](at::IntArrayRef sizes, const torch::TensorOptions& options) {
            ++calls;
            requested_sizes  = sizes.vec();
            captured_options = options;
            return owned_tensor;
        };

    pool.initializeCacheBuffer();

    EXPECT_EQ(calls, 1);
    EXPECT_EQ(requested_sizes, std::vector<int64_t>({kPoolBytes}));
    EXPECT_TRUE(captured_options.has_device());
    EXPECT_TRUE(captured_options.device().is_cpu());
    EXPECT_TRUE(captured_options.has_dtype());
    EXPECT_EQ(c10::typeMetaToScalarType(captured_options.dtype()), torch::kUInt8);
    EXPECT_EQ(captured_options.layout(), torch::kStrided);
    EXPECT_FALSE(captured_options.requires_grad());
    EXPECT_TRUE(captured_options.has_pinned_memory());
    EXPECT_TRUE(captured_options.pinned_memory());
    EXPECT_EQ(pool.getBaseAddress(), owned_tensor.data_ptr());
}

TEST(BlockPoolHostAllocationTest, LeavesDeviceAllocationUnpinned) {
    int calls = 0;

    BlockPoolConfig config;
    config.total_size_bytes = 4096;
    BlockPool pool(config, AllocationType::DEVICE);
    pool.cache_buffer_allocator_ =
        [&](at::IntArrayRef sizes, const torch::TensorOptions& options) {
            ++calls;
            EXPECT_EQ(sizes.vec(), std::vector<int64_t>({4096}));
            EXPECT_TRUE(options.device().is_cuda());
            EXPECT_EQ(c10::typeMetaToScalarType(options.dtype()), torch::kUInt8);
            EXPECT_FALSE(options.has_pinned_memory());
            return torch::empty({1}, torch::TensorOptions().dtype(torch::kUInt8));
        };

    pool.initializeCacheBuffer();

    EXPECT_EQ(calls, 1);
}

TEST(BlockPoolHostAllocationTest, PropagatesAllocationFailureWithoutFallback) {
    int calls = 0;

    BlockPoolConfig config;
    config.total_size_bytes = 4096;
    BlockPool pool(config, AllocationType::HOST);
    pool.cache_buffer_allocator_ = [&](at::IntArrayRef, const torch::TensorOptions&) -> torch::Tensor {
        ++calls;
        throw std::runtime_error("pinned allocation failed");
    };

    EXPECT_THROW(pool.initializeCacheBuffer(), std::runtime_error);
    EXPECT_EQ(calls, 1);
    EXPECT_FALSE(pool.cache_aligned_buffer_.defined());
    EXPECT_EQ(pool.getBaseAddress(), nullptr);
}

}  // namespace test
}  // namespace rtp_llm
