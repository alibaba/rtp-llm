#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/AlignedHostMemory.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostMemoryRegistration.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>

#include <ATen/cuda/CachingHostAllocator.h>
#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value): name_(name) {
        if (const char* old = std::getenv(name)) {
            old_value_ = old;
        }
        if (setenv(name, value, 1) != 0) {
            throw std::runtime_error("setenv failed");
        }
    }

    ~ScopedEnvVar() {
        if (old_value_) {
            (void)setenv(name_.c_str(), old_value_->c_str(), 1);
        } else {
            (void)unsetenv(name_.c_str());
        }
    }

private:
    std::string                name_;
    std::optional<std::string> old_value_;
};

TEST(AlignedHostMemoryTest, AllocatesAlignedWritablePinnedMemory) {
    constexpr size_t kUsableBytes = 8192;
    constexpr size_t kAlignment   = 4096;

    AlignedHostMemory memory(kUsableBytes, kAlignment, "test aligned host memory");

    ASSERT_NE(memory.data(), nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(memory.data()) % kAlignment, 0);
    const auto tensor = torch::from_blob(memory.data(), {kUsableBytes}, torch::TensorOptions().dtype(torch::kUInt8));
    EXPECT_TRUE(tensor.is_pinned());

    memory.data()[0]                = 0x12;
    memory.data()[kUsableBytes - 1] = 0x34;
    EXPECT_EQ(memory.data()[0], 0x12);
    EXPECT_EQ(memory.data()[kUsableBytes - 1], 0x34);
}

TEST(AlignedHostMemoryTest, TorchPinnedAllocatorRoundsLargeAllocationToPowerOfTwo) {
    constexpr size_t kGiB          = 1024ULL * 1024 * 1024;
    constexpr size_t kUsableBytes  = 33 * kGiB;
    constexpr size_t kAlignment    = 4096;
    constexpr size_t kRoundedBytes = 64 * kGiB;
    ScopedEnvVar     pin_mode("RTP_LLM_HOST_BLOCK_POOL_PIN_MODE", "allocator");

    auto* allocator = at::getHostAllocator(at::kCUDA);
    allocator->empty_cache();
    const auto before = allocator->get_stats();

    {
        AlignedHostMemory memory(kUsableBytes, kAlignment, "pinned allocator rounding test");
        ASSERT_NE(memory.data(), nullptr);

        const auto during = allocator->get_stats();
        EXPECT_EQ(during.allocated_bytes.current - before.allocated_bytes.current, kRoundedBytes);
        EXPECT_EQ(during.active_bytes.current - before.active_bytes.current, kRoundedBytes);
    }

    allocator->empty_cache();
}

TEST(AlignedHostMemoryTest, RegisteredPinnedAllocationKeepsExactLargeSize) {
    if (!hostMemoryRegistrationSupported()) {
        GTEST_SKIP() << "host memory registration is not supported by this device backend";
    }
    constexpr size_t kGiB         = 1024ULL * 1024 * 1024;
    constexpr size_t kUsableBytes = 33 * kGiB;
    constexpr size_t kAlignment   = 4096;
    ScopedEnvVar     pin_mode("RTP_LLM_HOST_BLOCK_POOL_PIN_MODE", "register");

    auto* allocator = at::getHostAllocator(at::kCUDA);
    allocator->empty_cache();
    const auto before = allocator->get_stats();

    {
        AlignedHostMemory memory(kUsableBytes, kAlignment, "registered pinned allocation size test");
        ASSERT_NE(memory.data(), nullptr);
        EXPECT_TRUE(memory.isRegistered());
        EXPECT_EQ(memory.backingBytes(), kUsableBytes + kAlignment);

        const auto during = allocator->get_stats();
        EXPECT_EQ(during.allocated_bytes.current, before.allocated_bytes.current);
        EXPECT_EQ(during.active_bytes.current, before.active_bytes.current);
    }
}

}  // namespace
}  // namespace rtp_llm
