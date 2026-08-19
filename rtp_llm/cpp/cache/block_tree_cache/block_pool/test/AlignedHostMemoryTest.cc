#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/AlignedHostMemory.h"

#include <cstdint>

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

TEST(AlignedHostMemoryTest, AllocatesAlignedWritablePageableMemory) {
    constexpr size_t kUsableBytes = 8192;
    constexpr size_t kAlignment   = 4096;

    AlignedHostMemory memory(kUsableBytes, kAlignment, false, "test aligned host memory");

    ASSERT_NE(memory.data(), nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(memory.data()) % kAlignment, 0);
    EXPECT_FALSE(memory.isPinned());

    memory.data()[0]                = 0x12;
    memory.data()[kUsableBytes - 1] = 0x34;
    EXPECT_EQ(memory.data()[0], 0x12);
    EXPECT_EQ(memory.data()[kUsableBytes - 1], 0x34);
}

}  // namespace
}  // namespace rtp_llm
