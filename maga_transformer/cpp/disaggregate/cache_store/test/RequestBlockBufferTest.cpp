#include "gtest/gtest.h"

#include "maga_transformer/cpp/disaggregate/cache_store/RequestBlockBuffer.h"

namespace rtp_llm {

class RequestBlockBufferTest: public ::testing::Test {};

TEST_F(RequestBlockBufferTest, testBlockOps) {

    RequestBlockBuffer buffer("test-request-id");

    ASSERT_EQ(0, buffer.getBlocksCount());

    std::shared_ptr<void> buffer1((void*)0x1, [](void* p) {});
    buffer.addBlock(std::make_shared<BlockBuffer>("b1", buffer1, 10, true, true));
    ASSERT_EQ(1, buffer.getBlocksCount());
    ASSERT_TRUE(buffer.isValid());
    ASSERT_EQ(buffer1, buffer.getBlock("b1")->addr);

    std::shared_ptr<void> buffer2((void*)0x2, [](void* p) {});
    buffer.addBlock("b2", buffer2, 10, true, true);
    ASSERT_EQ(2, buffer.getBlocksCount());
    ASSERT_TRUE(buffer.isValid());
    ASSERT_EQ(buffer2, buffer.getBlock("b2")->addr);

    buffer.addBlock("b3", nullptr, 10, true, true);
    ASSERT_EQ(3, buffer.getBlocksCount());
    ASSERT_FALSE(buffer.isValid());
    ASSERT_EQ(nullptr, buffer.getBlock("b3")->addr);
}

}  // namespace rtp_llm