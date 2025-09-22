#include "rtp_llm/cpp/cache/BlockRefCounter.h"
#include "rtp_llm/cpp/utils/Exception.h"
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <random>

namespace rtp_llm {

class BlockRefCounterTest: public ::testing::Test {
protected:
    void SetUp() override {
        // 测试用的块数量
        block_nums_ = 10;
    }

    void TearDown() override {
        // 清理工作
    }

protected:
    int block_nums_;
};

// ==================== 基础功能测试 ====================

// 测试带参数的构造函数
TEST_F(BlockRefCounterTest, testParameterizedConstructor) {
    BlockRefCounter counter(block_nums_);

    // 验证初始状态
    EXPECT_EQ(counter.busyBlockNum(), 0);

    // 验证所有块的初始引用计数为0（从索引1开始）
    for (int i = 1; i < block_nums_; ++i) {
        EXPECT_EQ(counter.getRefCounter(i), 0);
    }

    // 验证索引0不存在（应该抛异常）
    EXPECT_THROW(counter.getRefCounter(0), std::out_of_range);

    // 验证超出范围的索引不存在
    EXPECT_THROW(counter.getRefCounter(block_nums_), std::out_of_range);
}

// 测试多个块的引用计数操作
TEST_F(BlockRefCounterTest, testMultipleBlocksRefCount) {
    BlockRefCounter counter(block_nums_);

    std::vector<int> blocks = {1, 2, 3, 4, 5};

    // 初始状态
    for (int block : blocks) {
        EXPECT_EQ(counter.getRefCounter(block), 0);
    }
    EXPECT_EQ(counter.busyBlockNum(), 0);

    // 批量增加引用计数
    counter.incrementRefCounter(blocks);
    for (int block : blocks) {
        EXPECT_EQ(counter.getRefCounter(block), 1);
    }
    EXPECT_EQ(counter.busyBlockNum(), blocks.size());

    // 部分块再次增加引用计数
    std::vector<int> partial_blocks = {1, 3, 5};
    counter.incrementRefCounter(partial_blocks);
    EXPECT_EQ(counter.getRefCounter(1), 2);
    EXPECT_EQ(counter.getRefCounter(2), 1);
    EXPECT_EQ(counter.getRefCounter(3), 2);
    EXPECT_EQ(counter.getRefCounter(4), 1);
    EXPECT_EQ(counter.getRefCounter(5), 2);
    EXPECT_EQ(counter.busyBlockNum(), blocks.size());  // busy数量不变

    // 批量减少引用计数
    counter.decrementRefCounter(blocks);
    EXPECT_EQ(counter.getRefCounter(1), 1);
    EXPECT_EQ(counter.getRefCounter(2), 0);
    EXPECT_EQ(counter.getRefCounter(3), 1);
    EXPECT_EQ(counter.getRefCounter(4), 0);
    EXPECT_EQ(counter.getRefCounter(5), 1);
    EXPECT_EQ(counter.busyBlockNum(), 3);  // 2, 4号块不再busy

    // 再次减少引用计数
    counter.decrementRefCounter(partial_blocks);
    for (int block : blocks) {
        EXPECT_EQ(counter.getRefCounter(block), 0);
    }
    EXPECT_EQ(counter.busyBlockNum(), 0);
}

// 测试重复块索引
TEST_F(BlockRefCounterTest, testDuplicateBlockIndices) {
    BlockRefCounter counter(block_nums_);

    // 包含重复索引的向量
    std::vector<int> duplicate_blocks = {1, 2, 1, 3, 2, 1};

    // 增加引用计数
    counter.incrementRefCounter(duplicate_blocks);

    // 验证引用计数（每个块被增加的次数等于在向量中出现的次数）
    EXPECT_EQ(counter.getRefCounter(1), 3);  // 出现3次
    EXPECT_EQ(counter.getRefCounter(2), 2);  // 出现2次
    EXPECT_EQ(counter.getRefCounter(3), 1);  // 出现1次
    EXPECT_EQ(counter.busyBlockNum(), 3);    // 3个不同的块

    // 减少引用计数
    counter.decrementRefCounter(duplicate_blocks);

    // 验证所有块的引用计数都归零
    EXPECT_EQ(counter.getRefCounter(1), 0);
    EXPECT_EQ(counter.getRefCounter(2), 0);
    EXPECT_EQ(counter.getRefCounter(3), 0);
    EXPECT_EQ(counter.busyBlockNum(), 0);
}

// 测试空向量操作
TEST_F(BlockRefCounterTest, testEmptyVectorOperations) {
    BlockRefCounter counter(block_nums_);

    std::vector<int> empty_blocks;

    // 空向量操作不应该改变状态
    counter.incrementRefCounter(empty_blocks);
    EXPECT_EQ(counter.busyBlockNum(), 0);

    counter.decrementRefCounter(empty_blocks);
    EXPECT_EQ(counter.busyBlockNum(), 0);
}

// ==================== 错误处理测试 ====================

// 测试减少零引用计数的错误处理
TEST_F(BlockRefCounterTest, testDecrementZeroRefCount) {
    BlockRefCounter counter(block_nums_);

    std::vector<int> init_blocks  = {1};
    std::vector<int> mixed_blocks = {1, 2};  // 1有引用计数，2没有

    // 先给块1增加引用计数
    counter.incrementRefCounter(init_blocks);
    EXPECT_EQ(counter.getRefCounter(1), 1);
    EXPECT_EQ(counter.busyBlockNum(), 1);

    // 尝试减少混合状态的块，应该在遇到零引用计数时抛异常
    EXPECT_THROW(counter.decrementRefCounter(mixed_blocks), rtp_llm::FTException);

    // 验证状态：块1的引用计数应该已经被减少了
    EXPECT_EQ(counter.getRefCounter(1), 0);
    EXPECT_EQ(counter.busyBlockNum(), 0);
}

}  // namespace rtp_llm
