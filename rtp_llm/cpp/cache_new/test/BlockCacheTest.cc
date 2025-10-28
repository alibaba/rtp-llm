#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <algorithm>
#include "rtp_llm/cpp/cache_new/BlockCacheV1.h"

namespace rtp_llm {
namespace test {

typedef BlockCacheV1::CacheItem CacheItem;

class BlockCacheV1Test: public ::testing::Test {
protected:
    void SetUp() override {
        // 每个测试用例前初始化
        cache_ = std::make_unique<BlockCacheV1>(4);  // seq_size_per_block=4
    }

    void TearDown() override {
        cache_.reset();
    }

    std::unique_ptr<BlockCacheV1> cache_;
};

// ==================== 基础功能测试 ====================

TEST_F(BlockCacheV1Test, ConstructorTest) {
    // 测试构造函数
    BlockCacheV1 cache1(2);
    EXPECT_TRUE(cache1.empty());
    EXPECT_EQ(cache1.size(), 0);

    BlockCacheV1 cache2(8);
    EXPECT_TRUE(cache2.empty());
    EXPECT_EQ(cache2.size(), 0);
}

TEST_F(BlockCacheV1Test, MatchBasicTest) {
    // 测试put和match的基本功能
    std::vector<float> loss = {0.1f, 0.2f, 0.3f, 0.4f};

    // 空匹配
    auto result0 = cache_->match({1});
    EXPECT_TRUE(isNullBlockIdx(result0.matched_index));
    EXPECT_TRUE(result0.loss.empty());

    CacheItem item    = {101, 1, loss, false};
    auto      result1 = cache_->put(item);
    EXPECT_TRUE(result1);

    // put重复的key
    auto result2 = cache_->put(item);
    EXPECT_FALSE(result2);

    // TODO put loss，部分loss的情况？

    auto result3 = cache_->match(101);
    EXPECT_EQ(result3.matched_index, 1);
    EXPECT_EQ(result3.loss.size(), 4);

    auto result4 = cache_->match(102);
    EXPECT_TRUE(isNullBlockIdx(result4.matched_index));
    EXPECT_EQ(result4.loss.size(), 0);
}

TEST_F(BlockCacheV1Test, PopBasicTest) {
    // 测试基本的pop功能
    std::vector<int64_t> cache_keys = {101, 102, 103, 104, 105};
    std::vector<int>     block_ids  = {1, 2, 3, 4, 5};
    std::vector<float>   loss       = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f,
                                       1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f};

    CacheItem item1   = {101, 1, {0.1f, 0.2f, 0.3f, 0.4f}, false};
    auto      result1 = cache_->put(item1);
    EXPECT_TRUE(result1);
    CacheItem item2   = {102, 2, {0.5f, 0.6f, 0.7f, 0.8f}, false};
    auto      result2 = cache_->put(item2);
    EXPECT_TRUE(result2);
    CacheItem item3   = {103, 3, {0.9f, 1.0f, 1.1f, 1.2f}, false};
    auto      result3 = cache_->put(item3);
    EXPECT_TRUE(result3);
    CacheItem item4   = {104, 4, {1.3f, 1.4f, 1.5f, 1.6f}, false};
    auto      result4 = cache_->put(item4);
    EXPECT_TRUE(result4);
    CacheItem item5   = {105, 5, {1.7f, 1.8f, 1.9f, 2.0f}, false};
    auto      result5 = cache_->put(item5);
    EXPECT_TRUE(result5);

    EXPECT_EQ(cache_->size(), 5);

    // 最老的block被pop出来
    auto popped1 = cache_->pop(2);
    EXPECT_EQ(popped1.size(), 2);
    EXPECT_EQ(cache_->size(), 3);
    EXPECT_EQ(popped1[0], 1);
    EXPECT_EQ(popped1[1], 2);

    // 剩余的三个block被pop出来
    auto popped2 = cache_->pop(3);
    EXPECT_EQ(popped2.size(), 3);
    EXPECT_EQ(cache_->size(), 0);
    EXPECT_EQ(popped2[0], 3);
    EXPECT_EQ(popped2[1], 4);
    EXPECT_EQ(popped2[2], 5);

    // 空的cache，不能pop出来
    auto popped3 = cache_->pop(3);
    EXPECT_EQ(popped3.size(), 0);
    EXPECT_EQ(cache_->size(), 0);

    // 设置resident
    CacheItem item6   = {101, 1, {0.1f, 0.2f, 0.3f, 0.4f}, true};
    auto      result6 = cache_->put(item6);
    EXPECT_TRUE(result6);
    EXPECT_EQ(cache_->size(), 1);

    // resident项不会被pop
    auto popped4 = cache_->pop(2);
    EXPECT_EQ(popped4.size(), 0);
    EXPECT_EQ(cache_->size(), 1);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
