#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <algorithm>
#include "rtp_llm/cpp/cache_new/BlockCache.h"

namespace rtp_llm {
namespace test {

typedef BlockCache::CacheItem CacheItem;

class BlockCacheV1Test: public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize before each test case
        cache_ = std::make_unique<BlockCache>();
    }

    void TearDown() override {
        cache_.reset();
    }

    std::unique_ptr<BlockCache> cache_;
};

// ==================== Basic functionality tests ====================

TEST_F(BlockCacheV1Test, ConstructorTest) {
    // Test constructor
    BlockCache cache1;
    EXPECT_TRUE(cache1.empty());
    EXPECT_EQ(cache1.size(), 0);
}

TEST_F(BlockCacheV1Test, MatchBasicTest) {
    // 测试put和match的基本功能
    // 空匹配
    auto result0 = cache_->match(1);
    EXPECT_TRUE(isNullBlockIdx(result0.matched_index));

    CacheItem item    = {101, 0, 1, false};
    auto      result1 = cache_->put(item);
    EXPECT_TRUE(result1);

    // Put a duplicate key
    auto result2 = cache_->put(item);
    EXPECT_FALSE(result2);

    auto result3 = cache_->match(101);
    EXPECT_EQ(result3.matched_index, 1);

    auto result4 = cache_->match(102);
    EXPECT_TRUE(isNullBlockIdx(result4.matched_index));
}

TEST_F(BlockCacheV1Test, PopBasicTest) {
    // Test basic pop functionality
    std::vector<int64_t> cache_keys = {101, 102, 103, 104, 105};
    std::vector<int>     block_ids  = {1, 2, 3, 4, 5};

    CacheItem item1   = {101, 0, 1, false};
    auto      result1 = cache_->put(item1);
    EXPECT_TRUE(result1);
    CacheItem item2   = {102, 0, 2, false};
    auto      result2 = cache_->put(item2);
    EXPECT_TRUE(result2);
    CacheItem item3   = {103, 0, 3, false};
    auto      result3 = cache_->put(item3);
    EXPECT_TRUE(result3);
    CacheItem item4   = {104, 0, 4, false};
    auto      result4 = cache_->put(item4);
    EXPECT_TRUE(result4);
    CacheItem item5   = {105, 0, 5, false};
    auto      result5 = cache_->put(item5);
    EXPECT_TRUE(result5);

    EXPECT_EQ(cache_->size(), 5);

    // The oldest blocks are popped
    auto popped1 = cache_->pop(2);
    EXPECT_EQ(popped1.size(), 2);
    EXPECT_EQ(cache_->size(), 3);
    EXPECT_EQ(popped1[0], 1);
    EXPECT_EQ(popped1[1], 2);

    // The remaining three blocks are popped
    auto popped2 = cache_->pop(3);
    EXPECT_EQ(popped2.size(), 3);
    EXPECT_EQ(cache_->size(), 0);
    EXPECT_EQ(popped2[0], 3);
    EXPECT_EQ(popped2[1], 4);
    EXPECT_EQ(popped2[2], 5);

    // An empty cache cannot pop any items
    auto popped3 = cache_->pop(3);
    EXPECT_EQ(popped3.size(), 0);
    EXPECT_EQ(cache_->size(), 0);

    // 设置resident
    CacheItem item6   = {101, 0, 1, true};
    auto      result6 = cache_->put(item6);
    EXPECT_TRUE(result6);
    EXPECT_EQ(cache_->size(), 1);

    // Resident entries won't be popped
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
