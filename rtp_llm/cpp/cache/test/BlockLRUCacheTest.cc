#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <algorithm>
#include "rtp_llm/cpp/cache/BlockLRUCache.h"

namespace rtp_llm {
namespace test {

class BlockLRUCacheTest: public ::testing::Test {
protected:
    void SetUp() override {
        // 每个测试用例前初始化
        cache_ = std::make_unique<BlockLRUCache>(10, 4);  // capacity=10, seq_size_per_block=4
    }

    void TearDown() override {
        cache_.reset();
    }

    std::unique_ptr<BlockLRUCache> cache_;
};

// ==================== 基础功能测试 ====================

TEST_F(BlockLRUCacheTest, ConstructorTest) {
    // 测试构造函数
    BlockLRUCache cache1(5, 2);
    EXPECT_TRUE(cache1.empty());
    EXPECT_EQ(cache1.size(), 0);

    BlockLRUCache cache2(100, 8);
    EXPECT_TRUE(cache2.empty());
    EXPECT_EQ(cache2.size(), 0);
}

TEST_F(BlockLRUCacheTest, MatchBasicTest) {
    // 测试put和match的基本功能

    // 空匹配
    auto result0 = cache_->match({1, 2, 3});
    EXPECT_EQ(result0.matched_len, 0);
    EXPECT_TRUE(result0.block_ids.empty());
    EXPECT_TRUE(result0.losses.empty());

    std::vector<int64_t> cache_keys = {101, 102, 103};
    std::vector<int>     block_ids  = {1, 2, 3};
    std::vector<float>   losses     = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};

    auto not_cached = cache_->put(cache_keys, block_ids, losses, false);
    EXPECT_TRUE(not_cached.empty());

    // 完全匹配
    auto result1 = cache_->match(cache_keys);
    EXPECT_EQ(result1.matched_len, 3);
    EXPECT_EQ(result1.block_ids, block_ids);
    EXPECT_EQ(result1.losses.size(), 12);  // 3个block * 4个loss

    // 部分匹配
    auto result2 = cache_->match({101, 102, 104, 105, 106});
    EXPECT_EQ(result2.matched_len, 2);
    EXPECT_EQ(result2.block_ids, std::vector<int>({1, 2}));
    EXPECT_EQ(result2.losses.size(), 8);  // 2个block * 4个loss

    // 无效匹配
    auto result3 = cache_->match({105, 106, 101, 102, 103, 104});
    EXPECT_EQ(result3.matched_len, 0);
    EXPECT_TRUE(result3.block_ids.empty());
    EXPECT_TRUE(result3.losses.empty());

    auto result4 = cache_->match({106, 101, 102, 104, 105});
    EXPECT_EQ(result4.matched_len, 0);
    EXPECT_TRUE(result4.block_ids.empty());
    EXPECT_TRUE(result4.losses.empty());

    // 空输入
    auto result5 = cache_->match({});
    EXPECT_EQ(result5.matched_len, 0);
    EXPECT_TRUE(result5.block_ids.empty());
    EXPECT_TRUE(result5.losses.empty());
}

TEST_F(BlockLRUCacheTest, PutBasicTest) {
    // 测试基本的put功能
    std::vector<int64_t> cache_keys = {101, 102, 103};
    std::vector<int>     block_ids  = {1, 2, 3};
    std::vector<float>   losses     = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};

    // 第一轮 put，新插入的项应该都被缓存
    auto not_cached = cache_->put(cache_keys, block_ids, losses, false);
    EXPECT_TRUE(not_cached.empty());  // 新插入的项应该都被缓存
    EXPECT_EQ(cache_->size(), 3);
    EXPECT_EQ(cache_->availableBlockNum(), 3);

    // reuse cache
    auto result1 = cache_->match({101, 102, 103, 104});
    EXPECT_EQ(result1.matched_len, 3);
    EXPECT_EQ(result1.block_ids, std::vector<int>({1, 2, 3}));
    EXPECT_EQ(result1.losses.size(), 12);  // 3个block * 4个loss
    EXPECT_EQ(cache_->availableBlockNum(), 3);

    cache_->incrBlockRefCounter({1, 2, 3});
    EXPECT_EQ(cache_->availableBlockNum(), 0);

    // 复用三个block
    cache_->incrBlockRefCounter({1, 2, 3});
    EXPECT_EQ(cache_->availableBlockNum(), 0);

    // 再次put
    auto not_cached2 = cache_->put({101, 102, 103, 104, 105},
                                   {1, 2, 3, 4, 5},
                                   {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f,
                                    1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f},
                                   false);
    EXPECT_TRUE(not_cached2.empty());
    cache_->decrBlockRefCounter({1, 2, 3});  // 减少引用计数
    EXPECT_EQ(cache_->availableBlockNum(), 2);

    // 再次put，应该减少引用计数
    auto not_cached3 = cache_->put({101, 102, 103, 106, 107},
                                   {1, 2, 3, 6, 7},
                                   {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f,
                                    1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f},
                                   false);
    EXPECT_TRUE(not_cached3.empty());
    cache_->decrBlockRefCounter({1, 2, 3});  // 减少引用计数
    EXPECT_EQ(cache_->availableBlockNum(), 7);
}

TEST_F(BlockLRUCacheTest, PopBasicTest) {
    // 测试基本的pop功能
    std::vector<int64_t> cache_keys = {101, 102, 103, 104, 105};
    std::vector<int>     block_ids  = {1, 2, 3, 4, 5};
    std::vector<float>   losses     = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f,
                                       1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f};

    auto not_cached = cache_->put(cache_keys, block_ids, losses, false);
    EXPECT_TRUE(not_cached.empty());
    EXPECT_EQ(cache_->size(), 5);
    EXPECT_EQ(cache_->availableBlockNum(), 5);

    // 弹出2个项
    auto popped = cache_->pop(2);
    EXPECT_EQ(popped.size(), 2);
    EXPECT_EQ(cache_->size(), 3);
    EXPECT_EQ(cache_->availableBlockNum(), 3);

    // 验证弹出的是输入尾部的block_id（因为从最后向前push）
    EXPECT_EQ(popped[0], 5);
    EXPECT_EQ(popped[1], 4);

    //  有引用计数的不会被pop
    auto result = cache_->match({101, 102, 103, 104, 105});
    EXPECT_EQ(result.matched_len, 3);
    cache_->incrBlockRefCounter({1, 2, 3});
    EXPECT_EQ(cache_->availableBlockNum(), 0);

    auto popped2 = cache_->pop(2);
    EXPECT_EQ(popped2.size(), 0);
    EXPECT_EQ(cache_->availableBlockNum(), 0);

    // 重新put, 又可以pop了
    auto not_cached5 = cache_->put(cache_keys, block_ids, losses, false);
    EXPECT_TRUE(not_cached5.empty());
    cache_->decrBlockRefCounter({1, 2, 3});
    EXPECT_EQ(cache_->availableBlockNum(), 5);

    auto popped3 = cache_->pop(2);
    EXPECT_EQ(popped3.size(), 2);
    EXPECT_EQ(cache_->availableBlockNum(), 3);
    EXPECT_EQ(popped3, std::vector<int>({5, 4}));

    // 设置resident
    auto not_cached6 = cache_->put(cache_keys, block_ids, losses, true);
    EXPECT_TRUE(not_cached6.empty());
    EXPECT_EQ(cache_->availableBlockNum(), 5);

    // resident项不会被pop
    auto popped4 = cache_->pop(1);
    EXPECT_EQ(popped4.size(), 0);
    EXPECT_EQ(cache_->availableBlockNum(), 5);
}

// ==================== Put方法测试 ====================

TEST_F(BlockLRUCacheTest, PutWithEmptyLossesTest) {
    // 测试put时losses为空的情况
    std::vector<int64_t> cache_keys = {101, 102, 103};
    std::vector<int>     block_ids  = {1, 2, 3};
    std::vector<float>   losses;  // 空的losses

    auto not_cached = cache_->put(cache_keys, block_ids, losses, false);
    EXPECT_TRUE(not_cached.empty());  // 新插入的项应该都被缓存
    EXPECT_EQ(cache_->size(), 3);
    EXPECT_EQ(cache_->availableBlockNum(), 3);
}

TEST_F(BlockLRUCacheTest, PutWithPartialLossesTest) {
    // 测试losses数量不足的情况
    std::vector<int64_t> cache_keys = {101, 102, 103};
    std::vector<int>     block_ids  = {1, 2, 3};
    std::vector<float>   losses     = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};  // 只有5个loss，不够3个block*4

    auto not_cached = cache_->put(cache_keys, block_ids, losses, false);
    EXPECT_TRUE(not_cached.empty());  // 新插入的项应该都被缓存
    EXPECT_EQ(cache_->size(), 3);
    EXPECT_EQ(cache_->availableBlockNum(), 3);
}

TEST_F(BlockLRUCacheTest, PutWithInvalidInputTest) {

    // 测试无效输入
    std::vector<int64_t> cache_keys = {101, 102, 103};
    std::vector<int>     block_ids  = {1, 2};  // 长度不匹配
    std::vector<float>   losses     = {0.1f, 0.2f, 0.3f, 0.4f};

    auto not_cached = cache_->put(cache_keys, block_ids, losses, false);
    EXPECT_TRUE(not_cached.empty());  // 无效输入时返回空向量
    EXPECT_EQ(cache_->size(), 0);
}

TEST_F(BlockLRUCacheTest, PutDuplicateKeyTest) {
    // 测试重复key的处理
    std::vector<int64_t> cache_keys = {101, 102, 103};
    std::vector<int>     block_ids  = {1, 2, 3};
    std::vector<float>   losses     = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};

    // 第一次插入
    auto not_cached1 = cache_->put(cache_keys, block_ids, losses, false);
    EXPECT_TRUE(not_cached1.empty());
    EXPECT_EQ(cache_->size(), 3);
    EXPECT_EQ(cache_->availableBlockNum(), 3);

    auto result1 = cache_->match({101, 102, 103});
    EXPECT_EQ(result1.matched_len, 3);
    EXPECT_EQ(result1.block_ids, std::vector<int>({1, 2, 3}));
    EXPECT_EQ(result1.losses.size(), 12);  // 3个block * 4个loss

    // 第二次插入相同的key，应该更新热度
    std::vector<int> new_block_ids = {4, 5, 6};
    auto             not_cached2   = cache_->put(cache_keys, new_block_ids, losses, false);
    EXPECT_EQ(not_cached2.size(), 3);  // 返回旧的block_ids
    EXPECT_EQ(not_cached2, std::vector<int>({4, 5, 6}));
    EXPECT_EQ(cache_->size(), 3);  // 大小不变
    EXPECT_EQ(cache_->availableBlockNum(), 3);
}

// ==================== 错误处理测试 ====================

TEST_F(BlockLRUCacheTest, ErrorHandlingTest) {
    // 测试错误处理
    std::vector<int64_t> cache_keys = {101, 102, 103};
    std::vector<int32_t> block_ids  = {1, 2};  // 长度不匹配
    std::vector<float>   losses     = {0.1f, 0.2f, 0.3f, 0.4f};

    auto not_cached = cache_->put(cache_keys, block_ids, losses, false);
    EXPECT_TRUE(not_cached.empty());
    EXPECT_EQ(cache_->size(), 0);

    // 测试空输入
    auto not_cached2 = cache_->put({}, {}, {}, false);
    EXPECT_TRUE(not_cached2.empty());
    EXPECT_EQ(cache_->size(), 0);

    auto not_cached3 = cache_->put({1}, {}, {0.1f}, false);
    EXPECT_TRUE(not_cached3.empty());
    EXPECT_EQ(cache_->size(), 0);
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
