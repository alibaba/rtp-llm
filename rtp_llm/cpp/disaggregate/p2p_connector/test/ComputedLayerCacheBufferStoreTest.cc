#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <chrono>

#include "rtp_llm/cpp/disaggregate/p2p_connector/ComputedLayerCacheBuffer.h"
#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

class ComputedLayerCacheBufferStoreTest: public ::testing::Test {
protected:
    void SetUp() override {
        store_ = std::make_unique<ComputedLayerCacheBufferStore>();
    }

    void TearDown() override {
        store_.reset();
    }

    // 创建测试用的 LayerCacheBuffer
    std::shared_ptr<LayerCacheBuffer> createLayerCacheBuffer(int layer_id) {
        return std::make_shared<LayerCacheBuffer>(layer_id);
    }

    // 获取当前时间（毫秒）+ 偏移量
    int64_t getDeadlineMs(int64_t offset_ms = 1000) {
        return currentTimeMs() + offset_ms;
    }

protected:
    std::unique_ptr<ComputedLayerCacheBufferStore> store_;
};

// ==================== 基础功能测试 ====================

// 测试 addBuffer 和 getBuffer - 基本功能
TEST_F(ComputedLayerCacheBufferStoreTest, AddAndGetBuffer) {
    int64_t request_id  = 1001;
    auto    buffer      = createLayerCacheBuffer(0);
    int64_t deadline_ms = getDeadlineMs();

    store_->addBuffer(request_id, buffer, deadline_ms);

    auto retrieved = store_->getBuffer(request_id);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->request_id, request_id);
    EXPECT_EQ(retrieved->deadline_ms, deadline_ms);
    EXPECT_EQ(retrieved->layer_cache_buffers.size(), 1);
    EXPECT_NE(retrieved->layer_cache_buffers.find(0), retrieved->layer_cache_buffers.end());
    EXPECT_EQ(retrieved->layer_cache_buffers[0], buffer);
    EXPECT_EQ(retrieved->layer_cache_buffers[0]->getLayerId(), 0);

    // test add buffer with same request_id
    auto buffer1 = createLayerCacheBuffer(1);
    store_->addBuffer(request_id, buffer1, deadline_ms);
    retrieved = store_->getBuffer(request_id);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->request_id, request_id);
    EXPECT_EQ(retrieved->deadline_ms, deadline_ms);
    EXPECT_EQ(retrieved->layer_cache_buffers.size(), 2);
    EXPECT_NE(retrieved->layer_cache_buffers.find(0), retrieved->layer_cache_buffers.end());
    EXPECT_EQ(retrieved->layer_cache_buffers[0], buffer);
    EXPECT_EQ(retrieved->layer_cache_buffers[0]->getLayerId(), 0);
    EXPECT_NE(retrieved->layer_cache_buffers.find(1), retrieved->layer_cache_buffers.end());
    EXPECT_EQ(retrieved->layer_cache_buffers[1], buffer1);
    EXPECT_EQ(retrieved->layer_cache_buffers[1]->getLayerId(), 1);

    // test get non existent request_id
    auto retrieved2 = store_->getBuffer(9999);
    EXPECT_EQ(retrieved2, nullptr);

    // test remove buffer
    store_->removeBuffer(request_id);
    retrieved = store_->getBuffer(request_id);
    EXPECT_EQ(retrieved, nullptr);
}

TEST_F(ComputedLayerCacheBufferStoreTest, CheckTimeoutNotExpired) {
    int64_t request_id  = 5001;
    auto    buffer      = createLayerCacheBuffer(0);
    int64_t deadline_ms = getDeadlineMs(1000);  // 1秒后过期

    int64_t request_id1  = 5002;
    auto    buffer1      = createLayerCacheBuffer(0);
    int64_t deadline_ms1 = getDeadlineMs(-100);

    store_->addBuffer(request_id, buffer, deadline_ms);
    store_->addBuffer(request_id1, buffer1, deadline_ms1);

    // 立即检查超时，应该不会删除
    store_->checkTimeout();

    auto retrieved = store_->getBuffer(request_id);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->request_id, request_id);

    EXPECT_TRUE(store_->getBuffer(request_id1) == nullptr);
}

}  // namespace rtp_llm
