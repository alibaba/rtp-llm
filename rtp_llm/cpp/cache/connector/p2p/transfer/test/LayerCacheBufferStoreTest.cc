#include <gtest/gtest.h>
#include <memory>
#include <map>
#include <string>
#include <thread>
#include <chrono>
#include <vector>

#include "rtp_llm/cpp/cache/connector/p2p/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

class LayerCacheBufferStoreTest: public ::testing::Test {
protected:
    void SetUp() override {
        store_ = std::make_unique<LayerCacheBufferStore>();
    }

    void TearDown() override {
        store_.reset();
    }

    // 创建测试用的 LayerCacheBuffer
    std::shared_ptr<LayerCacheBuffer> createLayerCacheBuffer(int layer_id) {
        return std::make_shared<LayerCacheBuffer>(layer_id);
    }

protected:
    std::unique_ptr<LayerCacheBufferStore> store_;
};

// ==================== 基础功能测试 ====================

// 测试 addLayerCacheBuffer
TEST_F(LayerCacheBufferStoreTest, AddLayerCacheBufferTest) {
    std::string unique_key = "test_key_1";
    auto        buffer     = createLayerCacheBuffer(0);

    store_->addLayerCacheBuffer(unique_key, buffer);

    // 验证可以获取
    auto retrieved = store_->getLayerCacheBuffer(unique_key, 0);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->getLayerId(), 0);
    EXPECT_EQ(retrieved, buffer);

    auto no_exist_layer_retrieved = store_->getLayerCacheBuffer(unique_key, 1);
    EXPECT_EQ(no_exist_layer_retrieved, nullptr);

    auto no_exist_unique_key_retrieved = store_->getLayerCacheBuffer("no_exist_unique_key", 0);
    EXPECT_EQ(no_exist_unique_key_retrieved, nullptr);

    // test add new layer_id
    auto buffer1 = createLayerCacheBuffer(1);
    store_->addLayerCacheBuffer(unique_key, buffer1);
    auto new_layer_id_retrieved = store_->getLayerCacheBuffer(unique_key, 1);
    EXPECT_EQ(new_layer_id_retrieved, buffer1);
}

// ==================== 超时测试 ====================

// 测试 checkTimeout - 未过期
TEST_F(LayerCacheBufferStoreTest, CheckTimeoutNotExpiredTest) {
    std::string unique_key = "test_key_6";
    auto        buffer     = createLayerCacheBuffer(0);

    auto store = std::make_unique<LayerCacheBufferStore>(100);  // 100ms timeout
    store->addLayerCacheBuffer(unique_key, buffer);

    // 立即检查超时，应该不会清理（过期时间是100秒后）
    store->checkTimeout();

    // 验证仍然可以获取
    auto retrieved = store->getLayerCacheBuffer(unique_key, 0);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->getLayerId(), 0);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    store->checkTimeout();
    retrieved = store->getLayerCacheBuffer(unique_key, 0);
    EXPECT_EQ(retrieved, nullptr);
}

}  // namespace rtp_llm
