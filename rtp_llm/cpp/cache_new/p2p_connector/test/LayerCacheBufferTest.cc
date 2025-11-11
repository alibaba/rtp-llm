#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <chrono>
#include "rtp_llm/cpp/cache_new/p2p_connector/LayerCacheBuffer.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace cache_store {

// Mock Watcher 用于测试观察者功能
class MockWatcher: public SingleLayerCacheBufferStore::Watcher {
public:
    MockWatcher(int layer_id, bool should_consume = false):
        SingleLayerCacheBufferStore::Watcher(layer_id),
        should_consume_(should_consume),
        notify_count_(0),
        last_notified_buffer_(nullptr) {}

    bool notify(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer) override {
        notify_count_++;
        last_notified_buffer_ = layer_cache_buffer;
        return should_consume_;
    }

    int getNotifyCount() const {
        return notify_count_;
    }

    std::shared_ptr<LayerCacheBuffer> getLastNotifiedBuffer() const {
        return last_notified_buffer_;
    }

    void reset() {
        notify_count_         = 0;
        last_notified_buffer_ = nullptr;
    }

private:
    bool                              should_consume_;
    std::atomic<int>                  notify_count_;
    std::shared_ptr<LayerCacheBuffer> last_notified_buffer_;
};

class SingleLayerCacheBufferStoreTest: public ::testing::Test {
protected:
    void SetUp() override {
        layer_id_ = 0;
        store_    = std::make_unique<SingleLayerCacheBufferStore>(layer_id_);
    }

    void TearDown() override {
        store_.reset();
    }

    int64_t getCurrentTimeMs() {
        return currentTimeMs();
    }

    int64_t getFutureTimeMs(int64_t delay_ms) {
        return getCurrentTimeMs() + delay_ms;
    }

    std::shared_ptr<LayerCacheBuffer> createLayerCacheBuffer(int layer_id) {
        return std::make_shared<LayerCacheBuffer>(layer_id);
    }

    int                                          layer_id_;
    std::unique_ptr<SingleLayerCacheBufferStore> store_;
};

// ==================== setLayerCacheBuffer 测试 ====================

TEST_F(SingleLayerCacheBufferStoreTest, SetLayerCacheBufferBasicTest) {
    // 测试基本的设置缓存缓冲区功能
    auto    buffer      = createLayerCacheBuffer(layer_id_);
    int64_t deadline_ms = getFutureTimeMs(1000);

    bool result = store_->setLayerCacheBuffer(buffer, deadline_ms);
    EXPECT_TRUE(result);
    EXPECT_EQ(1, store_->layerCacheBufferMapSize());
}

TEST_F(SingleLayerCacheBufferStoreTest, SetLayerCacheBufferMultipleWatchers) {
    // 测试多个观察者的情况
    auto    watcher1            = std::make_shared<MockWatcher>(layer_id_, false);
    auto    watcher2            = std::make_shared<MockWatcher>(layer_id_, false);
    int64_t watcher_deadline_ms = getFutureTimeMs(5000);

    store_->setLayerCacheBufferWatchFunc(watcher1, watcher_deadline_ms);
    store_->setLayerCacheBufferWatchFunc(watcher2, watcher_deadline_ms);
    EXPECT_EQ(2, store_->layerCacheBufferWatcherMapSize());

    auto buffer1 = createLayerCacheBuffer(layer_id_);
    auto buffer2 = createLayerCacheBuffer(layer_id_);
    auto buffer3 = createLayerCacheBuffer(layer_id_);

    int64_t buffer_deadline_ms = getFutureTimeMs(1000);
    store_->setLayerCacheBuffer(buffer1, buffer_deadline_ms);
    EXPECT_EQ(watcher1->getNotifyCount(), 1);
    EXPECT_EQ(watcher2->getNotifyCount(), 1);
    EXPECT_EQ(1, store_->layerCacheBufferMapSize());

    store_->setLayerCacheBuffer(buffer2, getFutureTimeMs(2000));
    EXPECT_EQ(watcher1->getNotifyCount(), 2);
    EXPECT_EQ(watcher2->getNotifyCount(), 2);

    store_->setLayerCacheBuffer(buffer3, getFutureTimeMs(3000));
    EXPECT_EQ(watcher1->getNotifyCount(), 3);
    EXPECT_EQ(watcher2->getNotifyCount(), 3);

    EXPECT_EQ(3, store_->layerCacheBufferMapSize());
    EXPECT_EQ(2, store_->layerCacheBufferWatcherMapSize());
}

TEST_F(SingleLayerCacheBufferStoreTest, SetLayerCacheBufferWithConsumingWatcher) {
    // 测试观察者消费缓存缓冲区的情况
    auto    watcher             = std::make_shared<MockWatcher>(layer_id_, true);  // should_consume = true
    int64_t watcher_deadline_ms = getFutureTimeMs(5000);

    store_->setLayerCacheBufferWatchFunc(watcher, watcher_deadline_ms);

    auto    buffer             = createLayerCacheBuffer(layer_id_);
    int64_t buffer_deadline_ms = getFutureTimeMs(1000);
    store_->setLayerCacheBuffer(buffer, buffer_deadline_ms);

    // 观察者应该被通知
    EXPECT_EQ(watcher->getNotifyCount(), 1);
    EXPECT_EQ(0, store_->layerCacheBufferMapSize());
    EXPECT_EQ(0, store_->layerCacheBufferWatcherMapSize());
}

// ==================== setLayerCacheBufferWatchFunc 测试 ====================

TEST_F(SingleLayerCacheBufferStoreTest, SetLayerCacheBufferWatchFuncBasicTest) {
    // 测试基本的设置观察者功能
    auto    watcher     = std::make_shared<MockWatcher>(layer_id_, false);
    int64_t deadline_ms = getFutureTimeMs(1000);

    store_->setLayerCacheBufferWatchFunc(watcher, deadline_ms);
    // 如果没有现有缓冲区，观察者不应该被通知
    EXPECT_EQ(watcher->getNotifyCount(), 0);
}

TEST_F(SingleLayerCacheBufferStoreTest, SetLayerCacheBufferWatchFuncWithExistingBuffers) {
    // 测试设置观察者时检查现有缓存缓冲区
    auto buffer1 = createLayerCacheBuffer(layer_id_);
    auto buffer2 = createLayerCacheBuffer(layer_id_);
    auto buffer3 = createLayerCacheBuffer(layer_id_);

    // 先设置一些缓存缓冲区
    store_->setLayerCacheBuffer(buffer1, getFutureTimeMs(1000));
    store_->setLayerCacheBuffer(buffer2, getFutureTimeMs(2000));
    store_->setLayerCacheBuffer(buffer3, getFutureTimeMs(3000));

    auto watcher1 = std::make_shared<MockWatcher>(layer_id_, false);
    auto watcher2 = std::make_shared<MockWatcher>(layer_id_, false);
    auto watcher3 = std::make_shared<MockWatcher>(layer_id_, false);

    store_->setLayerCacheBufferWatchFunc(watcher1, getFutureTimeMs(5000));
    store_->setLayerCacheBufferWatchFunc(watcher2, getFutureTimeMs(5000));
    store_->setLayerCacheBufferWatchFunc(watcher3, getFutureTimeMs(5000));

    // 所有观察者都应该被通知
    EXPECT_GE(watcher1->getNotifyCount(), 1);
    EXPECT_GE(watcher2->getNotifyCount(), 1);
    EXPECT_GE(watcher3->getNotifyCount(), 1);

    EXPECT_EQ(3, store_->layerCacheBufferMapSize());
    EXPECT_EQ(3, store_->layerCacheBufferWatcherMapSize());
}

TEST_F(SingleLayerCacheBufferStoreTest, SetLayerCacheBufferWatchFuncConsumingFirstBuffer) {
    // 测试观察者消费第一个缓冲区的情况
    auto buffer1 = createLayerCacheBuffer(layer_id_);
    auto buffer2 = createLayerCacheBuffer(layer_id_);

    store_->setLayerCacheBuffer(buffer1, getFutureTimeMs(1000));
    store_->setLayerCacheBuffer(buffer2, getFutureTimeMs(2000));

    // 设置一个会消费缓冲区的观察者
    auto watcher = std::make_shared<MockWatcher>(layer_id_, true);
    store_->setLayerCacheBufferWatchFunc(watcher, getFutureTimeMs(5000));

    EXPECT_EQ(watcher->getNotifyCount(), 1);
    EXPECT_EQ(watcher->getLastNotifiedBuffer(), buffer1);
    EXPECT_EQ(1, store_->layerCacheBufferMapSize());
    EXPECT_EQ(0, store_->layerCacheBufferWatcherMapSize());
}

// ==================== checkTimeout 测试 ====================

TEST_F(SingleLayerCacheBufferStoreTest, CheckTimeoutNoExpiredBuffers) {
    // 测试没有过期缓冲区的情况
    auto    buffer1            = createLayerCacheBuffer(layer_id_);
    int64_t future_deadline_ms = getFutureTimeMs(1000);
    store_->setLayerCacheBuffer(buffer1, future_deadline_ms);

    auto    buffer2          = createLayerCacheBuffer(layer_id_);
    int64_t past_deadline_ms = getCurrentTimeMs() - 1000;  // 1秒前过期
    store_->setLayerCacheBuffer(buffer2, past_deadline_ms);

    EXPECT_EQ(2, store_->layerCacheBufferMapSize());

    auto watcher1 = std::make_shared<MockWatcher>(layer_id_, false);
    store_->setLayerCacheBufferWatchFunc(watcher1, getFutureTimeMs(5000));
    auto watcher2 = std::make_shared<MockWatcher>(layer_id_, false);
    store_->setLayerCacheBufferWatchFunc(watcher2, getFutureTimeMs(-5000));
    EXPECT_EQ(2, store_->layerCacheBufferWatcherMapSize());

    store_->checkTimeout();
    EXPECT_EQ(1, store_->layerCacheBufferMapSize());
    EXPECT_EQ(1, store_->layerCacheBufferWatcherMapSize());
}

// ==================== LayerCacheBufferStore 测试 ====================

class LayerCacheBufferStoreTest: public ::testing::Test {
protected:
    void SetUp() override {
        layer_num_ = 3;
        store_     = std::make_unique<LayerCacheBufferStore>(layer_num_);
    }

    void TearDown() override {
        // LayerCacheBufferStore 析构时会自动停止线程
        store_.reset();
    }

    int64_t getCurrentTimeMs() {
        return currentTimeMs();
    }

    int64_t getFutureTimeMs(int64_t delay_ms) {
        return getCurrentTimeMs() + delay_ms;
    }

    int64_t getPastTimeMs(int64_t delay_ms) {
        return getCurrentTimeMs() - delay_ms;
    }

    std::shared_ptr<LayerCacheBuffer> createLayerCacheBuffer(int layer_id) {
        return std::make_shared<LayerCacheBuffer>(layer_id);
    }

    int                                    layer_num_;
    std::unique_ptr<LayerCacheBufferStore> store_;
};

TEST_F(LayerCacheBufferStoreTest, CheckTimeoutThreadMixedExpiredAndValid) {
    // 测试混合过期和有效 buffer/watcher 的情况
    for (int i = 0; i < layer_num_; ++i) {
        auto layer_store = store_->getSingleLayerCacheBufferStore(i);
        ASSERT_NE(layer_store, nullptr);

        // 设置过期的 buffer
        auto    expired_buffer   = createLayerCacheBuffer(i);
        int64_t past_deadline_ms = getPastTimeMs(100);
        layer_store->setLayerCacheBuffer(expired_buffer, past_deadline_ms);

        // 设置有效的 buffer
        auto    valid_buffer       = createLayerCacheBuffer(i);
        int64_t future_deadline_ms = getFutureTimeMs(1000);
        layer_store->setLayerCacheBuffer(valid_buffer, future_deadline_ms);

        // 设置过期的 watcher
        auto    expired_watcher          = std::make_shared<MockWatcher>(i, false);
        int64_t past_watcher_deadline_ms = getPastTimeMs(100);
        layer_store->setLayerCacheBufferWatchFunc(expired_watcher, past_watcher_deadline_ms);

        // 设置有效的 watcher
        auto    valid_watcher              = std::make_shared<MockWatcher>(i, false);
        int64_t future_watcher_deadline_ms = getFutureTimeMs(1000);
        layer_store->setLayerCacheBufferWatchFunc(valid_watcher, future_watcher_deadline_ms);
    }

    // 等待 checkTimeoutThread 处理
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // 验证只有有效的 buffer 和 watcher 保留
    for (int i = 0; i < layer_num_; ++i) {
        auto layer_store = store_->getSingleLayerCacheBufferStore(i);
        EXPECT_EQ(1, layer_store->layerCacheBufferMapSize());
        EXPECT_EQ(1, layer_store->layerCacheBufferWatcherMapSize());
    }
}

}  // namespace cache_store
}  // namespace rtp_llm
