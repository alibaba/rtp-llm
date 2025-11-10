#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <chrono>
#include "rtp_llm/cpp/cache_new/p2p_connector/LayerCacheBuffer.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace test {

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

    int64_t getCurrentTimeUs() {
        return currentTimeUs();
    }

    int64_t getFutureTimeUs(int64_t delay_ms) {
        return getCurrentTimeUs() + delay_ms * 1000;
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
    int64_t deadline_ms = 1000;
    int64_t deadline_us = getFutureTimeUs(deadline_ms);

    bool result = store_->setLayerCacheBuffer(buffer, deadline_us);
    EXPECT_TRUE(result);
    EXPECT_EQ(1, store_->layerCacheBufferMapSize());
}

TEST_F(SingleLayerCacheBufferStoreTest, SetLayerCacheBufferMultipleWatchers) {
    // 测试多个观察者的情况
    auto    watcher1            = std::make_shared<MockWatcher>(layer_id_, false);
    auto    watcher2            = std::make_shared<MockWatcher>(layer_id_, false);
    int64_t watcher_deadline_us = getFutureTimeUs(5000);

    store_->setLayerCacheBufferWatchFunc(watcher1, watcher_deadline_us);
    store_->setLayerCacheBufferWatchFunc(watcher2, watcher_deadline_us);
    EXPECT_EQ(2, store_->layerCacheBufferWatcherMapSize());

    auto buffer1 = createLayerCacheBuffer(layer_id_);
    auto buffer2 = createLayerCacheBuffer(layer_id_);
    auto buffer3 = createLayerCacheBuffer(layer_id_);

    int64_t buffer_deadline_us = getFutureTimeUs(1000);
    store_->setLayerCacheBuffer(buffer1, buffer_deadline_us);
    EXPECT_EQ(watcher1->getNotifyCount(), 1);
    EXPECT_EQ(watcher2->getNotifyCount(), 1);
    EXPECT_EQ(1, store_->layerCacheBufferMapSize());

    store_->setLayerCacheBuffer(buffer2, getFutureTimeUs(2000));
    EXPECT_EQ(watcher1->getNotifyCount(), 2);
    EXPECT_EQ(watcher2->getNotifyCount(), 2);

    store_->setLayerCacheBuffer(buffer3, getFutureTimeUs(3000));
    EXPECT_EQ(watcher1->getNotifyCount(), 3);
    EXPECT_EQ(watcher2->getNotifyCount(), 3);

    EXPECT_EQ(3, store_->layerCacheBufferMapSize());
    EXPECT_EQ(2, store_->layerCacheBufferWatcherMapSize());
}

TEST_F(SingleLayerCacheBufferStoreTest, SetLayerCacheBufferWithConsumingWatcher) {
    // 测试观察者消费缓存缓冲区的情况
    auto    watcher             = std::make_shared<MockWatcher>(layer_id_, true);  // should_consume = true
    int64_t watcher_deadline_us = getFutureTimeUs(5000);

    store_->setLayerCacheBufferWatchFunc(watcher, watcher_deadline_us);

    auto    buffer             = createLayerCacheBuffer(layer_id_);
    int64_t buffer_deadline_us = getFutureTimeUs(1000);
    store_->setLayerCacheBuffer(buffer, buffer_deadline_us);

    // 观察者应该被通知
    EXPECT_EQ(watcher->getNotifyCount(), 1);
    EXPECT_EQ(0, store_->layerCacheBufferMapSize());
    EXPECT_EQ(0, store_->layerCacheBufferWatcherMapSize());
}

// ==================== setLayerCacheBufferWatchFunc 测试 ====================

TEST_F(SingleLayerCacheBufferStoreTest, SetLayerCacheBufferWatchFuncBasicTest) {
    // 测试基本的设置观察者功能
    auto    watcher     = std::make_shared<MockWatcher>(layer_id_, false);
    int64_t deadline_us = getFutureTimeUs(1000);

    store_->setLayerCacheBufferWatchFunc(watcher, deadline_us);
    // 如果没有现有缓冲区，观察者不应该被通知
    EXPECT_EQ(watcher->getNotifyCount(), 0);
}

TEST_F(SingleLayerCacheBufferStoreTest, SetLayerCacheBufferWatchFuncWithExistingBuffers) {
    // 测试设置观察者时检查现有缓存缓冲区
    auto buffer1 = createLayerCacheBuffer(layer_id_);
    auto buffer2 = createLayerCacheBuffer(layer_id_);
    auto buffer3 = createLayerCacheBuffer(layer_id_);

    // 先设置一些缓存缓冲区
    store_->setLayerCacheBuffer(buffer1, getFutureTimeUs(1000));
    store_->setLayerCacheBuffer(buffer2, getFutureTimeUs(2000));
    store_->setLayerCacheBuffer(buffer3, getFutureTimeUs(3000));

    auto watcher1 = std::make_shared<MockWatcher>(layer_id_, false);
    auto watcher2 = std::make_shared<MockWatcher>(layer_id_, false);
    auto watcher3 = std::make_shared<MockWatcher>(layer_id_, false);

    store_->setLayerCacheBufferWatchFunc(watcher1, getFutureTimeUs(5000));
    store_->setLayerCacheBufferWatchFunc(watcher2, getFutureTimeUs(5000));
    store_->setLayerCacheBufferWatchFunc(watcher3, getFutureTimeUs(5000));

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

    store_->setLayerCacheBuffer(buffer1, getFutureTimeUs(1000));
    store_->setLayerCacheBuffer(buffer2, getFutureTimeUs(2000));

    // 设置一个会消费缓冲区的观察者
    auto watcher = std::make_shared<MockWatcher>(layer_id_, true);
    store_->setLayerCacheBufferWatchFunc(watcher, getFutureTimeUs(5000));

    EXPECT_EQ(watcher->getNotifyCount(), 1);
    EXPECT_EQ(watcher->getLastNotifiedBuffer(), buffer1);
    EXPECT_EQ(1, store_->layerCacheBufferMapSize());
    EXPECT_EQ(0, store_->layerCacheBufferWatcherMapSize());
}

// ==================== checkTimeout 测试 ====================

TEST_F(SingleLayerCacheBufferStoreTest, CheckTimeoutNoExpiredBuffers) {
    // 测试没有过期缓冲区的情况
    auto    buffer1            = createLayerCacheBuffer(layer_id_);
    int64_t future_deadline_us = getFutureTimeUs(1000);
    store_->setLayerCacheBuffer(buffer1, future_deadline_us);

    auto    buffer2          = createLayerCacheBuffer(layer_id_);
    int64_t past_deadline_us = getCurrentTimeUs() - 1000;  // 1秒前过期
    store_->setLayerCacheBuffer(buffer2, past_deadline_us);

    EXPECT_EQ(2, store_->layerCacheBufferMapSize());

    auto watcher1 = std::make_shared<MockWatcher>(layer_id_, false);
    store_->setLayerCacheBufferWatchFunc(watcher1, getFutureTimeUs(5000));
    auto watcher2 = std::make_shared<MockWatcher>(layer_id_, false);
    store_->setLayerCacheBufferWatchFunc(watcher2, getFutureTimeUs(-5000));
    EXPECT_EQ(2, store_->layerCacheBufferWatcherMapSize());

    store_->checkTimeout();
    EXPECT_EQ(1, store_->layerCacheBufferMapSize());
    EXPECT_EQ(1, store_->layerCacheBufferWatcherMapSize());
}

}  // namespace test
}  // namespace rtp_llm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
