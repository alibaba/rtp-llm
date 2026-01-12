#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <chrono>
#include <atomic>

#include "rtp_llm/cpp/disaggregate/p2p_connector/StoreWaitContext.h"
#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

// Mock DeviceEvent for testing
class MockDeviceEvent: public DeviceEvent {
public:
    MockDeviceEvent(bool ready = false): ready_(ready) {}
    ~MockDeviceEvent() override = default;

    void synchronize() const override {
        // Do nothing in mock
    }

    bool checkReadiness() const override {
        return ready_.load();
    }

    void setReady(bool ready) {
        ready_.store(ready);
    }

private:
    mutable std::atomic<bool> ready_;
};

class StoreWaitContextTest: public ::testing::Test {
protected:
    void SetUp() override {
        computed_buffers_ = std::make_shared<ComputedLayerCacheBufferStore>();
    }

    void TearDown() override {
        computed_buffers_.reset();
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
    std::shared_ptr<ComputedLayerCacheBufferStore> computed_buffers_;
};

TEST_F(StoreWaitContextTest, CheckerCheckOnce_EventReady) {
    StoreWaitContextChecker checker(nullptr, computed_buffers_);

    int64_t request_id  = 3001;
    auto    event       = std::make_shared<MockDeviceEvent>(true);  // Event is ready
    auto    buffer      = createLayerCacheBuffer(0);
    int64_t deadline_ms = getDeadlineMs();
    auto    collector   = std::make_shared<P2PConnectorServerWorkerStoreMetricsCollector>();

    StoreWaitContext context(request_id, event, buffer, deadline_ms, collector);
    checker.addContext(context);

    EXPECT_EQ(checker.getContextCount(), 1);

    // 调用 checkOnce，由于 event 已就绪，应该被移除
    checker.checkOnce();

    EXPECT_EQ(checker.getContextCount(), 0);

    // 验证 buffer 被添加到 computed_buffers_
    auto computed_buffer = computed_buffers_->getBuffer(request_id);
    ASSERT_NE(computed_buffer, nullptr);
}

TEST_F(StoreWaitContextTest, CheckerCheckOnce_EventNotReady) {
    StoreWaitContextChecker checker(nullptr, computed_buffers_);

    int64_t request_id  = 3002;
    auto    event       = std::make_shared<MockDeviceEvent>(false);  // Event is not ready
    auto    buffer      = createLayerCacheBuffer(0);
    int64_t deadline_ms = getDeadlineMs();
    auto    collector   = std::make_shared<P2PConnectorServerWorkerStoreMetricsCollector>();

    StoreWaitContext context(request_id, event, buffer, deadline_ms, collector);
    checker.addContext(context);

    EXPECT_EQ(checker.getContextCount(), 1);

    // 调用 checkOnce，由于 event 未就绪，应该保留
    checker.checkOnce();

    EXPECT_EQ(checker.getContextCount(), 1);

    // 验证 buffer 没有被添加到 computed_buffers_
    auto computed_buffer = computed_buffers_->getBuffer(request_id);
    EXPECT_EQ(computed_buffer, nullptr);

    event->setReady(true);

    // 调用 checkOnce，由于 event 已就绪，应该被移除
    checker.checkOnce();

    EXPECT_EQ(checker.getContextCount(), 0);

    // 验证 buffer 被添加到 computed_buffers_
    computed_buffer = computed_buffers_->getBuffer(request_id);
    ASSERT_NE(computed_buffer, nullptr);
}

TEST_F(StoreWaitContextTest, CheckerCheckOnce_NullEvent) {
    StoreWaitContextChecker checker(nullptr, computed_buffers_);

    int64_t request_id  = 3003;
    auto    buffer      = createLayerCacheBuffer(0);
    int64_t deadline_ms = getDeadlineMs();
    auto    collector   = std::make_shared<P2PConnectorServerWorkerStoreMetricsCollector>();

    // null event 被视为已就绪
    StoreWaitContext context(request_id, nullptr, buffer, deadline_ms, collector);
    checker.addContext(context);

    EXPECT_EQ(checker.getContextCount(), 1);

    // 调用 checkOnce，null event 应该被视为就绪
    checker.checkOnce();

    EXPECT_EQ(checker.getContextCount(), 0);

    // 验证 buffer 被添加到 computed_buffers_
    auto computed_buffer = computed_buffers_->getBuffer(request_id);
    ASSERT_NE(computed_buffer, nullptr);
}

TEST_F(StoreWaitContextTest, CheckerCheckOnce_Timeout) {
    StoreWaitContextChecker checker(nullptr, computed_buffers_);

    int64_t request_id  = 3004;
    auto    event       = std::make_shared<MockDeviceEvent>(false);  // Event is not ready
    auto    buffer      = createLayerCacheBuffer(0);
    int64_t deadline_ms = currentTimeMs() - 100;  // 已过期
    auto    collector   = std::make_shared<P2PConnectorServerWorkerStoreMetricsCollector>();

    StoreWaitContext context(request_id, event, buffer, deadline_ms, collector);
    checker.addContext(context);

    EXPECT_EQ(checker.getContextCount(), 1);

    // 调用 checkOnce，由于已超时，应该被移除
    checker.checkOnce();

    EXPECT_EQ(checker.getContextCount(), 0);

    // 验证 buffer 没有被添加到 computed_buffers_（因为是超时，不是成功完成）
    auto computed_buffer = computed_buffers_->getBuffer(request_id);
    EXPECT_EQ(computed_buffer, nullptr);
}

}  // namespace rtp_llm
