#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <chrono>
#include "rtp_llm/cpp/cache/connector/p2p/StoreWaitContext.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBuffer.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

class StoreWaitContextTest: public ::testing::Test {
protected:
    void SetUp() override {
        computed_buffers_ = std::make_shared<ComputedLayerCacheBufferStore>();
    }

    void TearDown() override {
        computed_buffers_.reset();
    }

    std::shared_ptr<LayerCacheBuffer> createLayerCacheBuffer(int layer_id) {
        return std::make_shared<LayerCacheBuffer>(layer_id, "full");
    }

    int64_t getDeadlineMs(int64_t offset_ms = 1000) {
        return currentTimeMs() + offset_ms;
    }

protected:
    std::shared_ptr<ComputedLayerCacheBufferStore> computed_buffers_;
};

TEST_F(StoreWaitContextTest, ReadyBufferPublishedBeforeAddContextReturns) {
    StoreWaitContextChecker checker(nullptr, computed_buffers_);

    int64_t request_id  = 3001;
    auto    buffer      = createLayerCacheBuffer(0);
    int64_t deadline_ms = getDeadlineMs();
    auto    collector   = std::make_shared<P2PConnectorMetricsCollector>();

    // null event is treated as ready
    StoreWaitContext context(request_id, nullptr, buffer, deadline_ms, collector);
    computed_buffers_->registerRequestHorizon(request_id, deadline_ms, deadline_ms);
    checker.addContext(std::move(context));

    // Fast path publishes/rejects before any background tick.
    EXPECT_EQ(checker.getContextCount(), 0);

    auto computed_buffer = computed_buffers_->getBuffer(request_id, deadline_ms);
    ASSERT_NE(computed_buffer, nullptr);
}

#if USING_CUDA
TEST_F(StoreWaitContextTest, ReadyGpuEventPublishesWithoutBackgroundCheck) {
    StoreWaitContextChecker checker(nullptr, computed_buffers_);
    auto                    event = std::make_shared<torch::Event>(torch::kCUDA);
    ASSERT_TRUE(event->query());  // An unrecorded event is already complete.
    const auto deadline = getDeadlineMs();
    ASSERT_TRUE(computed_buffers_->registerRequestHorizon(3002, deadline, deadline));
    checker.addContext(StoreWaitContext(3002, event, createLayerCacheBuffer(0), deadline, nullptr));
    EXPECT_EQ(checker.getContextCount(), 0);
    EXPECT_NE(computed_buffers_->getBuffer(3002, deadline), nullptr);
}
#endif

TEST_F(StoreWaitContextTest, CheckerCheckOnce_Timeout) {
    StoreWaitContextChecker checker(nullptr, computed_buffers_);

    int64_t request_id  = 3004;
    auto    buffer      = createLayerCacheBuffer(0);
    int64_t deadline_ms = currentTimeMs() - 100;  // already expired
    auto    collector   = std::make_shared<P2PConnectorMetricsCollector>();

    // Even with null event, timeout takes precedence
    StoreWaitContext context(request_id, nullptr, buffer, deadline_ms, collector);
    computed_buffers_->registerRequestHorizon(request_id, deadline_ms, deadline_ms);
    checker.addContext(std::move(context));

    // Fast path publishes/rejects before any background tick.
    EXPECT_EQ(checker.getContextCount(), 0);

    // Buffer should NOT be added (timeout, not success)
    auto computed_buffer = computed_buffers_->getBuffer(request_id, deadline_ms);
    EXPECT_EQ(computed_buffer, nullptr);
}

TEST_F(StoreWaitContextTest, CheckerUsesActivatedTransferHorizon) {
    StoreWaitContextChecker checker(nullptr, computed_buffers_);

    const int64_t request_id       = 3005;
    const int64_t expired_horizon  = currentTimeMs() + 5000;
    const int64_t transfer_horizon = currentTimeMs() + 1000;
    auto          collector        = std::make_shared<P2PConnectorMetricsCollector>();

    ASSERT_TRUE(computed_buffers_->registerRequestHorizon(request_id, expired_horizon, expired_horizon).has_value());
    ASSERT_TRUE(computed_buffers_->activateRequestHorizon(request_id, transfer_horizon, expired_horizon).has_value());
    checker.addContext(StoreWaitContext(request_id, nullptr, createLayerCacheBuffer(0), expired_horizon, collector));

    checker.checkOnce();
    EXPECT_EQ(checker.getContextCount(), 0);
    auto buffer = computed_buffers_->getBuffer(request_id, expired_horizon);
    ASSERT_NE(buffer, nullptr);
    EXPECT_EQ(buffer->deadlineMs(), transfer_horizon);
    EXPECT_EQ(computed_buffers_->getBuffer(request_id, transfer_horizon), nullptr);
}

TEST_F(StoreWaitContextTest, TerminalRequestRejectsLateReadyEvent) {
    StoreWaitContextChecker checker(nullptr, computed_buffers_);
    const auto request_deadline_ms = currentTimeMs() + 5000;
    ASSERT_TRUE(computed_buffers_->registerRequestHorizon(3006, request_deadline_ms, request_deadline_ms).has_value());
    computed_buffers_->removeBuffer(3006, request_deadline_ms);
    checker.addContext(StoreWaitContext(3006, nullptr, createLayerCacheBuffer(0), request_deadline_ms, nullptr));
    checker.checkOnce();
    EXPECT_EQ(checker.getContextCount(), 0);
    EXPECT_EQ(computed_buffers_->getBuffer(3006, request_deadline_ms), nullptr);
}

TEST_F(StoreWaitContextTest, HorizonChangesNotifyAndRescheduleNearestTimeout) {
    auto       signal     = computed_buffers_->notification();
    auto       generation = signal->generation();
    const auto deadline   = currentTimeMs() + 5000;
    ASSERT_TRUE(computed_buffers_->registerRequestHorizon(4001, deadline, deadline));
    EXPECT_NE(signal->generation(), generation);
    EXPECT_EQ(computed_buffers_->nextTimeoutMs(), deadline);
    generation                   = signal->generation();
    const auto transfer_deadline = deadline - 1000;
    ASSERT_TRUE(computed_buffers_->activateRequestHorizon(4001, transfer_deadline, deadline));
    EXPECT_NE(signal->generation(), generation);
    EXPECT_EQ(computed_buffers_->nextTimeoutMs(), transfer_deadline);
    generation = signal->generation();
    const auto before_remove = currentTimeMs();
    computed_buffers_->removeBuffer(4001, deadline);
    EXPECT_NE(signal->generation(), generation);
    EXPECT_GE(computed_buffers_->nextTimeoutMs(), before_remove + 3600000);
    EXPECT_LE(computed_buffers_->nextTimeoutMs(), currentTimeMs() + 3600000);
}

TEST_F(StoreWaitContextTest, EmptyStoreHasNoPeriodicWakeupDeadline) {
    EXPECT_EQ(computed_buffers_->nextTimeoutMs(), std::numeric_limits<int64_t>::max());
}

TEST_F(StoreWaitContextTest, LateContextDoesNotPublishIntoSameIdWithNewDeadline) {
    StoreWaitContextChecker checker(nullptr, computed_buffers_);
    const int64_t           request_id = 3010;
    const auto              deadline_a = currentTimeMs() + 5000;
    const auto              deadline_b = deadline_a + 1000;
    ASSERT_TRUE(computed_buffers_->registerRequestHorizon(request_id, deadline_a, deadline_a));
    ASSERT_TRUE(computed_buffers_->registerRequestHorizon(request_id, deadline_b, deadline_b));
    computed_buffers_->removeBuffer(request_id, deadline_a);
    auto layer_b = createLayerCacheBuffer(0);
    checker.addContext(StoreWaitContext(request_id, nullptr, layer_b, deadline_b, nullptr));
    checker.addContext(StoreWaitContext(request_id, nullptr, createLayerCacheBuffer(0), deadline_a, nullptr));
    EXPECT_EQ(computed_buffers_->getBuffer(request_id, deadline_a), nullptr);
    auto buffer_b = computed_buffers_->getBuffer(request_id, deadline_b);
    ASSERT_NE(buffer_b, nullptr);
    EXPECT_EQ(buffer_b->getBuffers({"0:full"}).second.at(0), layer_b);
}

}  // namespace rtp_llm
