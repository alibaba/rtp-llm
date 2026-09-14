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
    auto    collector   = std::make_shared<PrefillWorkerStoreMetricsCollector>();

    // null event is treated as ready
    StoreWaitContext context(request_id, nullptr, buffer, deadline_ms, collector);
    computed_buffers_->registerRequestHorizon(request_id, deadline_ms, deadline_ms);
    checker.addContext(std::move(context));

    // Fast path publishes/rejects before any background tick.
    EXPECT_EQ(checker.getContextCount(), 0);

    auto computed_buffer = computed_buffers_->getBuffer(request_id);
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
    EXPECT_NE(computed_buffers_->getBuffer(3002), nullptr);
}
#endif

TEST_F(StoreWaitContextTest, CheckerCheckOnce_Timeout) {
    StoreWaitContextChecker checker(nullptr, computed_buffers_);

    int64_t request_id  = 3004;
    auto    buffer      = createLayerCacheBuffer(0);
    int64_t deadline_ms = currentTimeMs() - 100;  // already expired
    auto    collector   = std::make_shared<PrefillWorkerStoreMetricsCollector>();

    // Even with null event, timeout takes precedence
    StoreWaitContext context(request_id, nullptr, buffer, deadline_ms, collector);
    computed_buffers_->registerRequestHorizon(request_id, deadline_ms, deadline_ms);
    checker.addContext(std::move(context));

    // Fast path publishes/rejects before any background tick.
    EXPECT_EQ(checker.getContextCount(), 0);

    // Buffer should NOT be added (timeout, not success)
    auto computed_buffer = computed_buffers_->getBuffer(request_id);
    EXPECT_EQ(computed_buffer, nullptr);
}

TEST_F(StoreWaitContextTest, CheckerUsesActivatedTransferHorizon) {
    StoreWaitContextChecker checker(nullptr, computed_buffers_);

    const int64_t request_id       = 3005;
    const int64_t expired_horizon  = currentTimeMs() + 5000;
    const int64_t transfer_horizon = currentTimeMs() + 1000;
    auto          collector        = std::make_shared<PrefillWorkerStoreMetricsCollector>();

    ASSERT_TRUE(computed_buffers_->registerRequestHorizon(request_id, expired_horizon, expired_horizon).has_value());
    ASSERT_TRUE(computed_buffers_->activateRequestHorizon(request_id, transfer_horizon, expired_horizon).has_value());
    checker.addContext(StoreWaitContext(request_id, nullptr, createLayerCacheBuffer(0), expired_horizon, collector));

    checker.checkOnce();
    EXPECT_EQ(checker.getContextCount(), 0);
    EXPECT_NE(computed_buffers_->getBuffer(request_id), nullptr);
}

TEST_F(StoreWaitContextTest, TerminalRequestRejectsLateReadyEvent) {
    StoreWaitContextChecker checker(nullptr, computed_buffers_);
    const auto request_deadline_ms = currentTimeMs() + 5000;
    ASSERT_TRUE(computed_buffers_->registerRequestHorizon(3006, request_deadline_ms, request_deadline_ms).has_value());
    computed_buffers_->removeBuffer(3006, request_deadline_ms);
    checker.addContext(StoreWaitContext(3006, nullptr, createLayerCacheBuffer(0), request_deadline_ms, nullptr));
    checker.checkOnce();
    EXPECT_EQ(checker.getContextCount(), 0);
    EXPECT_EQ(computed_buffers_->getBuffer(3006), nullptr);
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
    computed_buffers_->removeBuffer(4001, deadline);
    EXPECT_NE(signal->generation(), generation);
    EXPECT_EQ(computed_buffers_->nextTimeoutMs(), deadline);  // Tombstone expiry.
}

TEST_F(StoreWaitContextTest, EmptyStoreHasNoPeriodicWakeupDeadline) {
    EXPECT_EQ(computed_buffers_->nextTimeoutMs(), std::numeric_limits<int64_t>::max());
}

}  // namespace rtp_llm
