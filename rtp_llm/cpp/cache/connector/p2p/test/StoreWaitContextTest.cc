#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <chrono>
#include <optional>

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
        return std::make_shared<LayerCacheBuffer>(layer_id);
    }

    int64_t getDeadlineMs(int64_t offset_ms = 1000) {
        return currentTimeMs() + offset_ms;
    }

protected:
    std::shared_ptr<ComputedLayerCacheBufferStore> computed_buffers_;
};

TEST_F(StoreWaitContextTest, CheckerCheckOnce_NoEvent_TreatedAsReady) {
    StoreWaitContextChecker checker(nullptr, computed_buffers_);

    int64_t request_id  = 3001;
    auto    buffer      = createLayerCacheBuffer(0);
    int64_t deadline_ms = getDeadlineMs();
    auto    collector   = std::make_shared<PrefillWorkerStoreMetricsCollector>();

    // nullopt event is treated as ready
    StoreWaitContext context(request_id, std::nullopt, buffer, deadline_ms, collector);
    checker.addContext(std::move(context));

    EXPECT_EQ(checker.getContextCount(), 1);
    checker.checkOnce();
    EXPECT_EQ(checker.getContextCount(), 0);

    auto computed_buffer = computed_buffers_->getBuffer(request_id);
    ASSERT_NE(computed_buffer, nullptr);
}

TEST_F(StoreWaitContextTest, CheckerCheckOnce_Timeout) {
    StoreWaitContextChecker checker(nullptr, computed_buffers_);

    int64_t request_id  = 3004;
    auto    buffer      = createLayerCacheBuffer(0);
    int64_t deadline_ms = currentTimeMs() - 100;  // already expired
    auto    collector   = std::make_shared<PrefillWorkerStoreMetricsCollector>();

    // Even with nullopt, timeout takes precedence
    StoreWaitContext context(request_id, std::nullopt, buffer, deadline_ms, collector);
    checker.addContext(std::move(context));

    EXPECT_EQ(checker.getContextCount(), 1);
    checker.checkOnce();
    EXPECT_EQ(checker.getContextCount(), 0);

    // Buffer should NOT be added (timeout, not success)
    auto computed_buffer = computed_buffers_->getBuffer(request_id);
    EXPECT_EQ(computed_buffer, nullptr);
}

}  // namespace rtp_llm
