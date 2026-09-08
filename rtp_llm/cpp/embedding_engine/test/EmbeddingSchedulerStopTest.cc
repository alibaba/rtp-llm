#include <chrono>
#include <future>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/ModelConfig.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingQuery.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingScheduler.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingStream.h"

namespace rtp_llm {

TEST(EmbeddingSchedulerStopTest, CancelsQueuedStreamAndWakesWaiter) {
    ModelConfig model_config;
    ConcurrencyConfig concurrency_config;
    RuntimeConfig runtime_config;
    EmbeddingScheduler scheduler(model_config, concurrency_config, runtime_config);

    auto input = std::make_shared<EmbeddingInput>(
        std::vector<int32_t>{1}, std::vector<int32_t>{0}, std::vector<int32_t>{1}, 42);
    auto stream = std::make_shared<EmbeddingStream>(input);
    ASSERT_TRUE(scheduler.enqueue(stream).ok());
    ASSERT_EQ(scheduler.waitingStreamsSize(), 1);

    std::promise<void> waiter_started;
    auto waiter_started_future = waiter_started.get_future();
    auto waiter = std::async(std::launch::async, [stream, &waiter_started] {
        waiter_started.set_value();
        try {
            stream->waitFinish();
            return std::string{};
        } catch (const std::runtime_error& error) {
            return std::string(error.what());
        }
    });
    ASSERT_EQ(waiter_started_future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    ASSERT_EQ(waiter.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);

    EXPECT_TRUE(scheduler.stop().ok());
    ASSERT_EQ(waiter.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_NE(waiter.get().find("embedding scheduler stopped"), std::string::npos);
    EXPECT_EQ(scheduler.waitingStreamsSize(), 0);

    EXPECT_TRUE(scheduler.stop().ok());
}

}  // namespace rtp_llm
