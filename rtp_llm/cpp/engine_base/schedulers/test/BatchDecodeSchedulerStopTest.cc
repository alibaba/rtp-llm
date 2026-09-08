#include <chrono>
#include <future>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/engine_base/schedulers/BatchDecodeScheduler.h"

namespace rtp_llm {

TEST(BatchDecodeSchedulerStopTest, StopWakesBlockedScheduleAndIsIdempotent) {
    RuntimeConfig runtime_config;
    BatchDecodeScheduler scheduler(runtime_config, nullptr, nullptr);

    auto scheduled = std::async(std::launch::async, [&scheduler] { return scheduler.schedule(); });
    ASSERT_EQ(scheduled.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);

    EXPECT_TRUE(scheduler.stop().ok());
    ASSERT_EQ(scheduled.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto result = scheduled.get();
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_TRUE(result.value().empty());
    EXPECT_TRUE(scheduler.empty());

    EXPECT_TRUE(scheduler.stop().ok());
    EXPECT_TRUE(scheduler.empty());
}

}  // namespace rtp_llm
