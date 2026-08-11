#include "rtp_llm/cpp/model_rpc/RemoteLoadTargetTracker.h"

#include <thread>
#include <vector>

#include "gtest/gtest.h"

namespace rtp_llm {
namespace {

TEST(RemoteLoadTargetTrackerTest, ReportsOnlyTargetsWhoseRpcStarted) {
    RemoteLoadTargetTracker tracker({"worker-0", "worker-1", "worker-2"});

    EXPECT_TRUE(tracker.markStarted(0));
    EXPECT_TRUE(tracker.markStarted(2));
    EXPECT_TRUE(tracker.markStarted(2));
    EXPECT_FALSE(tracker.markStarted(3));
    EXPECT_EQ(tracker.startedTargets(), (std::vector<std::string>{"worker-0", "worker-2"}));
}

TEST(RemoteLoadTargetTrackerTest, ConcurrentStartsAreVisibleToCleanup) {
    constexpr size_t kTargetCount = 32;
    std::vector<std::string> targets;
    targets.reserve(kTargetCount);
    for (size_t index = 0; index < kTargetCount; ++index) {
        targets.push_back("worker-" + std::to_string(index));
    }
    RemoteLoadTargetTracker tracker(std::move(targets));

    std::vector<std::thread> threads;
    threads.reserve(kTargetCount);
    for (size_t index = 0; index < kTargetCount; ++index) {
        threads.emplace_back([&tracker, index]() { EXPECT_TRUE(tracker.markStarted(index)); });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(tracker.startedTargets().size(), kTargetCount);
}

}  // namespace
}  // namespace rtp_llm
