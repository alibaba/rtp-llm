#include <atomic>
#include <chrono>
#include <thread>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/model_rpc/UpstreamCancellationRelay.h"

namespace rtp_llm {
namespace {

TEST(UpstreamCancellationRelayTest, PropagatesCancellation) {
    std::atomic<bool> upstream_cancelled{false};
    std::atomic<int>  downstream_cancels{0};
    UpstreamCancellationRelay relay(
        [&] { return upstream_cancelled.load(); },
        [&] { downstream_cancels.fetch_add(1); },
        std::chrono::milliseconds(1));
    upstream_cancelled.store(true);
    for (int i = 0; i < 1000 && downstream_cancels.load() == 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    relay.stop();
    EXPECT_EQ(downstream_cancels.load(), 1);
}

TEST(UpstreamCancellationRelayTest, StopPreventsLateCancellation) {
    std::atomic<bool> upstream_cancelled{false};
    std::atomic<int>  downstream_cancels{0};
    UpstreamCancellationRelay relay(
        [&] { return upstream_cancelled.load(); },
        [&] { downstream_cancels.fetch_add(1); },
        std::chrono::milliseconds(1));
    relay.stop();
    upstream_cancelled.store(true);
    EXPECT_EQ(downstream_cancels.load(), 0);
}

TEST(UpstreamCancellationRelayTest, AlreadyCancelledUpstreamIsRelayed) {
    std::atomic<int> downstream_cancels{0};
    UpstreamCancellationRelay relay(
        [] { return true; },
        [&] { downstream_cancels.fetch_add(1); },
        std::chrono::milliseconds(1));
    for (int i = 0; i < 1000 && downstream_cancels.load() == 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    relay.stop();
    EXPECT_EQ(downstream_cancels.load(), 1);
}

}  // namespace
}  // namespace rtp_llm
