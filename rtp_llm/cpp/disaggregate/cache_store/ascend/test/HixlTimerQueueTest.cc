// Copyright (c) 2026 Alibaba Cloud.
// SPDX-License-Identifier: Apache-2.0
#include "rtp_llm/cpp/disaggregate/cache_store/ascend/HixlTimerQueue.h"

#include <gtest/gtest.h>
#include <future>
#include <numeric>
#include <vector>

namespace rtp_llm {
namespace {

TEST(HixlTimerQueueTest, IdenticalDeadlinesRunEveryCallbackInRegistrationOrder) {
    using namespace std::chrono_literals;
    constexpr int count = 32;
    std::vector<int> order;
    std::promise<void> fired;
    auto done = fired.get_future();
    HixlTimerQueue queue;
    const auto deadline = HixlTimerQueue::Clock::now() + 20ms;
    for (int i = 0; i < count; ++i) {
        queue.schedule(deadline, [&, i] {
            order.push_back(i);
            if (i == count - 1) {
                fired.set_value();
            }
        });
    }
    ASSERT_EQ(done.wait_for(2s), std::future_status::ready);
    queue.stop();
    std::vector<int> expected(count);
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(order, expected);
}

}  // namespace
}  // namespace rtp_llm
