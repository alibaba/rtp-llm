// Copyright (c) 2026 Alibaba Cloud.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <thread>
#include <utility>

namespace rtp_llm {

class HixlTimerQueue {
public:
    using Clock = std::chrono::steady_clock;

    HixlTimerQueue();
    ~HixlTimerQueue();

    void schedule(Clock::time_point deadline, std::function<void()> callback);
    void stop();

private:
    void run();

    using Key = std::pair<Clock::time_point, uint64_t>;
    std::mutex                           mutex_;
    std::condition_variable              changed_;
    std::map<Key, std::function<void()>> pending_;
    uint64_t                             sequence_{0};
    bool                                 stopped_{false};
    std::thread                          worker_;
};

}  // namespace rtp_llm
