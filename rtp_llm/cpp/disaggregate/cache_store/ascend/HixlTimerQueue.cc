// Copyright (c) 2026 Alibaba Cloud.
// SPDX-License-Identifier: Apache-2.0
#include "rtp_llm/cpp/disaggregate/cache_store/ascend/HixlTimerQueue.h"

namespace rtp_llm {

HixlTimerQueue::HixlTimerQueue(): worker_([this] { run(); }) {}

HixlTimerQueue::~HixlTimerQueue() {
    stop();
}

void HixlTimerQueue::schedule(Clock::time_point deadline, std::function<void()> callback) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_) {
            return;
        }
        pending_.emplace(std::make_pair(deadline, sequence_++), std::move(callback));
    }
    changed_.notify_one();
}

void HixlTimerQueue::stop() {
    decltype(pending_) discarded;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        pending_.swap(discarded);
    }
    changed_.notify_one();
    if (worker_.joinable()) {
        worker_.join();
    }
}

void HixlTimerQueue::run() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stopped_) {
        if (pending_.empty()) {
            changed_.wait(lock);
            continue;
        }
        const auto deadline = pending_.begin()->first.first;
        if (deadline > Clock::now()) {
            changed_.wait_until(lock, deadline);
            continue;
        }
        {
            auto entry = pending_.extract(pending_.begin());
            lock.unlock();
            entry.mapped()();
        }
        lock.lock();
    }
}

}  // namespace rtp_llm
