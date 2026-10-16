#pragma once

#include <chrono>
#include <condition_variable>
#include <grpc++/grpc++.h>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

namespace rtp_llm {

template<typename Entry, typename Traits>
class DeferredCompletionQueueDrainer {
public:
    static DeferredCompletionQueueDrainer& instance() {
        static DeferredCompletionQueueDrainer drainer;
        return drainer;
    }

    void enqueue(std::shared_ptr<Entry> entry) {
        if (!entry || !Traits::completionQueue(entry)) {
            return;
        }
        std::lock_guard<std::mutex> lock(mu_);
        if (stop_) {
            return;
        }
        pending_.push_back(std::move(entry));
        if (!started_) {
            started_ = true;
            thread_  = std::thread([this] { run(); });
        }
        cv_.notify_one();
    }

    ~DeferredCompletionQueueDrainer() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    DeferredCompletionQueueDrainer() = default;

    void run() {
        while (true) {
            std::list<std::shared_ptr<Entry>> local;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait_for(lock, std::chrono::milliseconds(100), [this] { return stop_ || !pending_.empty(); });
                if (stop_ && pending_.empty()) {
                    return;
                }
                local.splice(local.end(), pending_);
            }

            std::list<std::shared_ptr<Entry>> remaining;
            for (auto& entry : local) {
                auto completion_queue = Traits::completionQueue(entry);
                if (!completion_queue) {
                    continue;
                }

                void*      tag      = nullptr;
                bool       ok       = false;
                const auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(1);
                auto       status   = completion_queue->AsyncNext(&tag, &ok, deadline);
                if (status == grpc::CompletionQueue::NextStatus::SHUTDOWN) {
                    Traits::markDrained(entry);
                    continue;
                }
                remaining.push_back(std::move(entry));
            }

            if (!remaining.empty()) {
                std::lock_guard<std::mutex> lock(mu_);
                pending_.splice(pending_.end(), remaining);
            }
        }
    }

    std::mutex                       mu_;
    std::condition_variable          cv_;
    std::list<std::shared_ptr<Entry>> pending_;
    bool                             stop_    = false;
    bool                             started_ = false;
    std::thread                      thread_;
};

}  // namespace rtp_llm
