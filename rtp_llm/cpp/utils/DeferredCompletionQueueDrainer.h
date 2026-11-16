#pragma once

#include <chrono>
#include <condition_variable>
#include <grpc++/grpc++.h>
#include <list>
#include <memory>
#include <mutex>
#include <thread>

namespace rtp_llm {

template<typename Entry, typename Traits>
class DeferredCompletionQueueDrainer {
    struct State {
        std::mutex                        mutex;
        std::condition_variable           cv;
        std::list<std::shared_ptr<Entry>> pending;
        bool                              stopping = false;
        bool                              stopped  = false;
    };

public:
    static DeferredCompletionQueueDrainer& instance() {
        static DeferredCompletionQueueDrainer drainer;
        return drainer;
    }

    void enqueue(std::list<std::shared_ptr<Entry>>& entries) {
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            state_->pending.splice(state_->pending.end(), entries);
        }
        state_->cv.notify_one();
    }

    ~DeferredCompletionQueueDrainer() {
        std::unique_lock<std::mutex> lock(state_->mutex);
        state_->stopping = true;
        state_->cv.notify_all();
        const bool stopped =
            state_->cv.wait_for(lock, std::chrono::milliseconds(100), [this] { return state_->stopped; });
        lock.unlock();
        if (stopped) {
            thread_.join();
        } else {
            // Pending CQ state belongs to the worker until gRPC confirms drain.
            thread_.detach();
        }
    }

private:
    DeferredCompletionQueueDrainer(): state_(std::make_shared<State>()), thread_([state = state_] { run(state); }) {}

    static void run(const std::shared_ptr<State>& state) {
        std::list<std::shared_ptr<Entry>> local;
        std::unique_lock<std::mutex>      lock(state->mutex);
        while (true) {
            if (local.empty()) {
                state->cv.wait(lock, [&] { return state->stopping || !state->pending.empty(); });
                if (state->stopping && state->pending.empty()) {
                    state->stopped = true;
                    state->cv.notify_all();
                    return;
                }
            } else {
                state->cv.wait_for(lock, std::chrono::milliseconds(10));
            }
            local.splice(local.end(), state->pending);
            lock.unlock();
            for (auto it = local.begin(); it != local.end();) {
                auto       cq       = Traits::completionQueue(*it);
                void*      tag      = nullptr;
                bool       ok       = false;
                const auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(1);
                if (!cq || cq->AsyncNext(&tag, &ok, deadline) == grpc::CompletionQueue::NextStatus::SHUTDOWN) {
                    if (cq) {
                        Traits::markDrained(*it);
                    }
                    it = local.erase(it);
                } else {
                    ++it;
                }
            }
            lock.lock();
        }
    }

    std::shared_ptr<State> state_;
    std::thread            thread_;
};

}  // namespace rtp_llm
