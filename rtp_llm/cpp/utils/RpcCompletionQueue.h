#pragma once

#include <grpc++/grpc++.h>
#include <grpc++/alarm.h>
#include <grpc/support/time.h>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>

namespace rtp_llm {

// One blocking CQ consumer per process. A tag owns its RPC storage until Finish,
// independently of the request/checker lifetime. Completion callbacks must not block.
class RpcCompletionQueue {
    struct Tag {
        std::shared_ptr<grpc::ClientContext> context;
        std::function<void(bool)>            complete;
        std::shared_ptr<grpc::Alarm>        alarm;
    };
    struct State {
        grpc::CompletionQueue                           queue;
        std::mutex                                      submit_mutex;
        std::mutex                                      mutex;
        std::condition_variable                         drained_cv;
        std::unordered_map<void*, std::unique_ptr<Tag>> tags;
        bool                                            stopping{false};
        bool                                            drained{false};
    };

public:
    static RpcCompletionQueue& instance() {
        static RpcCompletionQueue queue;
        return queue;
    }

    RpcCompletionQueue():
        state_(std::make_shared<State>()), thread_([state = state_] {
            void* key = nullptr;
            bool  ok  = false;
            while (state->queue.Next(&key, &ok)) {
                std::unique_ptr<Tag> tag;
                {
                    std::lock_guard<std::mutex> lock(state->mutex);
                    auto                        it = state->tags.find(key);
                    if (it == state->tags.end()) {
                        std::fputs("RPC completion queue received an unregistered tag\n", stderr);
                        continue;
                    }
                    tag = std::move(it->second);
                    state->tags.erase(it);
                }
                try {
                    tag->complete(ok);
                } catch (const std::exception& error) {
                    std::fprintf(stderr, "RPC completion callback failed: %s\n", error.what());
                } catch (...) {
                    std::fputs("RPC completion callback failed with an unknown exception\n", stderr);
                }
            }
            {
                std::lock_guard<std::mutex> lock(state->mutex);
                state->drained = true;
            }
            state->drained_cv.notify_all();
        }) {}

    ~RpcCompletionQueue() {
        // Serialize Shutdown with RPC creation without delaying CQ dispatch.
        std::unique_lock<std::mutex> submit_lock(state_->submit_mutex);
        std::unique_lock<std::mutex> lock(state_->mutex);
        state_->stopping = true;
        for (const auto& entry : state_->tags) {
            if (entry.second->context) {
                entry.second->context->TryCancel();
            }
            if (entry.second->alarm) {
                entry.second->alarm->Cancel();
            }
        }
        state_->queue.Shutdown();
        submit_lock.unlock();
        // A stalled transport must not make shutdown unbounded. The consumer
        // owns State (CQ and outstanding RPCs), so detaching cannot free live data.
        const bool drained =
            state_->drained_cv.wait_for(lock, std::chrono::milliseconds(100), [this] { return state_->drained; });
        lock.unlock();
        if (drained) {
            thread_.join();
        } else {
            thread_.detach();
        }
    }

    // start must enqueue exactly one Finish event on success, none on failure.
    template<typename Start>
    bool submit(const std::shared_ptr<grpc::ClientContext>& context,
                const Start&                                start,
                std::function<void(bool)>                   complete) {
        auto                        tag = std::make_unique<Tag>(Tag{context, std::move(complete), nullptr});
        void*                       key = tag.get();
        std::lock_guard<std::mutex> submit_lock(state_->submit_mutex);
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            if (state_->stopping) {
                return false;
            }
            state_->tags.emplace(key, std::move(tag));
        }
        try {
            if (start(&state_->queue, key)) {
                return true;
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(state_->mutex);
            state_->tags.erase(key);
            throw;
        }
        std::lock_guard<std::mutex> lock(state_->mutex);
        state_->tags.erase(key);
        return false;
    }

    std::function<void()> scheduleAfter(int64_t delay_ms, std::function<void(bool)> complete) {
        auto alarm = std::make_shared<grpc::Alarm>();
        auto tag = std::make_unique<Tag>(Tag{nullptr, std::move(complete), alarm});
        void* key = tag.get();
        const auto deadline = gpr_time_add(gpr_now(GPR_CLOCK_MONOTONIC), gpr_time_from_millis(delay_ms, GPR_TIMESPAN));
        std::function<void()> cancel =
            [weak_state = std::weak_ptr<State>(state_), weak_alarm = std::weak_ptr<grpc::Alarm>(alarm), key] {
                auto state = weak_state.lock();
                auto alarm = weak_alarm.lock();
                if (!state || !alarm) {
                    return;
                }
                std::lock_guard<std::mutex> lock(state->mutex);
                const auto it = state->tags.find(key);
                if (!state->stopping && it != state->tags.end() && it->second->alarm == alarm) {
                    alarm->Cancel();
                }
            };
        // Scheduling a timer must not wait behind another caller starting an RPC.
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->stopping) {
            return {};
        }
        state_->tags.emplace(key, std::move(tag));
        alarm->Set(&state_->queue, deadline, key);
        return cancel;
    }

private:
    std::shared_ptr<State> state_;
    std::thread            thread_;
};

}  // namespace rtp_llm
