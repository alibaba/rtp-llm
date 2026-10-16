#pragma once

#include <grpc++/grpc++.h>
#include <chrono>
#include <condition_variable>
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
    };
    struct State {
        grpc::CompletionQueue                           queue;
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
                        std::terminate();  // Only registered Finish tags may enter this CQ.
                    }
                    tag = std::move(it->second);
                    state->tags.erase(it);
                }
                tag->complete(ok);
            }
            {
                std::lock_guard<std::mutex> lock(state->mutex);
                state->drained = true;
            }
            state->drained_cv.notify_all();
        }) {}

    ~RpcCompletionQueue() {
        std::unique_lock<std::mutex> lock(state_->mutex);
        state_->stopping = true;
        for (const auto& entry : state_->tags) {
            if (entry.second->context) {
                entry.second->context->TryCancel();
            }
        }
        state_->queue.Shutdown();
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
        auto                        tag = std::make_unique<Tag>(Tag{context, std::move(complete)});
        void*                       key = tag.get();
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->stopping) {
            return false;
        }
        state_->tags.emplace(key, std::move(tag));
        try {
            if (start(&state_->queue, key)) {
                return true;
            }
        } catch (...) {
            state_->tags.erase(key);
            throw;
        }
        state_->tags.erase(key);
        return false;
    }

private:
    std::shared_ptr<State> state_;
    std::thread            thread_;
};

}  // namespace rtp_llm
