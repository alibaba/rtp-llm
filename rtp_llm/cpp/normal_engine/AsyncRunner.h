#pragma once

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <ATen/ThreadLocalState.h>
#include <torch/torch.h>

namespace rtp_llm {

class AsyncRunner {
public:
    explicit AsyncRunner(torch::Stream stream, bool propagate_thread_local_state = true);
    ~AsyncRunner();

    AsyncRunner(const AsyncRunner&)            = delete;
    AsyncRunner& operator=(const AsyncRunner&) = delete;

    void launch(std::function<void()> fn);
    void sync(const torch::Stream& wait_stream);

private:
    void workerLoop();
    void rethrowPendingExceptionIfAny(std::unique_lock<std::mutex>& lk);

    torch::Stream stream_;
    torch::Event  event_;

    std::thread             thread_;
    std::mutex              mutex_;
    std::condition_variable cv_task_;
    std::condition_variable cv_done_;

    struct Task {
        std::function<void()>               fn;
        std::optional<at::ThreadLocalState> tls_state;
    };
    std::optional<Task> pending_task_;
    std::exception_ptr  pending_exception_;
    bool                task_done_ = true;
    bool                shutdown_  = false;
    bool                propagate_thread_local_state_;
};

}  // namespace rtp_llm
