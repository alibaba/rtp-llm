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
    explicit AsyncRunner(torch::Stream stream);
    ~AsyncRunner();

    AsyncRunner(const AsyncRunner&)            = delete;
    AsyncRunner& operator=(const AsyncRunner&) = delete;

    void launch(std::function<void()> fn);
    void sync(const torch::Stream& wait_stream);
    // CPU-only completion wait: blocks the calling thread until the pending
    // task finishes, without chaining any GPU stream behind the worker's
    // event.  Use for pipeline-depth bounding where the caller consumes no
    // device output of the worker.
    void syncCpu();

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
        std::function<void()> fn;
        at::ThreadLocalState  tls_state;
    };
    std::optional<Task> pending_task_;
    std::exception_ptr  pending_exception_;
    bool                task_done_ = true;
    bool                shutdown_  = false;
};

}  // namespace rtp_llm
