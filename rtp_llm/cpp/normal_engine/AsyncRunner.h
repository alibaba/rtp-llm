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
    // Wait for the pending task to finish and rethrow its exception; then make
    // the caller's stream wait on the completion event. On single-stream
    // devices (Ascend default-stream execution) use the no-arg overload: the
    // host join alone preserves ordering.
    void sync(const torch::Stream& wait_stream);
    void sync();

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
