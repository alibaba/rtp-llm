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

    // An accepted task may fail while restoring TLS/device state, before fn
    // runs. Its external CPU claims still need exactly-once cleanup.
    void launch(std::function<void()> fn, std::function<void(std::exception_ptr)> on_unstarted_failure = {});
    void sync(const torch::Stream& wait_stream);
    void streamWait(const torch::Stream& wait_stream);
    // Exceptional teardown only: wait for the CPU task and its actual GPU stream,
    // even when the task threw before recording event_. Return the task error
    // after draining; a GPU drain failure still throws and forbids KV reuse.
    std::exception_ptr joinAndDrain();

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
        std::function<void()>                   fn;
        std::optional<at::ThreadLocalState>     tls_state;
        std::function<void(std::exception_ptr)> on_unstarted_failure;
    };
    static void         finishUnstartedTaskFailure(Task& task, std::exception_ptr& error);
    std::optional<Task> pending_task_;
    std::exception_ptr  pending_exception_;
    bool                task_done_ = true;
    bool                draining_  = false;
    bool                shutdown_  = false;
    bool                propagate_thread_local_state_;
};

}  // namespace rtp_llm
