#pragma once

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <ATen/ThreadLocalState.h>
#include <torch/torch.h>
#include "rtp_llm/cpp/utils/cuda_call_trace/CudaCallTrace.h"

namespace rtp_llm {

class AsyncRunner {
public:
    explicit AsyncRunner(torch::Stream stream);
    ~AsyncRunner();

    AsyncRunner(const AsyncRunner&)            = delete;
    AsyncRunner& operator=(const AsyncRunner&) = delete;

    void launch(std::function<void()> fn);
    void launch(std::string task_name, std::function<void()> fn);
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
        std::function<void()> fn;
        at::ThreadLocalState  tls_state;
        std::string           name;
        cuda_trace::Context   trace_context{};
        uint64_t              trace_task_id = 0;
    };
    std::optional<Task> pending_task_;
    std::exception_ptr  pending_exception_;
    bool                task_done_ = true;
    bool                shutdown_  = false;
    // Diagnostic scalar, protected by mutex_ just like task_done_. This does
    // not own GPU storage or change the existing event's lifetime.
    uint64_t trace_completed_task_id_ = 0;
};

}  // namespace rtp_llm
