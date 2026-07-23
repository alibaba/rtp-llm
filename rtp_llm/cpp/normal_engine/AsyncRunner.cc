#include "rtp_llm/cpp/normal_engine/AsyncRunner.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include <ATen/record_function.h>
#include <thread>
#include <utility>

namespace rtp_llm {

AsyncRunner::AsyncRunner(torch::Stream stream, bool propagate_thread_local_state):
    stream_(stream), event_(stream.device_type()), propagate_thread_local_state_(propagate_thread_local_state) {
    thread_ = std::thread([this] {
        cuda_graph::setDevice(static_cast<int>(stream_.device_index()));
        workerLoop();
    });
}

AsyncRunner::~AsyncRunner() {
    {
        std::lock_guard<std::mutex> lk(mutex_);
        shutdown_ = true;
    }
    cv_task_.notify_one();
    if (thread_.joinable()) {
        thread_.join();
    }
}

void AsyncRunner::launch(std::function<void()> fn) {
    RTP_LLM_PROFILE_SCOPE("async_runner.launch");
    std::optional<at::ThreadLocalState> tls_state;
    if (propagate_thread_local_state_) {
        tls_state.emplace();
    }
    {
        std::unique_lock<std::mutex> lk(mutex_);
        cv_done_.wait(lk, [this] { return task_done_; });
        rethrowPendingExceptionIfAny(lk);
        pending_task_ = Task{std::move(fn), std::move(tls_state)};
        task_done_    = false;
    }
    cv_task_.notify_one();
}

void AsyncRunner::sync(const torch::Stream& wait_stream) {
    RTP_LLM_PROFILE_SCOPE("async_runner.sync");
    std::unique_lock<std::mutex> lk(mutex_);
    cv_done_.wait(lk, [this] { return task_done_; });
    rethrowPendingExceptionIfAny(lk);
    lk.unlock();
    event_.block(wait_stream);
}

void AsyncRunner::streamWait(const torch::Stream& wait_stream) {
    // A cudaStreamWaitEvent issued before the worker records this task's event
    // only observes the event's previous generation (or is a no-op when it has
    // never been recorded).  Wait until the worker has enqueued the current
    // task and recorded event_ before adding the cross-stream dependency.
    //
    // This CPU wait is required for correctness: callers use streamWait() at
    // the beginning of the next iteration, while the worker may still be doing
    // host bookkeeping before it can enqueue any CUDA work.
    std::unique_lock<std::mutex> lk(mutex_);
    cv_done_.wait(lk, [this] { return task_done_; });
    rethrowPendingExceptionIfAny(lk);
    lk.unlock();
    event_.block(wait_stream);
}

void AsyncRunner::rethrowPendingExceptionIfAny(std::unique_lock<std::mutex>& lk) {
    if (!pending_exception_) {
        return;
    }
    auto exception = std::exchange(pending_exception_, nullptr);
    lk.unlock();
    std::rethrow_exception(exception);
}

void AsyncRunner::workerLoop() {
    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lk(mutex_);
            cv_task_.wait(lk, [this] { return pending_task_.has_value() || shutdown_; });
            if (shutdown_ && !pending_task_.has_value()) {
                return;
            }
            task = std::move(*pending_task_);
            pending_task_.reset();
        }

        std::exception_ptr exception;
        auto               run_task = [&]() {
            // REBASE CONFLICT CONTEXT(704f3c147): new base disables record
            // function callbacks in this worker; source branch added the
            // explicit async thread profile scope. Keep both around the worker
            // task body.
            RTP_LLM_PROFILE_SCOPE("async_runner.thread");
            at::DisableRecordFunctionGuard record_function_guard;
            cuda_graph::GraphStreamGuard   stream_guard(cuda_graph::toGraphStream(stream_));
            task.fn();
            event_.record(stream_);
        };
        try {
            if (task.tls_state.has_value()) {
                at::ThreadLocalStateGuard tls_guard(*task.tls_state);
                run_task();
            } else {
                run_task();
            }
        } catch (...) {
            exception = std::current_exception();
        }

        {
            std::lock_guard<std::mutex> lk(mutex_);
            pending_exception_ = exception;
            task_done_         = true;
        }
        cv_done_.notify_one();
    }
}

}  // namespace rtp_llm
