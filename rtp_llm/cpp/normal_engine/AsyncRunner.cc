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

void AsyncRunner::launch(std::function<void()> fn, std::function<void(std::exception_ptr)> on_unstarted_failure) {
    RTP_LLM_PROFILE_SCOPE("async_runner.launch");
    std::optional<at::ThreadLocalState> tls_state;
    if (propagate_thread_local_state_) {
        tls_state.emplace();
    }
    {
        std::unique_lock<std::mutex> lk(mutex_);
        cv_done_.wait(lk, [this] { return task_done_ && !draining_; });
        rethrowPendingExceptionIfAny(lk);
        pending_task_ = Task{std::move(fn), std::move(tls_state), std::move(on_unstarted_failure)};
        task_done_    = false;
    }
    cv_task_.notify_one();
}

void AsyncRunner::sync(const torch::Stream& wait_stream) {
    RTP_LLM_PROFILE_SCOPE("async_runner.sync");
    std::unique_lock<std::mutex> lk(mutex_);
    cv_done_.wait(lk, [this] { return task_done_ && !draining_; });
    rethrowPendingExceptionIfAny(lk);
    lk.unlock();
    event_.block(wait_stream);
}

void AsyncRunner::streamWait(const torch::Stream& wait_stream) {
    event_.block(wait_stream);
}

std::exception_ptr AsyncRunner::joinAndDrain() {
    if (std::this_thread::get_id() == thread_.get_id()) {
        throw std::logic_error("AsyncRunner cannot drain its own worker");
    }
    std::unique_lock<std::mutex> lk(mutex_);
    cv_done_.wait(lk, [this] { return task_done_ && !draining_; });
    // Exclude launch without keeping a CPU mutex held during GPU completion.
    // event_ may belong to an older successful task after a throwing task.
    draining_ = true;
    lk.unlock();
    try {
        cuda_graph::GraphStreamGuard stream_guard(cuda_graph::toGraphStream(stream_));
        stream_.synchronize();
    } catch (...) {
        lk.lock();
        draining_ = false;
        lk.unlock();
        cv_done_.notify_all();
        throw;
    }
    lk.lock();
    auto error = std::exchange(pending_exception_, nullptr);
    draining_  = false;
    lk.unlock();
    cv_done_.notify_all();
    return error;
}

void AsyncRunner::rethrowPendingExceptionIfAny(std::unique_lock<std::mutex>& lk) {
    if (!pending_exception_) {
        return;
    }
    auto exception = std::exchange(pending_exception_, nullptr);
    lk.unlock();
    std::rethrow_exception(exception);
}

void AsyncRunner::finishUnstartedTaskFailure(Task& task, std::exception_ptr& error) {
    if (!error || !task.on_unstarted_failure) {
        return;
    }
    auto cleanup              = std::move(task.on_unstarted_failure);
    task.on_unstarted_failure = {};
    try {
        cleanup(error);
    } catch (...) {
        error = std::current_exception();
    }
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
        bool               task_started = false;
        auto               run_task     = [&]() {
            // Kineto callbacks are thread-affine; async worker tasks must not
            // inherit or create record-function scopes.
            at::DisableRecordFunctionGuard record_function_guard;
            cuda_graph::GraphStreamGuard   stream_guard(cuda_graph::toGraphStream(stream_));
            task_started = true;
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
        if (!task_started) {
            finishUnstartedTaskFailure(task, exception);
        }

        {
            std::lock_guard<std::mutex> lk(mutex_);
            pending_exception_ = exception;
            task_done_         = true;
        }
        cv_done_.notify_all();
    }
}

}  // namespace rtp_llm
