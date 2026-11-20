#include "rtp_llm/cpp/normal_engine/AsyncRunner.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include <ATen/record_function.h>
#include <thread>
#include <utility>

namespace rtp_llm {

AsyncRunner::AsyncRunner(torch::Stream stream): stream_(stream), event_(stream.device_type()) {
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
    try {
        drainFailedTask();
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("failed to drain AsyncRunner stream during destruction: %s", error.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("failed to drain AsyncRunner stream during destruction");
    }
}

void AsyncRunner::launch(std::function<void()> fn) {
    RTP_LLM_PROFILE_SCOPE("async_runner.launch");
    at::ThreadLocalState tls_state;
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

void AsyncRunner::rethrowPendingExceptionIfAny(std::unique_lock<std::mutex>& lk) {
    if (!pending_exception_) {
        return;
    }
    drainFailedTask();
    auto exception = std::exchange(pending_exception_, nullptr);
    lk.unlock();
    std::rethrow_exception(exception);
}

void AsyncRunner::drainFailedTask() {
    if (!failed_task_) {
        return;
    }
    // Success keeps the existing asynchronous stream dependency. Errors must
    // finish device reads before the caller unwinds and releases host inputs.
    if (completion_event_recorded_) {
        event_.synchronize();
    } else {
        // Recording the completion event itself can fail after work was queued.
        stream_.synchronize();
    }
    failed_task_.reset();
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
        bool               completion_event_recorded = false;
        {
            at::ThreadLocalStateGuard tls_guard(task.tls_state);
            // Do not propagate Torch profiler callbacks into this worker: Kineto
            // callbacks are thread-affine and can crash when the main engine thread
            // starts/stops profiling while async bookkeeping still runs ATen ops.
            at::DisableRecordFunctionGuard record_function_guard;
            cuda_graph::GraphStreamGuard   stream_guard(cuda_graph::toGraphStream(stream_));
            try {
                task.fn();
            } catch (...) {
                exception = std::current_exception();
            }
            try {
                event_.record(stream_);
                completion_event_recorded = true;
            } catch (...) {
                if (!exception) {
                    exception = std::current_exception();
                }
            }
        }

        {
            std::lock_guard<std::mutex> lk(mutex_);
            pending_exception_         = exception;
            completion_event_recorded_ = completion_event_recorded;
            if (exception) {
                failed_task_ = std::move(task);
            }
            task_done_ = true;
        }
        cv_done_.notify_one();
    }
}

}  // namespace rtp_llm
