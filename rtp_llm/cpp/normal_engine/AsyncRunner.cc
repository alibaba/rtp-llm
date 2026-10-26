#include "rtp_llm/cpp/normal_engine/AsyncRunner.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include <ATen/record_function.h>
#include <thread>
#include <utility>

namespace rtp_llm {
namespace {

void logAsyncFailure(const char* context, const std::exception_ptr& exception) noexcept {
    try {
        std::rethrow_exception(exception);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("[AsyncRunner] %s failed: %s", context, e.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("[AsyncRunner] %s failed with unknown exception", context);
    }
}

}  // namespace

AsyncRunner::AsyncRunner(torch::Stream stream): stream_(stream), event_(stream.device_type()) {
    thread_ = std::thread([this] {
        cuda_graph::setDevice(static_cast<int>(stream_.device_index()));
        workerLoop();
    });
}

AsyncRunner::~AsyncRunner() {
    // Finish both CPU work and submitted GPU work before shutting down.
    try {
        std::unique_lock<std::mutex> lk(mutex_);
        cv_done_.wait(lk, [this] { return task_done_; });
        stream_.synchronize();
    } catch (...) {
        logAsyncFailure("shutdown", std::current_exception());
    }
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
    launch("unnamed task", std::move(fn));
}

void AsyncRunner::launch(std::string task_name, std::function<void()> fn) {
    RTP_LLM_PROFILE_SCOPE("async_runner.launch");
    at::ThreadLocalState tls_state;
    {
        std::unique_lock<std::mutex> lk(mutex_);
        cv_done_.wait(lk, [this] { return task_done_; });
        rethrowPendingExceptionIfAny(lk);
        pending_task_ = Task{std::move(fn), std::move(tls_state), std::move(task_name)};
        task_done_    = false;
    }
    cv_task_.notify_one();
}

void AsyncRunner::sync(const torch::Stream& wait_stream) {
    RTP_LLM_PROFILE_SCOPE("async_runner.sync");
    std::unique_lock<std::mutex> lk(mutex_);
    cv_done_.wait(lk, [this] { return task_done_; });
    rethrowPendingExceptionIfAny(lk);
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
        {
            at::ThreadLocalStateGuard tls_guard(task.tls_state);
            // Do not propagate Torch profiler callbacks into this worker: Kineto
            // callbacks are thread-affine and can crash when the main engine thread
            // starts/stops profiling while async bookkeeping still runs ATen ops.
            at::DisableRecordFunctionGuard record_function_guard;
            cuda_graph::GraphStreamGuard stream_guard(cuda_graph::toGraphStream(stream_));
            try {
                task.fn();
                event_.record(stream_);
            } catch (...) {
                exception = std::current_exception();
            }
        }

        {
            std::lock_guard<std::mutex> lk(mutex_);
            // Log even when exit cleanup drains the task without a sync().
            // Keep the original exception for launch()/sync() to rethrow.
            if (exception) {
                logAsyncFailure(task.name.c_str(), exception);
            }
            pending_exception_ = exception;
            task_done_         = true;
        }
        cv_done_.notify_one();
    }
}

}  // namespace rtp_llm
