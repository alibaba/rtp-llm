#include "rtp_llm/cpp/normal_engine/AsyncRunner.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include <ATen/record_function.h>
#include <thread>
#include <utility>

namespace rtp_llm {

AsyncRunner::AsyncRunner(torch::Stream stream): stream_(stream) {
    thread_ = std::thread([this] {
        cuda_graph::setDevice(static_cast<int>(stream_.device_index()));
        workerLoop();
    });
}

AsyncRunner::~AsyncRunner() {
    try {
        drain();
    } catch (...) {
        // Destructors cannot propagate CUDA teardown errors.
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
    if (!completed_tasks_.empty()) {
        completed_tasks_.back().event.block(wait_stream);
    }
}

void AsyncRunner::drain() {
    std::unique_lock<std::mutex> lk(mutex_);
    cv_done_.wait(lk, [this] { return task_done_; });
    pending_exception_ = nullptr;
    // No worker can publish or release a capture while task_done_ is true.
    stream_.synchronize();
    completed_tasks_.clear();
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
        torch::Event       completion(stream_.device_type());
        {
            at::ThreadLocalStateGuard tls_guard(task.tls_state);
            // Do not propagate Torch profiler callbacks into this worker: Kineto
            // callbacks are thread-affine and can crash when the main engine thread
            // starts/stops profiling while async bookkeeping still runs ATen ops.
            at::DisableRecordFunctionGuard record_function_guard;
            cuda_graph::GraphStreamGuard stream_guard(cuda_graph::toGraphStream(stream_));
            try {
                task.fn();
            } catch (...) {
                exception = std::current_exception();
            }
            // A failed task may already have submitted GPU copies. Record their
            // completion too, and retain its snapshot until those copies finish.
            try {
                completion.record(stream_);
            } catch (...) {
                if (!exception) {
                    exception = std::current_exception();
                }
                try {
                    stream_.synchronize();
                } catch (...) {
                    exception = std::current_exception();
                }
            }
        }

        {
            std::lock_guard<std::mutex> lk(mutex_);
            try {
                while (!completed_tasks_.empty() && completed_tasks_.front().event.query()) {
                    completed_tasks_.pop_front();
                }
            } catch (...) {
                if (!exception)
                    exception = std::current_exception();
            }
            completed_tasks_.push_back({std::move(completion), std::move(task.fn)});
            pending_exception_ = exception;
            task_done_         = true;
        }
        cv_done_.notify_one();
    }
}

}  // namespace rtp_llm
