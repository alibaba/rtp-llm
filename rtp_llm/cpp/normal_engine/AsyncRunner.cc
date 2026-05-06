#include "rtp_llm/cpp/normal_engine/AsyncRunner.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include <thread>
#include <utility>

namespace rtp_llm {

AsyncRunner::AsyncRunner(torch::Stream stream): stream_(stream), event_(stream.device_type()) {
    thread_ = std::thread([this] { workerLoop(); });
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
            RTP_LLM_PROFILE_SCOPE("async_runner.thread");
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
            pending_exception_ = exception;
            task_done_         = true;
        }
        cv_done_.notify_one();
    }
}

}  // namespace rtp_llm
