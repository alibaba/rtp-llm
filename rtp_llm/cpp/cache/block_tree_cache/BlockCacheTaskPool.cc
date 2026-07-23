#include "rtp_llm/cpp/cache/block_tree_cache/BlockCacheTaskPool.h"

#include <utility>

#include "autil/LambdaWorkItem.h"
#include "autil/LockFreeThreadPool.h"

namespace rtp_llm {
namespace {

template<typename Cleanup>
class ScopeExit {
public:
    explicit ScopeExit(Cleanup cleanup): cleanup_(std::move(cleanup)) {}

    ~ScopeExit() noexcept {
        cleanup_();
    }

    ScopeExit(const ScopeExit&)            = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;
    ScopeExit(ScopeExit&&)                 = delete;
    ScopeExit& operator=(ScopeExit&&)      = delete;

private:
    Cleanup cleanup_;
};

}  // namespace

BlockCacheTaskPool::BlockCacheTaskPool(size_t thread_count, size_t queue_size, std::string thread_name):
    thread_count_(thread_count), queue_size_(queue_size), thread_name_(std::move(thread_name)) {}

BlockCacheTaskPool::~BlockCacheTaskPool() {
    shutdown();
}

bool BlockCacheTaskPool::start() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    if (started_ || shutdown_ || thread_count_ == 0 || queue_size_ == 0) {
        return false;
    }

    auto thread_pool =
        std::make_shared<autil::LockFreeThreadPool>(thread_count_, queue_size_, nullptr, thread_name_.c_str());
    if (!thread_pool->start()) {
        return false;
    }
    thread_pool_ = std::move(thread_pool);
    started_     = true;
    return true;
}

bool BlockCacheTaskPool::submit(std::function<void()> task) {
    if (!task) {
        return false;
    }

    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    if (!started_ || shutdown_ || thread_pool_ == nullptr) {
        return false;
    }

    auto* work_item = new autil::LambdaWorkItem([this, task = std::move(task)]() mutable {
        auto                               task_finished = [this]() { taskFinished(); };
        ScopeExit<decltype(task_finished)> task_finish_guard(std::move(task_finished));

        task();
    });

    taskStarted();
    const autil::ThreadPool::ERROR_TYPE error = thread_pool_->pushWorkItem(work_item);
    if (error != autil::ThreadPool::ERROR_NONE) {
        work_item->destroy();
        taskFinished();
        return false;
    }
    return true;
}

void BlockCacheTaskPool::waitForIdle() {
    std::unique_lock<std::mutex> lock(wait_mutex_);
    bool                         wait_observer_invoked = false;
    wait_cv_.wait(lock, [this, &wait_observer_invoked] {
        const int pending_tasks = pending_tasks_.load();
        if (pending_tasks > 0 && !wait_observer_invoked) {
            wait_observer_invoked = true;
            const auto observer   = pending_task_wait_observer_for_test_;
            if (observer) {
                observer();
            }
        }
        return pending_tasks <= 0;
    });
}

void BlockCacheTaskPool::shutdown() {
    std::shared_ptr<autil::LockFreeThreadPool> thread_pool;
    bool                                       was_started = false;
    {
        std::lock_guard<std::mutex> lock(lifecycle_mutex_);
        if (shutdown_) {
            return;
        }
        shutdown_   = true;
        thread_pool = thread_pool_;
        was_started = started_;
    }

    if (thread_pool != nullptr && was_started) {
        thread_pool->stop(autil::ThreadPool::STOP_AFTER_QUEUE_EMPTY);
        thread_pool->join();
    }
}

void BlockCacheTaskPool::taskStarted() {
    pending_tasks_.fetch_add(1);
}

void BlockCacheTaskPool::taskFinished() {
    const int remaining = pending_tasks_.fetch_sub(1) - 1;
    if (remaining <= 0) {
        std::lock_guard<std::mutex> lock(wait_mutex_);
        wait_cv_.notify_all();
    }
}

}  // namespace rtp_llm
