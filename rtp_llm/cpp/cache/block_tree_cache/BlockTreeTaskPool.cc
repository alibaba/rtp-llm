#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"

#include <cassert>
#include <utility>

#include "autil/LambdaWorkItem.h"
#include "autil/LockFreeThreadPool.h"

namespace rtp_llm {

BlockTreeTaskPool::BlockTreeTaskPool(size_t thread_count, size_t queue_size, std::string thread_name):
    thread_count_(thread_count), queue_size_(queue_size), thread_name_(std::move(thread_name)) {}

BlockTreeTaskPool::~BlockTreeTaskPool() {
    shutdown();
}

bool BlockTreeTaskPool::start() {
    std::unique_lock<std::mutex> lock(lifecycle_mutex_);
    if (started_ || shutdown_ || thread_count_ == 0 || queue_size_ == 0) {
        return false;
    }

    auto thread_pool =
        std::make_shared<autil::LockFreeThreadPool>(thread_count_, thread_count_, nullptr, thread_name_.c_str());
    if (!thread_pool->start()) {
        return false;
    }
    thread_pool_ = thread_pool;
    started_     = true;
    for (size_t index = 0; index < thread_count_; ++index) {
        auto* work_item = new autil::LambdaWorkItem([this] { workerLoop(); });
        const autil::ThreadPool::ERROR_TYPE error = thread_pool->pushWorkItem(work_item, false);
        if (error != autil::ThreadPool::ERROR_NONE) {
            work_item->destroy();
            shutdown_ = true;
            queue_cv_.notify_all();
            lock.unlock();
            thread_pool->stop(autil::ThreadPool::STOP_AFTER_QUEUE_EMPTY);
            thread_pool->join();
            return false;
        }
    }
    return true;
}

bool BlockTreeTaskPool::submit(std::function<void()> task) {
    if (!task) {
        return false;
    }

    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    if (!started_ || admission_stopped_ || shutdown_ || normal_queue_.size() >= queue_size_) {
        return false;
    }
    normal_queue_.push_back(std::move(task));
    taskStarted();
    queue_cv_.notify_one();
    return true;
}

bool BlockTreeTaskPool::submitCompletion(std::function<void()> task) {
    if (!task) {
        return false;
    }
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    if (!started_ || shutdown_) {
        return false;
    }
    completion_queue_.push_back(std::move(task));
    taskStarted();
    queue_cv_.notify_one();
    return true;
}

bool BlockTreeTaskPool::acquireBusinessCredit() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    if (!started_ || admission_stopped_ || shutdown_ || business_credits_.load() >= queue_size_) {
        return false;
    }
    business_credits_.fetch_add(1);
    return true;
}

void BlockTreeTaskPool::releaseBusinessCredit() {
    const size_t previous = business_credits_.fetch_sub(1);
    assert(previous > 0);
    (void)previous;
    std::lock_guard<std::mutex> lock(wait_mutex_);
    wait_cv_.notify_all();
}

void BlockTreeTaskPool::stopAdmission() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    admission_stopped_ = true;
}

void BlockTreeTaskPool::workerLoop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(lifecycle_mutex_);
            queue_cv_.wait(lock, [this] {
                return shutdown_ || !completion_queue_.empty() || !normal_queue_.empty();
            });
            if (!completion_queue_.empty()) {
                task = std::move(completion_queue_.front());
                completion_queue_.pop_front();
            } else if (!normal_queue_.empty()) {
                task = std::move(normal_queue_.front());
                normal_queue_.pop_front();
            } else if (shutdown_) {
                return;
            }
        }

        try {
            task();
        } catch (...) {
            // Keep the persistent worker alive. Business tasks publish their
            // own terminal error through the associated async context.
        }
        taskFinished();
    }
}

void BlockTreeTaskPool::waitForIdle() {
    std::unique_lock<std::mutex> lock(wait_mutex_);
    bool                         wait_observer_invoked = false;
    wait_cv_.wait(lock, [this, &wait_observer_invoked] {
        const int pending_tasks = pending_tasks_.load();
        if ((pending_tasks > 0 || business_credits_.load() > 0) && !wait_observer_invoked) {
            wait_observer_invoked = true;
            const auto observer   = pending_task_wait_observer_for_test_;
            if (observer) {
                observer();
            }
        }
        return pending_tasks <= 0 && business_credits_.load() == 0;
    });
}

void BlockTreeTaskPool::shutdown() {
    std::shared_ptr<autil::LockFreeThreadPool> thread_pool;
    bool                                       was_started = false;
    {
        std::lock_guard<std::mutex> lock(lifecycle_mutex_);
        if (shutdown_) {
            return;
        }
        admission_stopped_ = true;
        shutdown_          = true;
        thread_pool = thread_pool_;
        was_started = started_;
        queue_cv_.notify_all();
    }

    if (thread_pool != nullptr && was_started) {
        thread_pool->stop(autil::ThreadPool::STOP_AFTER_QUEUE_EMPTY);
        thread_pool->join();
    }
}

void BlockTreeTaskPool::taskStarted() {
    pending_tasks_.fetch_add(1);
}

void BlockTreeTaskPool::taskFinished() {
    const int remaining = pending_tasks_.fetch_sub(1) - 1;
    if (remaining <= 0) {
        std::lock_guard<std::mutex> lock(wait_mutex_);
        wait_cv_.notify_all();
    }
}

}  // namespace rtp_llm
