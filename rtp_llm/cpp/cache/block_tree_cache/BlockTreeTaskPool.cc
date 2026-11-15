#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"

#include <cassert>
#include <algorithm>
#include <utility>
#include <stdexcept>
#include <limits>

#include "autil/LambdaWorkItem.h"
#include "autil/LockFreeThreadPool.h"

namespace rtp_llm {

BlockTreeTaskPool::BlockTreeTaskPool(size_t thread_count, size_t queue_size, std::string thread_name,
                                     size_t completion_capacity):
    thread_count_(thread_count), queue_size_(queue_size),
    completion_capacity_(completion_capacity), thread_name_(std::move(thread_name)) {}

BlockTreeTaskPool::~BlockTreeTaskPool() {
    shutdown();
}

bool BlockTreeTaskPool::start() {
    std::unique_lock<std::mutex> lock(lifecycle_mutex_);
    if (started_ || shutdown_ || thread_count_ == 0) {
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
        auto*                               work_item = new autil::LambdaWorkItem([this] { workerLoop(); });
        const autil::ThreadPool::ERROR_TYPE error     = thread_pool->pushWorkItem(work_item, false);
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

bool BlockTreeTaskPool::submit(BlockTreeTaskClass               task_class,
                               std::function<void()>            task,
                               std::optional<Clock::time_point> deadline,
                               std::function<void()>            on_timeout) {
    if (!task || deadline.has_value() != static_cast<bool>(on_timeout)) {
        return false;
    }

    std::unique_lock<std::mutex> lock(lifecycle_mutex_);
    if (!started_ || admission_stopped_ || shutdown_) {
        return false;
    }
    if (deadline.has_value() && Clock::now() >= *deadline) {
        lock.unlock();
        try {
            on_timeout();
        } catch (...) {}
        return true;
    }
    if (queue_size_ != 0) {
        if (normalQueueSizeLocked() >= queue_size_) {
            return false;
        }
        if (task_class == BlockTreeTaskClass::BACKGROUND && background_queue_.size() >= backgroundLimit()) {
            return false;
        }
    }

    auto& queue = task_class == BlockTreeTaskClass::LOAD ? load_queue_ : background_queue_;
    queue.push_back(QueuedTask{std::move(task), std::move(on_timeout), std::move(deadline)});
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

void BlockTreeTaskPool::stopAdmission() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    admission_stopped_ = true;
}

size_t BlockTreeTaskPool::backgroundLimit() const {
    return queue_size_ > kLoadReservedSlots ? queue_size_ - kLoadReservedSlots : queue_size_;
}

BlockTreeQueueSizes BlockTreeTaskPool::queueSizes() const {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    return {load_queue_.size(), background_queue_.size(), completion_queue_.size() + completion_load_ + completion_background_};
}

size_t BlockTreeTaskPool::normalQueueSizeLocked() const {
    return load_queue_.size() + background_queue_.size();
}

BlockTreeTaskPool::QueuedTask BlockTreeTaskPool::popNextNormalTaskLocked() {
    const bool take_background =
        !background_queue_.empty() && (load_queue_.empty() || consecutive_load_dispatches_ >= kMaxLoadBurst);
    if (take_background) {
        QueuedTask task = std::move(background_queue_.front());
        background_queue_.pop_front();
        consecutive_load_dispatches_ = 0;
        return task;
    }

    QueuedTask task = std::move(load_queue_.front());
    load_queue_.pop_front();
    consecutive_load_dispatches_ = std::min(consecutive_load_dispatches_ + 1, kMaxLoadBurst);
    return task;
}

std::shared_ptr<BlockTreeTaskPool::Completion>
BlockTreeTaskPool::reserveCompletion(std::function<void()> task, BlockTreeTaskClass task_class) {
    if (!task) return {};
    // Allocate the callback record before charging; failed construction cannot
    // debit an existing obligation or recursively return unacquired capacity.
    auto node = std::make_shared<Completion>(shared_from_this(), std::move(task), task_class == BlockTreeTaskClass::LOAD);
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    const size_t limit = node->load_ ? completion_capacity_ :
        completion_capacity_ - std::min(kLoadReservedSlots, completion_capacity_ / 2);
    if (!started_ || shutdown_ || completion_admission_stopped_ || completion_outstanding_ >= limit) {
        ++completion_rejected_;
        return {};
    }
    node->admitted_ = true;
    ++completion_outstanding_;
    return node;
}
bool BlockTreeTaskPool::postCompletion(const std::shared_ptr<Completion>& node) noexcept {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    if (!node->admitted_ || node->posted_.exchange(true)) return false;
    auto& head = node->load_ ? completed_load_ : completed_background_;
    auto*& tail = node->load_ ? completed_load_tail_ : completed_background_tail_;
    if (tail) tail->next_ = node; else head = node;
    tail = node.get();
    ++(node->load_ ? completion_load_ : completion_background_);
    taskStarted(); queue_cv_.notify_one();
    return true;
}
void BlockTreeTaskPool::abandonCompletion(Completion& node) noexcept {
    // Only the last owner of an unposted record reaches here. Release its
    // callback owners outside the queue lock before returning capacity.
    auto retired = std::move(node.task_);
    retired = {};
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    --completion_outstanding_; completion_idle_.notify_all();
}
void BlockTreeTaskPool::stopCompletionAdmission() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_); completion_admission_stopped_ = true;
}
void BlockTreeTaskPool::waitForCompletionIdle() {
    std::unique_lock<std::mutex> lock(lifecycle_mutex_);
    completion_idle_.wait(lock, [this] { return completion_outstanding_ == 0; });
}
BlockTreeTaskPool::CompletionSnapshot BlockTreeTaskPool::completionSnapshot() const {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    return {completion_capacity_, completion_outstanding_, completion_load_, completion_background_, completion_running_,
            completion_rejected_, started_ && !shutdown_ && !completion_admission_stopped_};
}
void BlockTreeTaskPool::workerLoop() {
    for (;;) {
        std::function<void()> task;
        QueuedTask            queued_task;
        std::shared_ptr<Completion> reserved;
        {
            std::unique_lock<std::mutex> lock(lifecycle_mutex_);
            const auto can_reserve = [this] {
                return completed_load_ || completed_background_;
            };
            queue_cv_.wait(lock, [&] {
                return shutdown_ || can_reserve() || !completion_queue_.empty()
                    || normalQueueSizeLocked();
            });
            if (can_reserve()) {
                const bool load = completed_load_ && (!completed_background_ || completion_burst_ < kMaxLoadBurst);
                auto& head = load ? completed_load_ : completed_background_;
                auto*& tail = load ? completed_load_tail_ : completed_background_tail_;
                reserved = std::move(head); head = std::move(reserved->next_); if (!head) tail = nullptr;
                --(load ? completion_load_ : completion_background_);
                completion_burst_ = load ? std::min(completion_burst_ + 1, kMaxLoadBurst) : 0;
                ++completion_running_;
                task.swap(reserved->task_);
            } else if (!completion_queue_.empty()) {
                task = std::move(completion_queue_.front()); completion_queue_.pop_front();
            } else if (normalQueueSizeLocked()) {
                queued_task = popNextNormalTaskLocked();
                task = queued_task.deadline && Clock::now() >= *queued_task.deadline ? std::move(queued_task.on_timeout)
                                                                                     : std::move(queued_task.run);
            } else if (shutdown_) return;
        }
        try { task(); } catch (...) {}
        // Captured staging leases can dispatch another task on destruction.
        // Release both selected and discarded callbacks outside lifecycle_mutex_,
        // before publishing idle so their follow-up submissions are accounted for.
        task = {};
        queued_task = {};
        taskFinished();
        if (reserved) {
            std::lock_guard<std::mutex> lock(lifecycle_mutex_);
            --completion_running_; --completion_outstanding_; completion_idle_.notify_all();
        }
    }
}

void BlockTreeTaskPool::waitForIdle() {
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

void BlockTreeTaskPool::shutdown() {
    std::lock_guard<std::mutex> shutdown_lock(shutdown_mutex_);
    if (completion_capacity_) {
        stopAdmission();
        waitForIdle(); // Accepted queued business operations may still reserve.
        stopCompletionAdmission();
        waitForCompletionIdle();
        waitForIdle();
    }
    std::shared_ptr<autil::LockFreeThreadPool> thread_pool;
    std::deque<QueuedTask>                    dropped_load;
    std::deque<QueuedTask>                    dropped_background;
    std::deque<std::function<void()>>         dropped_completions;
    bool                                       was_started   = false;
    size_t                                     dropped_tasks = 0;
    {
        std::lock_guard<std::mutex> lock(lifecycle_mutex_);
        if (shutdown_) {
            return;
        }
        admission_stopped_ = true;
        shutdown_          = true;
        dropped_tasks      = normalQueueSizeLocked() + completion_queue_.size();
        dropped_load.swap(load_queue_);
        dropped_background.swap(background_queue_);
        dropped_completions.swap(completion_queue_);
        thread_pool = thread_pool_;
        was_started = started_;
        queue_cv_.notify_all();
    }

    // Lease destruction may try to submit follow-up work, which shutdown rejects.
    dropped_load.clear();
    dropped_background.clear();
    dropped_completions.clear();

    if (dropped_tasks > 0) {
        const int dropped   = static_cast<int>(dropped_tasks);
        const int remaining = pending_tasks_.fetch_sub(dropped) - dropped;
        if (remaining <= 0) {
            std::lock_guard<std::mutex> lock(wait_mutex_);
            wait_cv_.notify_all();
        }
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
