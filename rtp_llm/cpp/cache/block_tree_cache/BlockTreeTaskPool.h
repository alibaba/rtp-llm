#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

namespace autil {
class LockFreeThreadPool;
}

namespace rtp_llm {

enum class BlockTreeTaskClass {
    LOAD,
    BACKGROUND,
};

struct BlockTreeQueueSizes {
    size_t load{0};
    size_t background{0};
    size_t completion{0};
};

class BlockTreeTaskPool: public std::enable_shared_from_this<BlockTreeTaskPool> {
public:
    using Clock = std::chrono::steady_clock;

    static constexpr size_t                    kDefaultQueueSize = 10000;
    static constexpr std::chrono::milliseconds kDefaultQueueWaitTimeout{30000};
    // Normal-queue slots only LOAD tasks may occupy, so
    // loads remain admissible while BACKGROUND transfers are queued or awaiting
    // asynchronous settlement. Skipped when queue_size does not exceed it, so
    // small pools never starve BACKGROUND.
    static constexpr size_t kLoadReservedSlots = 64;

    BlockTreeTaskPool(size_t thread_count, size_t queue_size, std::string thread_name,
                      size_t completion_capacity = 0);
    class Completion;
    struct CompletionSnapshot {
        size_t capacity, outstanding, queued_load, queued_background, running, rejected;
        bool accepting;
    };
    std::shared_ptr<Completion> reserveCompletion(std::function<void()> task, BlockTreeTaskClass task_class);
    void stopCompletionAdmission();
    void waitForCompletionIdle();
    CompletionSnapshot completionSnapshot() const;
    ~BlockTreeTaskPool();

    BlockTreeTaskPool(const BlockTreeTaskPool&)            = delete;
    BlockTreeTaskPool& operator=(const BlockTreeTaskPool&) = delete;

    bool start();
    bool submit(BlockTreeTaskClass               task_class,
                std::function<void()>            task,
                std::optional<Clock::time_point> deadline   = std::nullopt,
                std::function<void()>            on_timeout = {});
    bool submitCompletion(std::function<void()> task);
    void stopAdmission();
    // Wait only for queued/running task bodies, including submitted completions.
    // External asynchronous transfers must be drained by their owner.
    void waitForIdle();
    void shutdown();

    BlockTreeQueueSizes queueSizes() const;

private:
    static constexpr size_t kMaxLoadBurst = 4;

    struct QueuedTask {
        std::function<void()>            run;
        std::function<void()>            on_timeout;
        std::optional<Clock::time_point> deadline;
    };

public:
    // One pre-I/O callback record, charged by the existing outstanding counter.
    // It is itself the queue node; no ticket or separate phase machine.
    class Completion: public std::enable_shared_from_this<Completion> {
        friend class BlockTreeTaskPool;
        std::shared_ptr<BlockTreeTaskPool> pool_;
        std::function<void()> task_;
        std::shared_ptr<Completion> next_;
        const bool load_;
        bool admitted_{false};
        std::atomic<bool> posted_{false};
    public:
        // Only reserveCompletion arms admission. A directly constructed record
        // cannot post or return capacity that it never acquired.
        Completion(std::shared_ptr<BlockTreeTaskPool> pool, std::function<void()> task, bool load):
            pool_(std::move(pool)), task_(std::move(task)), load_(load) {}
        Completion(const Completion&) = delete;
        Completion& operator=(const Completion&) = delete;
        ~Completion() { if (admitted_ && !posted_.load()) pool_->abandonCompletion(*this); }
        // The caller must also hold an independent pool/executor owner through
        // post(). Production makeIoCompletion captures that executor strongly;
        // otherwise dropping the last pool owner here could join this worker.
        bool post() noexcept {
            if (!admitted_) return false;
            auto pool = std::atomic_load(&pool_);
            if (!pool) return false;
            auto self = shared_from_this();
            const bool accepted = pool->postCompletion(self);
            // A queued record must not create pool -> record -> pool ownership.
            // The posting caller keeps pool/self alive through this handoff;
            // executor ownership and shutdown draining cover the queued work.
            if (accepted) std::atomic_store(&pool_, std::shared_ptr<BlockTreeTaskPool>{});
            return accepted;
        }
    };
private:
    bool postCompletion(const std::shared_ptr<Completion>& node) noexcept;
    void abandonCompletion(Completion& node) noexcept;
    void       workerLoop();
    size_t     backgroundLimit() const;
    size_t     normalQueueSizeLocked() const;
    QueuedTask popNextNormalTaskLocked();
    void       taskStarted();
    void       taskFinished();

    const size_t      thread_count_;
    const size_t      queue_size_;
    const size_t      completion_capacity_;
    const std::string thread_name_;

    std::shared_ptr<autil::LockFreeThreadPool> thread_pool_;
    mutable std::mutex                         lifecycle_mutex_;
    std::condition_variable                    queue_cv_, completion_idle_;
    std::shared_ptr<Completion> completed_load_, completed_background_;
    Completion *completed_load_tail_{nullptr}, *completed_background_tail_{nullptr};
    size_t completion_outstanding_{0}, completion_load_{0}, completion_background_{0}, completion_running_{0};
    size_t completion_rejected_{0}, completion_burst_{0};
    bool completion_admission_stopped_{false};
    std::mutex shutdown_mutex_;
    std::deque<QueuedTask>                     load_queue_;
    std::deque<QueuedTask>                     background_queue_;
    std::deque<std::function<void()>>          completion_queue_;
    size_t                                     consecutive_load_dispatches_{0};
    bool                                       started_{false};
    bool                                       admission_stopped_{false};
    bool                                       shutdown_{false};

    std::atomic<int>        pending_tasks_{0};
    std::mutex              wait_mutex_;
    std::condition_variable wait_cv_;
    std::function<void()>   pending_task_wait_observer_for_test_;
};

}  // namespace rtp_llm
