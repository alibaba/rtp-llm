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

namespace autil {
class LockFreeThreadPool;
}

namespace rtp_llm {

enum class BlockTreeTaskClass {
    LOAD,
    BACKGROUND,
};

class BlockTreeTaskPool {
public:
    static constexpr size_t                    kDefaultQueueSize = 10000;
    static constexpr std::chrono::milliseconds kDefaultQueueWaitTimeout{30000};
    // Normal-queue slots only LOAD tasks may occupy, so loads can still enqueue
    // when BACKGROUND transfers flood the pool. Skipped when queue_size does not
    // exceed it, so small pools never starve BACKGROUND.
    static constexpr size_t kLoadReservedSlots = 64;

    BlockTreeTaskPool(size_t thread_count, size_t queue_size, std::string thread_name);
    ~BlockTreeTaskPool();

    BlockTreeTaskPool(const BlockTreeTaskPool&)            = delete;
    BlockTreeTaskPool& operator=(const BlockTreeTaskPool&) = delete;

    bool start();
    bool submit(std::function<void()>     task,
                std::chrono::milliseconds max_queue_wait = std::chrono::milliseconds::zero(),
                std::function<void()>     on_timeout     = {});
    bool submit(BlockTreeTaskClass        task_class,
                std::function<void()>     task,
                std::chrono::milliseconds max_queue_wait = std::chrono::milliseconds::zero(),
                std::function<void()>     on_timeout     = {});
    bool submitCompletion(std::function<void()> task);
    void stopAdmission();
    void waitForIdle();
    void shutdown();

private:
    static constexpr size_t kMaxLoadBurst = 4;

    struct QueuedTask {
        std::function<void()>                                run;
        std::function<void()>                                on_timeout;
        std::optional<std::chrono::steady_clock::time_point> deadline;
    };

    void       workerLoop();
    size_t     normalQueueSizeLocked() const;
    QueuedTask popNextNormalTaskLocked();
    void       taskStarted();
    void       taskFinished();

    const size_t      thread_count_;
    const size_t      queue_size_;
    const std::string thread_name_;

    std::shared_ptr<autil::LockFreeThreadPool> thread_pool_;
    std::mutex                                 lifecycle_mutex_;
    std::condition_variable                    queue_cv_;
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
