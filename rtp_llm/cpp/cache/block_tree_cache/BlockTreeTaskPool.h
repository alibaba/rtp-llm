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

class BlockTreeTaskPool {
public:
    static constexpr size_t kDefaultQueueSize = 10000;

    BlockTreeTaskPool(size_t thread_count, size_t queue_size, std::string thread_name);
    ~BlockTreeTaskPool();

    BlockTreeTaskPool(const BlockTreeTaskPool&)            = delete;
    BlockTreeTaskPool& operator=(const BlockTreeTaskPool&) = delete;

    bool start();
    bool submit(std::function<void()>      task,
                std::chrono::milliseconds max_queue_wait = std::chrono::milliseconds::zero(),
                std::function<void()>      on_timeout     = {});
    bool submitCompletion(std::function<void()> task);
    bool startBusiness();
    void finishBusiness();
    void stopAdmission();
    void waitForIdle();
    void shutdown();

private:
    struct QueuedTask {
        std::function<void()>                                      run;
        std::function<void()>                                      on_timeout;
        std::optional<std::chrono::steady_clock::time_point>       deadline;
    };

    void workerLoop();
    void taskStarted();
    void taskFinished();

    const size_t      thread_count_;
    const size_t      queue_size_;
    const std::string thread_name_;

    std::shared_ptr<autil::LockFreeThreadPool> thread_pool_;
    std::mutex                                 lifecycle_mutex_;
    std::condition_variable                   queue_cv_;
    std::deque<QueuedTask>                     normal_queue_;
    std::deque<std::function<void()>>          completion_queue_;
    bool                                       started_{false};
    bool                                       admission_stopped_{false};
    bool                                       shutdown_{false};

    std::atomic<int>        pending_tasks_{0};
    std::atomic<size_t>     active_businesses_{0};
    std::mutex              wait_mutex_;
    std::condition_variable wait_cv_;
    std::function<void()>   pending_task_wait_observer_for_test_;
};

}  // namespace rtp_llm
