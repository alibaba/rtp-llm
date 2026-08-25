#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

namespace autil {
class LockFreeThreadPool;
}

namespace rtp_llm {

class BlockTreeTaskPool {
public:
    BlockTreeTaskPool(size_t thread_count, size_t queue_size, std::string thread_name);
    ~BlockTreeTaskPool();

    BlockTreeTaskPool(const BlockTreeTaskPool&)            = delete;
    BlockTreeTaskPool& operator=(const BlockTreeTaskPool&) = delete;

    bool start();
    bool submit(std::function<void()> task);
    bool submitCompletion(std::function<void()> task);
    bool acquireBusinessCredit();
    void releaseBusinessCredit();
    void stopAdmission();
    void waitForIdle();
    void shutdown();

private:
    void workerLoop();
    void taskStarted();
    void taskFinished();

    const size_t      thread_count_;
    const size_t      queue_size_;
    const std::string thread_name_;

    std::shared_ptr<autil::LockFreeThreadPool> thread_pool_;
    std::mutex                                 lifecycle_mutex_;
    std::condition_variable                   queue_cv_;
    std::deque<std::function<void()>>          normal_queue_;
    std::deque<std::function<void()>>          completion_queue_;
    bool                                       started_{false};
    bool                                       admission_stopped_{false};
    bool                                       shutdown_{false};

    std::atomic<int>        pending_tasks_{0};
    std::atomic<size_t>     business_credits_{0};
    std::mutex              wait_mutex_;
    std::condition_variable wait_cv_;
    std::function<void()>   pending_task_wait_observer_for_test_;
};

}  // namespace rtp_llm
