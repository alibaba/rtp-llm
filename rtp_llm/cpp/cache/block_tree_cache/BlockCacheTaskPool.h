#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

namespace autil {
class LockFreeThreadPool;
}

namespace rtp_llm {

class BlockCacheTaskPool {
public:
    BlockCacheTaskPool(size_t thread_count, size_t queue_size, std::string thread_name);
    ~BlockCacheTaskPool();

    BlockCacheTaskPool(const BlockCacheTaskPool&)            = delete;
    BlockCacheTaskPool& operator=(const BlockCacheTaskPool&) = delete;

    bool start();
    bool submit(std::function<void()> task);
    void waitForIdle();
    void shutdown();

private:
    void taskStarted();
    void taskFinished();

    const size_t      thread_count_;
    const size_t      queue_size_;
    const std::string thread_name_;

    std::shared_ptr<autil::LockFreeThreadPool> thread_pool_;
    std::mutex                                 lifecycle_mutex_;
    bool                                       started_{false};
    bool                                       shutdown_{false};

    std::atomic<int>        pending_tasks_{0};
    std::mutex              wait_mutex_;
    std::condition_variable wait_cv_;
    std::function<void()>   pending_task_wait_observer_for_test_;
};

}  // namespace rtp_llm
