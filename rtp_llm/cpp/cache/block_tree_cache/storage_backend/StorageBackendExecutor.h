#pragma once

#include <cstddef>
#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
namespace rtp_llm {

// submit returns true iff the task was accepted for exactly-once execution.
// shutdown concurrently stops admission and drains every accepted task.
class StorageBackendExecutor {
public:
    using Task                        = std::function<void()>;
    using Clock                       = BlockTreeTaskPool::Clock;
    virtual ~StorageBackendExecutor() = default;
    virtual bool start()              = 0;
    virtual bool submit(Task task, BlockTreeTaskClass task_class = BlockTreeTaskClass::BACKGROUND) = 0;
    // Storage operations check their own deadline before I/O. A deadline-aware
    // queue can run an already-expired operation immediately to release its pins.
    virtual bool submitWithDeadline(Task task, BlockTreeTaskClass task_class, std::optional<Clock::time_point>) {
        return submit(std::move(task), task_class);
    }
    virtual bool submitCompletion(Task task) {
        return submit(std::move(task), BlockTreeTaskClass::LOAD);
    }
    virtual BlockTreeQueueSizes queueSizes() const { return {}; }
    // Async executors leave the completion queue open while the backend drains
    // external I/O. Synchronous executors can finish their queued work here.
    virtual void stopAdmission() noexcept { shutdown(); }
    virtual void shutdown() noexcept = 0;

private:
    friend class StorageBackend;
    // Executors cannot be restarted or shared by multiple backend lifecycles.
    std::atomic<bool> bound_to_backend_{false};
};
std::shared_ptr<StorageBackendExecutor> makeStorageBackendExecutor(size_t thread_count, size_t queue_size);
std::shared_ptr<StorageBackendExecutor> makeDefaultStorageBackendExecutor();

}  // namespace rtp_llm
