#pragma once

#include <cstddef>
#include <functional>
#include <memory>
namespace rtp_llm {

// submit returns true iff the task was accepted for exactly-once execution.
// shutdown concurrently stops admission and drains every accepted task.
class StorageBackendExecutor {
public:
    using Task                        = std::function<void()>;
    virtual ~StorageBackendExecutor() = default;
    virtual bool start()              = 0;
    virtual bool submit(Task task)    = 0;
    virtual void shutdown() noexcept  = 0;
};
std::shared_ptr<StorageBackendExecutor> makeStorageBackendExecutor(size_t thread_count, size_t queue_size);
std::shared_ptr<StorageBackendExecutor> makeDefaultStorageBackendExecutor();

}  // namespace rtp_llm
