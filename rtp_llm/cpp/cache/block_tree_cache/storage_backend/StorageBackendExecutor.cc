#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackendExecutor.h"

#include <stdexcept>
#include <mutex>
#include <utility>
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"

namespace rtp_llm {
namespace {

class DefaultStorageBackendExecutor final: public StorageBackendExecutor {
public:
    DefaultStorageBackendExecutor(size_t thread_count, size_t queue_size):
        pool_(thread_count, queue_size, "StorageBackendExecutor") {}

    ~DefaultStorageBackendExecutor() override {
        shutdown();
    }

    bool start() override {
        return pool_.start();
    }
    bool submit(Task task) override {
        return pool_.submit(BlockTreeTaskClass::BACKGROUND, std::move(task));
    }
    void shutdown() noexcept override {
        std::lock_guard<std::mutex> lock(shutdown_mutex_);
        // The task pool's shutdown discards queued work. Settle every accepted
        // storage operation before stopping workers and releasing backend state.
        pool_.stopAdmission();
        pool_.waitForIdle();
        pool_.shutdown();
    }

private:
    BlockTreeTaskPool pool_;
    std::mutex        shutdown_mutex_;
};

}  // namespace

std::shared_ptr<StorageBackendExecutor> makeStorageBackendExecutor(size_t thread_count, size_t queue_size) {
    if (thread_count == 0 || queue_size == 0) {
        throw std::invalid_argument("StorageBackendExecutor thread count and queue size must be positive");
    }
    return std::make_shared<DefaultStorageBackendExecutor>(thread_count, queue_size);
}

std::shared_ptr<StorageBackendExecutor> makeDefaultStorageBackendExecutor() {
    return makeStorageBackendExecutor(/*thread_count=*/4, /*queue_size=*/1024);
}

}  // namespace rtp_llm
