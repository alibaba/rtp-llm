#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackendExecutor.h"

#include <stdexcept>
#include <mutex>
#include <utility>
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

class DefaultStorageBackendExecutor final: public StorageBackendExecutor {
public:
    DefaultStorageBackendExecutor(size_t thread_count, size_t queue_size):
        pool_(thread_count, queue_size, "StorageBackendExecutor"),
        background_limit_(thread_count > 1 ? thread_count - 1 : 1) {}

    ~DefaultStorageBackendExecutor() override {
        shutdown();
    }

    bool start() override {
        return pool_.start();
    }
    bool submit(Task task, BlockTreeTaskClass task_class) override {
        if (!task) {
            return false;
        }
        if (task_class == BlockTreeTaskClass::LOAD) {
            return pool_.submit(task_class, std::move(task));
        }
        if (background_tasks_.fetch_add(1) >= background_limit_) {
            background_tasks_.fetch_sub(1);
            RTP_LLM_INTERVAL_LOG(
                5, WARN, "StorageBackend background task limit reached: limit=%zu", background_limit_);
            return false;
        }
        block_tree_cache_detail::ScopeRollback rejected([this] { background_tasks_.fetch_sub(1); });
        if (!pool_.submit(task_class, [this, task = std::move(task)] {
                // Reserve a worker for LOAD until the actual background I/O retires.
                block_tree_cache_detail::ScopeRollback finished([this] { background_tasks_.fetch_sub(1); });
                task();
            })) {
            return false;
        }
        rejected.dismiss();
        return true;
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
    BlockTreeTaskPool   pool_;
    std::mutex          shutdown_mutex_;
    const size_t        background_limit_;
    std::atomic<size_t> background_tasks_{0};
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
