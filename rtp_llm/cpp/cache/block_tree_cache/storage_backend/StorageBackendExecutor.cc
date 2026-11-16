#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackendExecutor.h"

#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace rtp_llm {
namespace {

class DefaultStorageBackendExecutor final: public StorageBackendExecutor {
public:
    DefaultStorageBackendExecutor(size_t thread_count, size_t queue_size, bool reserve_completion):
        pool_(std::make_shared<BlockTreeTaskPool>(thread_count, queue_size, "remote_cache",
                                                reserve_completion ? queue_size : 0)),
        has_completion_reservations_(reserve_completion && queue_size != 0) {}

    ~DefaultStorageBackendExecutor() override {
        shutdown();
    }

    bool start() override {
        std::lock_guard<std::mutex> lock(shutdown_mutex_);
        return pool_->start();
    }

    bool submit(Task task, BlockTreeTaskClass task_class) override {
        return pool_->submit(task_class, std::move(task));
    }

    bool submitWithDeadline(Task task, BlockTreeTaskClass task_class, std::optional<Clock::time_point> deadline) override {
        Task on_timeout = deadline ? task : Task{};
        return pool_->submit(task_class, std::move(task), deadline, std::move(on_timeout));
    }

    bool submitCompletion(Task task) override {
        return pool_->submitCompletion(std::move(task));
    }

    bool hasCompletionReservations() const noexcept override { return has_completion_reservations_; }
    std::shared_ptr<BlockTreeTaskPool::Completion>
    reserveCompletion(Task task, BlockTreeTaskClass task_class) override {
        return pool_->reserveCompletion(std::move(task), task_class);
    }

    BlockTreeQueueSizes queueSizes() const override {
        return pool_->queueSizes();
    }

    void stopAdmission() noexcept override {
        pool_->stopAdmission();
        // Accepted queued tasks reserve their completion before starting I/O.
        // Keep that admission open until the business pool has drained; pool
        // shutdown then closes reservations and drains all accepted results.
    }

    void shutdown() noexcept override {
        std::lock_guard<std::mutex> lock(shutdown_mutex_);
        stopAdmission();
        pool_->waitForIdle();
        pool_->shutdown();
    }

private:
    std::shared_ptr<BlockTreeTaskPool> pool_;
    const bool has_completion_reservations_; // Immutable capability; reserveCompletion still checks admission.
    std::mutex        shutdown_mutex_;
};

}  // namespace

std::shared_ptr<StorageBackendExecutor> makeStorageBackendExecutor(size_t thread_count, size_t queue_size,
                                                                  bool reserve_completion) {
    if (thread_count == 0) {
        throw std::invalid_argument("StorageBackendExecutor thread count must be positive");
    }
    // Preserve upstream LOAD headroom on both roles; only Decode requires
    // pre-I/O completion reservations.
    constexpr size_t reserve = BlockTreeTaskPool::kLoadReservedSlots;
    if (queue_size > std::numeric_limits<size_t>::max() - reserve) {
        throw std::invalid_argument("StorageBackendExecutor queue size is too large");
    }
    return std::make_shared<DefaultStorageBackendExecutor>(thread_count, queue_size == 0 ? 0 : queue_size + reserve, reserve_completion);
}

std::shared_ptr<StorageBackendExecutor> makeDefaultStorageBackendExecutor() {
    return makeStorageBackendExecutor(/*thread_count=*/4, BlockTreeTaskPool::kDefaultQueueSize);
}

}  // namespace rtp_llm
