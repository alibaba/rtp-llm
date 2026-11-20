#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackendExecutor.h"

#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace rtp_llm {
namespace {

class DefaultStorageBackendExecutor final: public StorageBackendExecutor {
public:
    DefaultStorageBackendExecutor(size_t thread_count, size_t queue_size):
        pool_(thread_count, queue_size, "remote_cache") {}

    ~DefaultStorageBackendExecutor() override {
        shutdown();
    }

    bool start() override {
        std::lock_guard<std::mutex> lock(shutdown_mutex_);
        return pool_.start();
    }

    bool submit(Task task, BlockTreeTaskClass task_class) override {
        return pool_.submit(task_class, std::move(task));
    }

    bool submitWithDeadline(Task task, BlockTreeTaskClass task_class, std::optional<Clock::time_point> deadline) override {
        Task on_timeout = deadline ? task : Task{};
        return pool_.submit(task_class, std::move(task), deadline, std::move(on_timeout));
    }

    bool submitCompletion(Task task) override {
        return pool_.submitCompletion(std::move(task));
    }

    BlockTreeQueueSizes queueSizes() const override {
        return pool_.queueSizes();
    }

    void stopAdmission() noexcept override {
        pool_.stopAdmission();
    }

    void shutdown() noexcept override {
        std::lock_guard<std::mutex> lock(shutdown_mutex_);
        stopAdmission();
        pool_.waitForIdle();
        pool_.shutdown();
    }

private:
    BlockTreeTaskPool pool_;
    std::mutex        shutdown_mutex_;
};

}  // namespace

std::shared_ptr<StorageBackendExecutor> makeStorageBackendExecutor(size_t thread_count, size_t queue_size) {
    if (thread_count == 0) {
        throw std::invalid_argument("StorageBackendExecutor thread count must be positive");
    }
    // Match the BTC business pool: background capacity plus reserved LOAD slots.
    constexpr size_t reserve = BlockTreeTaskPool::kLoadReservedSlots;
    if (queue_size > std::numeric_limits<size_t>::max() - reserve) {
        throw std::invalid_argument("StorageBackendExecutor queue size is too large");
    }
    return std::make_shared<DefaultStorageBackendExecutor>(thread_count, queue_size == 0 ? 0 : queue_size + reserve);
}

std::shared_ptr<StorageBackendExecutor> makeDefaultStorageBackendExecutor() {
    return makeStorageBackendExecutor(/*thread_count=*/4, BlockTreeTaskPool::kDefaultQueueSize);
}

}  // namespace rtp_llm
