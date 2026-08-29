#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackendExecutor.h"

#include <stdexcept>
#include <utility>
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"

namespace rtp_llm {
namespace {

class DefaultStorageBackendExecutor final: public StorageBackendExecutor {
public:
    DefaultStorageBackendExecutor(size_t thread_count, size_t queue_size):
        pool_(thread_count, queue_size, "StorageBackendExecutor") {}

    bool start() override {
        return pool_.start();
    }
    bool submit(Task task) override {
        return pool_.submit(std::move(task));
    }
    void shutdown() noexcept override {
        pool_.shutdown();
    }

private:
    BlockTreeTaskPool pool_;
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
