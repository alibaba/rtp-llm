#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/StorageBackendExecutor.h"

#include <utility>
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"

namespace rtp_llm {
namespace {

class DefaultStorageBackendExecutor final: public StorageBackendExecutor {
public:
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
    BlockTreeTaskPool pool_{4, 1024, "StorageBackendExecutor"};
};

}  // namespace

std::shared_ptr<StorageBackendExecutor> makeDefaultStorageBackendExecutor() {
    return std::make_shared<DefaultStorageBackendExecutor>();
}

}  // namespace rtp_llm
