#pragma once

#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <vector>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class MultiRankBlockTransferEngine;
class PerRankBlockTransferEngine;
struct BlockTreeQueueSizes;

namespace block_tree_cache_test {
class BlockTreeCacheTestPeer;
}

class BlockTransferDispatcher {
public:
    using TransferDoneCallback = std::function<void(ErrorInfo)>;

    BlockTransferDispatcher(std::shared_ptr<PerRankBlockTransferEngine>   per_rank_engine,
                            std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine = nullptr,
                            size_t                                        max_descriptors_per_batch = 8);
    ~BlockTransferDispatcher();

    std::shared_ptr<AsyncContext> executePerRank(TransferTask task) const;

    // Callback-driven path used by Load/Store/Evict: stable grouping and bounded batches.
    void runTransfer(TransferTask task, TransferDoneCallback callback) const;

    void                cancelPendingStagingTransfers() const;
    BlockTreeQueueSizes queueSizes() const;

    // Lifecycle-only drain, not task admission or a request-level sync API.
    // Call after producers stop; callbacks can still submit a subsequent stage.
    void drainTransfers() const;

private:
    friend class block_tree_cache_test::BlockTreeCacheTestPeer;

    std::shared_ptr<AsyncContext> executeMultiRank(TransferTask task) const;

    std::shared_ptr<PerRankBlockTransferEngine>   per_rank_engine_;
    std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine_;
    size_t                                        max_descriptors_per_batch_{8};
    mutable std::mutex                            completion_mutex_;
    mutable std::vector<std::shared_future<void>>  transfer_completions_;
    std::function<void()>                         drain_observer_for_test_;
};

using BlockTransferDispatcherPtr = std::shared_ptr<BlockTransferDispatcher>;

}  // namespace rtp_llm
