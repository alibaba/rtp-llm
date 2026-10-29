#pragma once

#include <atomic>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

namespace block_tree_cache_detail {
// Fault seam for a single publication-copy failure; null in production.
inline std::atomic<void (*)()> before_transfer_publication_copy_for_test{nullptr};
}  // namespace block_tree_cache_detail

class BroadcastManager;
class BlockTreeCacheMetricsReporter;

class MultiRankBlockTransferEngine {
public:
    MultiRankBlockTransferEngine(std::vector<GroupSetPtr>                       group_sets,
                                 std::shared_ptr<BroadcastManager>              broadcast_manager,
                                 std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter = nullptr);

    std::shared_ptr<TransferAsyncContext> execute(TransferTask task) const;

private:
    friend class BlockTransferDispatcher;

    std::vector<GroupSetPtr>                       group_sets_;
    std::shared_ptr<BroadcastManager>              broadcast_manager_;
    std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter_;
};

using MultiRankBlockTransferEnginePtr = std::shared_ptr<MultiRankBlockTransferEngine>;

}  // namespace rtp_llm
