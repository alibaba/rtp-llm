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
// A bounded test barrier after publication ownership is claimed; null in production.
inline std::atomic<void (*)() noexcept> after_transfer_completion_claim_for_test{nullptr};
}  // namespace block_tree_cache_detail

class BroadcastManager;
class BlockTreeCacheMetricsReporter;

class MultiRankBlockTransferEngine {
public:
    MultiRankBlockTransferEngine(std::vector<GroupSetPtr>                       group_sets,
                                 std::shared_ptr<BroadcastManager>              broadcast_manager,
                                 std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter = nullptr);

    std::shared_ptr<TransferAsyncContext> execute(TransferTask task, const char* completion_operation = nullptr) const;

private:
    friend class BlockTransferDispatcher;

    std::vector<GroupSetPtr>                       group_sets_;
    std::shared_ptr<BroadcastManager>              broadcast_manager_;
    std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter_;
};

using MultiRankBlockTransferEnginePtr = std::shared_ptr<MultiRankBlockTransferEngine>;

}  // namespace rtp_llm
