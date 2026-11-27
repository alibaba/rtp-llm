#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferCompletionTypes.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

namespace block_tree_cache_detail {
// Fault seam for a single publication-copy failure; null in production.
inline std::atomic<void (*)()> before_transfer_publication_copy_for_test{nullptr};
// A bounded test barrier after publication ownership is claimed; null in production.
inline std::atomic<void (*)() noexcept> after_transfer_completion_claim_for_test{nullptr};
// Registration can fail after RPC submission; its result must remain UNKNOWN.
inline std::atomic<void (*)()> before_transfer_broadcast_subscription_for_test{nullptr};
}  // namespace block_tree_cache_detail

class BroadcastManager;
class BlockTreeCacheMetricsReporter;
class CacheObserver;

struct RankTransferReply {
    TransferRankSnapshot          snapshot;
    std::optional<TransferResult> result;
};

// A delivery mechanism supplies the rank observations. The multi-rank engine
// owns descriptor-wise reduction, diagnostics and publication of the result.
class RankTransferBroadcast {
public:
    using DoneCallback                                                   = std::function<void()>;
    virtual ~RankTransferBroadcast()                                     = default;
    virtual void                           waitDone()                    = 0;
    virtual void                           onDone(DoneCallback callback) = 0;
    virtual bool                           done() const                  = 0;
    virtual std::string                    failureSummary() const        = 0;
    virtual std::vector<RankTransferReply> replies() const               = 0;
    virtual std::vector<int64_t>           completionTimesUs() const {
        return {};
    }
    virtual std::vector<int64_t> completionWallTimesUs() const {
        return {};
    }
};

struct RankTransferRequest {
    std::vector<TransferCompletionPeer>                                              peers;
    std::function<std::shared_ptr<RankTransferBroadcast>(std::chrono::milliseconds)> send;
};

struct RankTransferTransport {
    std::vector<std::string> addresses;
    // Preparation validates descriptors and builds messages with no I/O.
    std::function<std::optional<RankTransferRequest>(const TransferTask&)> prepare;
};

class MultiRankBlockTransferEngine {
public:
    MultiRankBlockTransferEngine(std::vector<GroupSetPtr>                       group_sets,
                                 std::shared_ptr<BroadcastManager>              broadcast_manager,
                                 std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter = nullptr);
    MultiRankBlockTransferEngine(std::vector<GroupSetPtr>       group_sets,
                                 RankTransferTransport          transport,
                                 std::shared_ptr<CacheObserver> metrics_reporter = nullptr);

    // One copy direction per request; the receiving PerRank engines own batching.
    std::shared_ptr<TransferAsyncContext> execute(TransferTask task, const char* completion_operation = nullptr) const;

    const std::vector<std::string>& participantAddresses() const;
    const std::vector<GroupSetPtr>& groupSets() const noexcept {
        return group_sets_;
    }

private:
    std::vector<GroupSetPtr>       group_sets_;
    RankTransferTransport          transport_;
    std::shared_ptr<CacheObserver> metrics_reporter_;
};

using MultiRankBlockTransferEnginePtr = std::shared_ptr<MultiRankBlockTransferEngine>;

}  // namespace rtp_llm
