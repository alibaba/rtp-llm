#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <functional>
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
    std::vector<TaskCompletion> queryTasks(const std::vector<std::string>& ids, int timeout_ms = 200) const;
    std::vector<TaskCompletion> queryKVCMOperations(const std::vector<std::string>& ids, int timeout_ms = 200) const;

    // Enqueue only after successful completion or release of the owning FrozenTask.
    void enqueueFinishedTasks(const std::vector<std::string>& ids) const;
    void enqueueFinishedKVCMOperations(const std::vector<std::string>& ids) const;
    void forgetFinishedTasks() const;
    void forgetFinishedKVCMOperations() const;

private:
    struct CleanupQueue {
        std::mutex mutex;
        std::unordered_set<std::string> ids;
    };
    // Callbacks can outlive execute(); they retain only the queue, not the engine.
    std::shared_ptr<CleanupQueue> cleanup_{std::make_shared<CleanupQueue>()};
    std::shared_ptr<CleanupQueue> kvcm_cleanup_{std::make_shared<CleanupQueue>()};
    mutable std::mutex cleanup_poll_mutex_;
    mutable std::mutex kvcm_cleanup_poll_mutex_;
    using ForgetTasks = std::function<bool(const std::vector<std::string>&)>;
    static void forgetFinishedWith(const std::shared_ptr<CleanupQueue>& cleanup,
                                   std::mutex& poll_mutex,
                                   const ForgetTasks& forget);
    void forgetFinishedTasksWith(const ForgetTasks& forget) const;
    void forgetFinishedKVCMOperationsWith(const ForgetTasks& forget) const;
    friend class BlockTransferDispatcher;

    std::vector<GroupSetPtr>                       group_sets_;
    std::shared_ptr<BroadcastManager>              broadcast_manager_;
    std::shared_ptr<BlockTreeCacheMetricsReporter> metrics_reporter_;
};

using MultiRankBlockTransferEnginePtr = std::shared_ptr<MultiRankBlockTransferEngine>;

}  // namespace rtp_llm
