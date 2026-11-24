#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

class BlockTransferDispatcher;
class BlockTreeCacheMetricsReporter;
class BlockTree;

class StoreTaskRunner {
public:
    using TransferDoneCallback = std::function<void(TransferResult)>;

    struct Task {
        enum class Phase {
            CREATED,
            TRANSFERRING,
            FINISHED
        };

        Task(Tier target, CacheKeysType keys, std::chrono::milliseconds timeout):
            target_tier(target), cache_keys(std::move(keys)), transfer_task({}, timeout) {}

        DescriptorView descriptors() const {
            return transfer_task.descriptors();
        }

        void recordMockSubmission(BlockTreeCacheMetricsReporter& metrics_reporter);
        void recordMockTransferOutcome(BlockTreeCacheMetricsReporter& metrics_reporter, bool success);

        Tier          target_tier{Tier::NONE};
        CacheKeysType cache_keys;
        TransferTask  transfer_task;
        Phase         phase{Phase::CREATED};
        bool          mock_memory_cache{false};
        bool          mock_disk_cache{false};

    private:
        // Either submit's return or a terminal callback can establish accepted
        // admission first. No guard is held across submission or callbacks.
        std::mutex mock_outcome_mutex;
        bool       mock_submission_recorded{false};
        bool       mock_outcome_recorded{false};
    };
    using TaskPtr = std::shared_ptr<Task>;

    explicit StoreTaskRunner(const std::vector<GroupSetPtr>& group_sets, BlockTree* pressure_tree = nullptr);

    bool prepareTask(Task& task, const std::vector<std::vector<GroupSetResource>>& resources);
    void runTransfer(TaskPtr                        task,
                     const BlockTransferDispatcher& transfer_dispatcher,
                     BlockTreeCacheMetricsReporter& metrics_reporter,
                     TransferDoneCallback           callback);
    void releaseTaskResources(const Task& task);

private:
    void freezeUnconfirmedEndpoints(const Task& task, const TransferResult& result) const;

    BlockTree*                      pressure_tree_{nullptr};
    const std::vector<GroupSetPtr>& group_sets_;
};

}  // namespace rtp_llm
