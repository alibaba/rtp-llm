#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

class BlockTransferDispatcher;
class BlockTreeCacheMetricsReporter;

class StoreTaskRunner {
public:
    using TransferDoneCallback = std::function<void(ErrorInfo)>;

    struct Task {
        enum class Phase {
            CREATED,
            TRANSFERRING,
            FINISHED
        };

        Task(Tier target, CacheKeysType keys, std::chrono::milliseconds timeout):
            target_tier(target), cache_keys(std::move(keys)), transfer_task({}, timeout) {}

        const std::vector<TransferDescriptor>& descriptors() const {
            return transfer_task.descriptors();
        }

        Tier          target_tier{Tier::NONE};
        CacheKeysType cache_keys;
        TransferTask  transfer_task;
        Phase         phase{Phase::CREATED};
    };
    using TaskPtr = std::shared_ptr<Task>;

    explicit StoreTaskRunner(const std::vector<GroupSetPtr>& group_sets);

    bool prepareTask(Task& task, const std::vector<std::vector<GroupSetResource>>& resources);
    void runTransfer(TaskPtr                        task,
                     const BlockTransferDispatcher& transfer_dispatcher,
                     BlockTreeCacheMetricsReporter& metrics_reporter,
                     TransferDoneCallback           callback);
    void releaseTaskResources(const Task& task);

private:
    const std::vector<GroupSetPtr>& group_sets_;
};

}  // namespace rtp_llm
