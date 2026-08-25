#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
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
        enum class Phase { CREATED, TRANSFERRING, FINISHED };

        Tier                            target_tier{Tier::NONE};
        CacheKeysType                   cache_keys;
        std::vector<TransferDescriptor> descriptors;
        Phase                           phase{Phase::CREATED};
        int64_t                         transfer_begin_time_us{0};
    };
    using TaskPtr = std::shared_ptr<Task>;

    explicit StoreTaskRunner(const std::vector<GroupSetPtr>& group_sets);

    bool prepareTask(Task& task, const std::vector<std::vector<GroupSetResource>>& resources);
    void runTransfer(TaskPtr                         task,
                     const BlockTransferDispatcher& transfer_dispatcher,
                     BlockTreeCacheMetricsReporter& metrics_reporter,
                     int                            host_timeout_ms,
                     int                            disk_timeout_ms,
                     TransferDoneCallback           callback);
    void releaseTaskResources(const Task& task);

private:
    const std::vector<GroupSetPtr>& group_sets_;
};

}  // namespace rtp_llm
