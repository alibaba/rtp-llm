#pragma once

#include <cstddef>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class BlockTransferDispatcher;
class BlockTreeCacheMetricsReporter;

class StoreTaskRunner {
public:
    struct Task {
        Tier                            target_tier{Tier::NONE};
        CacheKeysType                   cache_keys;
        std::vector<TransferDescriptor> descriptors;
    };

    explicit StoreTaskRunner(const std::vector<GroupSetPtr>& group_sets);

    bool prepareTask(Task& task, const std::vector<std::vector<GroupSetResource>>& resources);
    bool runTransfer(Task&                          task,
                     const BlockTransferDispatcher& transfer_dispatcher,
                     BlockTreeCacheMetricsReporter& metrics_reporter,
                     int                            host_timeout_ms,
                     int                            disk_timeout_ms);
    void releaseTaskResources(const Task& task);

private:
    const std::vector<GroupSetPtr>& group_sets_;
};

}  // namespace rtp_llm
