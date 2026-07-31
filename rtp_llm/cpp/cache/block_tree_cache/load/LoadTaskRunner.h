#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class BlockTreeCacheMetricsReporter;
class BlockTransferDispatcher;

class LoadTaskRunner {
public:
    struct Task {
        std::vector<TransferDescriptor> load_descs;
        std::vector<GroupSetPtr> desc_group_sets;
        std::vector<TransferDescriptor>    host_to_device_descriptors;
        std::vector<TransferDescriptor>    disk_to_device_descriptors;
        std::vector<bool>                  target_installed;
        // A submitted task owns the committed context until settlement completes.
        std::shared_ptr<LoadAsyncContext> context;
    };
    using TaskPtr = std::shared_ptr<Task>;

    TaskPtr createTask(const std::vector<TransferDescriptor>& load_descs,
                       const std::vector<bool>&                  joined_load,
                       const std::vector<GroupSetPtr>&           group_sets,
                       const std::shared_ptr<LoadAsyncContext>&  context);
    bool    prepareTransferDescriptor(Task& task, size_t desc_index);
    bool    runTransfer(Task&                          task,
                        const BlockTransferDispatcher& transfer_dispatcher,
                        BlockTreeCacheMetricsReporter& metrics_reporter,
                        int                            disk_timeout_ms,
                        int                            host_timeout_ms,
                        bool                           prepared);
    void    releaseTaskResources(const Task& task);

private:
    void releaseUninstalledTargetHolders(const Task& task);
};

}  // namespace rtp_llm
