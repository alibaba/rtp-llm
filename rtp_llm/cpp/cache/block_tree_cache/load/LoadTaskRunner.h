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
        std::vector<TransferDescriptor>    host_to_device_descriptors;
        std::vector<TransferDescriptor>    disk_to_device_descriptors;
        std::vector<bool>                  target_installed;
        // A submitted task owns the committed context until settlement completes.
        std::shared_ptr<LoadAsyncContext> context;
    };
    using TaskPtr = std::shared_ptr<Task>;

    explicit LoadTaskRunner(const std::vector<GroupSetPtr>& group_sets);

    TaskPtr createTask(const std::shared_ptr<LoadAsyncContext>& context);
    bool    runTransfer(Task&                          task,
                        const BlockTransferDispatcher& transfer_dispatcher,
                        BlockTreeCacheMetricsReporter& metrics_reporter,
                        int                            disk_timeout_ms,
                        int                            host_timeout_ms);
    void    releaseTaskResources(const Task& task);

private:
    const std::vector<GroupSetPtr>& group_sets_;
};

}  // namespace rtp_llm
