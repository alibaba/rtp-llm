#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
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
    using TransferDoneCallback = std::function<void(ErrorInfo)>;

    struct Task {
        enum class Phase { CREATED, HOST_TO_DEVICE, DISK_TO_DEVICE, FINISHED };

        std::vector<TransferDescriptor>    load_descs;
        std::vector<TransferDescriptor>    host_to_device_descriptors;
        std::vector<TransferDescriptor>    disk_to_device_descriptors;
        std::vector<bool>                  target_installed;
        // A submitted task owns the committed context until settlement completes.
        std::shared_ptr<LoadAsyncContext>  context;
        Phase                              phase{Phase::CREATED};
        int64_t                            host_transfer_begin_time_us{0};
        int64_t                            disk_transfer_begin_time_us{0};
    };
    using TaskPtr = std::shared_ptr<Task>;

    explicit LoadTaskRunner(const std::vector<GroupSetPtr>& group_sets);

    TaskPtr createTask(const std::shared_ptr<LoadAsyncContext>& context);
    void runTransfer(TaskPtr                         task,
                     const BlockTransferDispatcher& transfer_dispatcher,
                     BlockTreeCacheMetricsReporter& metrics_reporter,
                     int                            disk_timeout_ms,
                     int                            host_timeout_ms,
                     TransferDoneCallback           callback);
    void    releaseTaskResources(const Task& task);

private:
    void startDiskTransfer(TaskPtr                         task,
                           const BlockTransferDispatcher& transfer_dispatcher,
                           BlockTreeCacheMetricsReporter& metrics_reporter,
                           int                            disk_timeout_ms,
                           TransferDoneCallback           callback);
    void reportStageFinished(const Task&                              task,
                             BlockTreeCacheMetricsReporter&           metrics_reporter,
                             Tier                                     source_tier,
                             const std::vector<TransferDescriptor>& descriptors,
                             int64_t                                  begin_time_us,
                             bool                                     success) const;

    const std::vector<GroupSetPtr>& group_sets_;
};

}  // namespace rtp_llm
