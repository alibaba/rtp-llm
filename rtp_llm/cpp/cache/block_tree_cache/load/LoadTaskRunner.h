#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
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
        enum class Phase {
            CREATED,
            HOST_TO_DEVICE,
            DISK_TO_DEVICE,
            FINISHED
        };

        Task(std::vector<TransferDescriptor>   load_descriptors,
             TransferTask                      host_to_device_transfer_task,
             TransferTask                      disk_to_device_transfer_task,
             std::shared_ptr<LoadAsyncContext> load_context = nullptr):
            load_descs(std::move(load_descriptors)),
            host_to_device_task(std::move(host_to_device_transfer_task)),
            disk_to_device_task(std::move(disk_to_device_transfer_task)),
            target_installed(load_descs.size(), false),
            context(std::move(load_context)) {}

        std::vector<TransferDescriptor> load_descs;
        TransferTask                    host_to_device_task;
        TransferTask                    disk_to_device_task;
        std::vector<bool>               target_installed;
        // A submitted task owns the committed context until settlement completes.
        std::shared_ptr<LoadAsyncContext> context;
        Phase                             phase{Phase::CREATED};
    };
    using TaskPtr = std::shared_ptr<Task>;

    LoadTaskRunner(const std::vector<GroupSetPtr>& group_sets, int host_timeout_ms, int disk_timeout_ms);

    TaskPtr createTask(const std::shared_ptr<LoadAsyncContext>& context);
    void    runTransfer(TaskPtr                        task,
                        const BlockTransferDispatcher& transfer_dispatcher,
                        BlockTreeCacheMetricsReporter& metrics_reporter,
                        TransferDoneCallback           callback);
    void    releaseTaskResources(const Task& task);

private:
    void startDiskTransfer(TaskPtr                        task,
                           const BlockTransferDispatcher& transfer_dispatcher,
                           BlockTreeCacheMetricsReporter& metrics_reporter,
                           TransferDoneCallback           callback);
    void reportStageFinished(const Task&                            task,
                             BlockTreeCacheMetricsReporter&         metrics_reporter,
                             Tier                                   source_tier,
                             const std::vector<TransferDescriptor>& descriptors,
                             int64_t                                begin_time_us,
                             bool                                   success) const;

    const std::vector<GroupSetPtr>& group_sets_;
    int                             host_timeout_ms_{0};
    int                             disk_timeout_ms_{0};
};

}  // namespace rtp_llm
