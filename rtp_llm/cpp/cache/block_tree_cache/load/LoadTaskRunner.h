#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageTimer.h"

namespace rtp_llm {

class BlockTreeCacheMetricsReporter;
class BlockTree;
class BlockTransferDispatcher;

class LoadTaskRunner {
public:
    using TransferDoneCallback = std::function<void(ErrorInfo)>;

    struct Task {
        enum class Phase {
            CREATED,
            TRANSFERRING,
            FINISHED
        };

        Task(std::vector<TransferDescriptor>   load_descriptors,
             TransferTask                      host_to_device_transfer_task,
             TransferTask                      disk_to_device_transfer_task,
             std::shared_ptr<LoadAsyncContext> load_context = nullptr,
             TransferTask                      remote_task  = TransferTask({}, std::chrono::milliseconds(0))):
            load_descs(std::move(load_descriptors)),
            host_to_device_task(std::move(host_to_device_transfer_task)),
            disk_to_device_task(std::move(disk_to_device_transfer_task)),
            remote_to_device_task(std::move(remote_task)),
            target_installed(load_descs.size(), false),
            context(std::move(load_context)) {}

        // Called after all directions settle; prefer DISK/REMOTE when present.
        const TransferTask& completionTransfer() const {
            if (!disk_result.block_errors.empty() || !disk_to_device_task.descriptors().empty()) {
                return disk_to_device_task;
            }
            if (!remote_result.block_errors.empty() || !remote_to_device_task.descriptors().empty()) {
                return remote_to_device_task;
            }
            return host_to_device_task;
        }

        std::vector<TransferDescriptor> load_descs;
        TransferTask                    host_to_device_task;
        TransferTask                    disk_to_device_task;
        TransferTask                    remote_to_device_task;
        // Published before completion; settlement invalidates only source records with a confirmed CRC mismatch.
        TransferResult    host_result;
        TransferResult    disk_result;
        TransferResult    remote_result;
        std::vector<bool> target_installed;
        // A submitted task owns the committed context until settlement completes.
        std::shared_ptr<LoadAsyncContext> context;
        Phase                             phase{Phase::CREATED};
        std::mutex                        wait_mutex;
        bool                              host_completed{false};
        bool                              disk_completed{false};
        SharedTransferStageTimer          direction_wait;
        SharedTransferStageTimer          wait;
    };
    using TaskPtr = std::shared_ptr<Task>;

    LoadTaskRunner(const std::vector<GroupSetPtr>& group_sets,
                   int                             host_timeout_ms,
                   int                             disk_timeout_ms,
                   BlockTree*                      pressure_tree     = nullptr,
                   int                             remote_timeout_ms = 10000);

    TaskPtr createTask(const std::shared_ptr<LoadAsyncContext>& context);
    void    runTransfer(TaskPtr                        task,
                        const BlockTransferDispatcher& transfer_dispatcher,
                        BlockTreeCacheMetricsReporter& metrics_reporter,
                        TransferDoneCallback           callback);
    void    releaseTaskResources(const Task& task);

private:
    void reportStageFinished(const Task&                    task,
                             BlockTreeCacheMetricsReporter& metrics_reporter,
                             Tier                           source_tier,
                             DescriptorView                 descriptors,
                             int64_t                        begin_time_us,
                             bool                           success) const;

    BlockTree*                      pressure_tree_{nullptr};
    const std::vector<GroupSetPtr>& group_sets_;
    int                             host_timeout_ms_{0};
    int                             disk_timeout_ms_{0};
    int                             remote_timeout_ms_{10000};
};

}  // namespace rtp_llm
