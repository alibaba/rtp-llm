#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageTimer.h"

namespace rtp_llm {

struct LoadTask {
    static std::shared_ptr<LoadTask>
    create(std::shared_ptr<LoadAsyncContext> context, int host_timeout_ms, int disk_timeout_ms);

    LoadTask(std::vector<TransferDescriptor>             load_descriptors,
             TransferTask                                host_task,
             TransferTask                                disk_task,
             std::vector<std::shared_ptr<LoadOperation>> load_operations,
             std::shared_ptr<LoadAsyncContext>           load_context);

    const TransferTask& completionTransfer() const {
        return !disk_result.block_errors.empty() || host_to_device_task.descriptors().empty() ? disk_to_device_task :
                                                                                                host_to_device_task;
    }

    std::vector<TransferDescriptor>             load_descs;
    TransferTask                                host_to_device_task;
    TransferTask                                disk_to_device_task;
    TransferResult                              host_result;
    TransferResult                              disk_result;
    std::vector<std::shared_ptr<LoadOperation>> operations;
    const std::shared_ptr<LoadAsyncContext>     context;
    std::mutex                                  wait_mutex;
    bool                                        host_completed{false};
    bool                                        disk_completed{false};
    SharedTransferStageTimer                    direction_wait;
    SharedTransferStageTimer                    wait;
};

}  // namespace rtp_llm
