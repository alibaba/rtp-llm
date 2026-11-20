#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadTask.h"

#include <chrono>
#include <utility>

namespace rtp_llm {

std::shared_ptr<LoadTask>
LoadTask::create(std::shared_ptr<LoadAsyncContext> context, int host_timeout_ms, int disk_timeout_ms) {
    std::vector<TransferDescriptor>             load_descs;
    std::vector<std::shared_ptr<LoadOperation>> operations;
    std::vector<TransferDescriptor>             host_to_device_descriptors;
    std::vector<TransferDescriptor>             disk_to_device_descriptors;
    for (size_t desc_index : context->newLoadIndices()) {
        const auto& desc = context->loadDescs()[desc_index];
        load_descs.emplace_back(desc);
        operations.emplace_back(context->loadOperations()[desc_index]);
        if (desc.source_tier == Tier::HOST) {
            host_to_device_descriptors.emplace_back(desc);
        } else {
            disk_to_device_descriptors.emplace_back(desc);
        }
    }
    const auto request_id = context->requestId();
    return std::make_shared<LoadTask>(
        std::move(load_descs),
        TransferTask(std::move(host_to_device_descriptors), std::chrono::milliseconds(host_timeout_ms), request_id),
        TransferTask(std::move(disk_to_device_descriptors), std::chrono::milliseconds(disk_timeout_ms), request_id),
        std::move(operations),
        std::move(context));
}

LoadTask::LoadTask(std::vector<TransferDescriptor>             load_descriptors,
                   TransferTask                                host_task,
                   TransferTask                                disk_task,
                   std::vector<std::shared_ptr<LoadOperation>> load_operations,
                   std::shared_ptr<LoadAsyncContext>           load_context):
    load_descs(std::move(load_descriptors)),
    host_to_device_task(std::move(host_task)),
    disk_to_device_task(std::move(disk_task)),
    operations(std::move(load_operations)),
    context(std::move(load_context)) {
    host_to_device_task.setWorkflow(context->workflowId(), context->workflowId());
    disk_to_device_task.setWorkflow(context->workflowId(), context->workflowId());
}

}  // namespace rtp_llm
