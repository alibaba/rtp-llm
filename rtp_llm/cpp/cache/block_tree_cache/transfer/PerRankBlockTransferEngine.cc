#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceDiskTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/HostDiskTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

ErrorInfo transferStatusToErrorInfo(TransferStatus status) {
    switch (status) {
        case TransferStatus::OK:
            return ErrorInfo::OkStatus();
        case TransferStatus::INVALID_ARGS:
            return ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid block transfer request");
        case TransferStatus::DEVICE_IO_ERROR:
            return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "device block transfer failed");
        case TransferStatus::DISK_IO_ERROR:
            return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "disk block transfer failed");
        case TransferStatus::RESOURCE_EXHAUSTED:
            return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "device-disk staging pool exhausted");
    }
    return ErrorInfo(ErrorCode::UNKNOWN_ERROR, "unknown block transfer status");
}

}  // namespace

PerRankBlockTransferEngine::PerRankBlockTransferEngine(std::vector<GroupSetPtr> group_sets,
                                                       DeviceHostCopyOptions    device_host_options,
                                                       size_t                   device_disk_staging_block_count):
    group_sets_(std::move(group_sets)),
    device_host_executor_(std::make_unique<DeviceHostTransferExecutor>(std::move(device_host_options))),
    host_disk_executor_(std::make_unique<HostDiskTransferExecutor>()) {
    const bool any_disk_pool = std::any_of(group_sets_.begin(), group_sets_.end(), [](const GroupSetPtr& group_set) {
        return group_set->diskPool() != nullptr;
    });
    if (any_disk_pool) {
        device_disk_executor_ = std::make_unique<DeviceDiskTransferExecutor>(
            *device_host_executor_, *host_disk_executor_, group_sets_, device_disk_staging_block_count);
    }
}

std::shared_ptr<AsyncContext> PerRankBlockTransferEngine::submit(const TransferDescriptor& desc) {
    const TransferStatus status = execute(desc);
    return std::make_shared<CompletedAsyncContext>(transferStatusToErrorInfo(status));
}

TransferStatus PerRankBlockTransferEngine::execute(const TransferDescriptor& desc) {
    const GroupSet& group_set = *group_sets_[desc.group_set_id];
    if ((desc.source_tier == Tier::DEVICE && desc.target_tier == Tier::HOST)
        || (desc.source_tier == Tier::HOST && desc.target_tier == Tier::DEVICE)) {
        return device_host_executor_->execute(
            resolveHostView(group_set, desc.singleBlockAt(Tier::HOST)), desc, group_set);
    }
    if ((desc.source_tier == Tier::HOST && desc.target_tier == Tier::DISK)
        || (desc.source_tier == Tier::DISK && desc.target_tier == Tier::HOST)) {
        return host_disk_executor_->execute(
            resolveHostView(group_set, desc.singleBlockAt(Tier::HOST)), desc, group_set);
    }
    if ((desc.source_tier == Tier::DEVICE && desc.target_tier == Tier::DISK)
        || (desc.source_tier == Tier::DISK && desc.target_tier == Tier::DEVICE)) {
        return device_disk_executor_->execute(desc, group_set);
    }
    RTP_LLM_LOG_WARNING(
        "unsupported transfer tier pair source=%s target=%s", tierName(desc.source_tier), tierName(desc.target_tier));
    return TransferStatus::INVALID_ARGS;
}

HostBufferView PerRankBlockTransferEngine::resolveHostView(const GroupSet& group_set, BlockIdxType host_block) {
    const HostBlockBuffer buffer = group_set.hostPool()->blockBuffer(host_block);
    return HostBufferView{buffer.addr, buffer.payload_bytes, buffer.stride_bytes};
}

}  // namespace rtp_llm
