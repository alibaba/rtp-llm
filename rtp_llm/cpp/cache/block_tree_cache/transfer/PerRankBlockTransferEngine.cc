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

enum class TransferPath {
    DEVICE_HOST,
    HOST_DISK,
    DEVICE_DISK,
    UNSUPPORTED,
};

TransferPath classifyTransferPath(Tier source, Tier target) {
    if ((source == Tier::DEVICE && target == Tier::HOST) || (source == Tier::HOST && target == Tier::DEVICE)) {
        return TransferPath::DEVICE_HOST;
    }
    if ((source == Tier::HOST && target == Tier::DISK) || (source == Tier::DISK && target == Tier::HOST)) {
        return TransferPath::HOST_DISK;
    }
    if ((source == Tier::DEVICE && target == Tier::DISK) || (source == Tier::DISK && target == Tier::DEVICE)) {
        return TransferPath::DEVICE_DISK;
    }
    return TransferPath::UNSUPPORTED;
}

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
        return group_set != nullptr && group_set->diskPool() != nullptr;
    });
    if (any_disk_pool) {
        device_disk_executor_ = std::make_unique<DeviceDiskTransferExecutor>(
            *device_host_executor_, *host_disk_executor_, group_sets_, device_disk_staging_block_count);
    }
}

PerRankBlockTransferEngine::~PerRankBlockTransferEngine() = default;

std::shared_ptr<AsyncContext> PerRankBlockTransferEngine::submit(const TransferDescriptor& desc) {
    const TransferStatus status = execute(desc);
    return std::make_shared<CompletedAsyncContext>(transferStatusToErrorInfo(status));
}

TransferStatus PerRankBlockTransferEngine::execute(const TransferDescriptor& desc) {
    const GroupSet*      group_set = nullptr;
    const TransferStatus status    = validateRequest(desc, group_set);
    if (status != TransferStatus::OK) {
        return status;
    }

    switch (classifyTransferPath(desc.source_tier, desc.target_tier)) {
        case TransferPath::DEVICE_HOST:
            if (desc.source_tier == Tier::DEVICE) {
                return device_host_executor_->deviceToHost(
                    desc, *group_set, resolveHostView(*group_set, desc.host_block));
            }
            return device_host_executor_->hostToDevice(
                resolveHostView(*group_set, desc.host_block), desc, *group_set);
        case TransferPath::HOST_DISK:
            if (desc.source_tier == Tier::HOST) {
                return host_disk_executor_->hostToDisk(
                    resolveHostView(*group_set, desc.host_block), desc, *group_set);
            }
            return host_disk_executor_->diskToHost(
                desc, *group_set, resolveHostView(*group_set, desc.host_block));
        case TransferPath::DEVICE_DISK:
            return device_disk_executor_->execute(desc, *group_set);
        case TransferPath::UNSUPPORTED:
            break;
    }
    RTP_LLM_LOG_WARNING(
        "unsupported transfer tier pair source=%s target=%s", tierName(desc.source_tier), tierName(desc.target_tier));
    return TransferStatus::INVALID_ARGS;
}

HostBufferView PerRankBlockTransferEngine::resolveHostView(const GroupSet& group_set, BlockIdxType host_block) {
    const HostBlockBuffer buffer = group_set.hostPool()->blockBuffer(host_block);
    return HostBufferView{buffer.addr, buffer.payload_bytes, buffer.stride_bytes};
}

TransferStatus PerRankBlockTransferEngine::validateRequest(const TransferDescriptor& desc,
                                                           const GroupSet*&          group_set) const {
    if (desc.group_set_id >= group_sets_.size()) {
        RTP_LLM_LOG_WARNING("invalid group_set_id=%zu", desc.group_set_id);
        return TransferStatus::INVALID_ARGS;
    }
    const GroupSetPtr& group_ptr = group_sets_[desc.group_set_id];
    if (group_ptr == nullptr) {
        RTP_LLM_LOG_WARNING("null group set=%zu", desc.group_set_id);
        return TransferStatus::INVALID_ARGS;
    }
    group_set = group_ptr.get();

    const TransferPath path = classifyTransferPath(desc.source_tier, desc.target_tier);
    if (path == TransferPath::UNSUPPORTED) {
        RTP_LLM_LOG_WARNING("unsupported transfer tier pair source=%s target=%s",
                            tierName(desc.source_tier),
                            tierName(desc.target_tier));
        return TransferStatus::INVALID_ARGS;
    }

    if (path == TransferPath::DEVICE_HOST || path == TransferPath::HOST_DISK) {
        const auto host_pool = group_set->hostPool();
        if (host_pool == nullptr || !host_pool->validBlock(desc.host_block)) {
            RTP_LLM_LOG_WARNING("transfer request has invalid host block group=%zu", desc.group_set_id);
            return TransferStatus::INVALID_ARGS;
        }
    }

    if (path == TransferPath::HOST_DISK || path == TransferPath::DEVICE_DISK) {
        const auto disk_pool = group_set->diskPool();
        if (disk_pool == nullptr || !disk_pool->validBlock(desc.disk_block)) {
            RTP_LLM_LOG_WARNING("transfer request has invalid disk block group=%zu", desc.group_set_id);
            return TransferStatus::INVALID_ARGS;
        }
    }

    if (path == TransferPath::DEVICE_DISK) {
        if (device_disk_executor_ == nullptr) {
            RTP_LLM_LOG_WARNING("device-disk request but direct executor disabled, group=%zu", desc.group_set_id);
            return TransferStatus::INVALID_ARGS;
        }
    }

    return path == TransferPath::DEVICE_HOST || path == TransferPath::DEVICE_DISK
               ? validateDeviceBlocks(desc, *group_set)
               : TransferStatus::OK;
}

TransferStatus PerRankBlockTransferEngine::validateDeviceBlocks(const TransferDescriptor& desc,
                                                                const GroupSet&           group_set) const {
    if (desc.device_blocks.size() != group_set.devicePools().size()) {
        RTP_LLM_LOG_WARNING("device block count %zu != pool count %zu group_set=%zu",
                            desc.device_blocks.size(),
                            group_set.devicePools().size(),
                            desc.group_set_id);
        return TransferStatus::INVALID_ARGS;
    }
    bool has_device_block = false;
    for (size_t local_group_index = 0; local_group_index < desc.device_blocks.size(); ++local_group_index) {
        const BlockIdxType block = desc.device_blocks[local_group_index];
        if (isNullBlockIdx(block)) {
            continue;
        }
        const DeviceBlockPoolPtr& pool = group_set.devicePools()[local_group_index];
        if (pool == nullptr || !pool->validBlock(block)) {
            RTP_LLM_LOG_WARNING("invalid device block %d for local_group=%zu", block, local_group_index);
            return TransferStatus::INVALID_ARGS;
        }
        has_device_block = true;
    }
    return has_device_block ? TransferStatus::OK : TransferStatus::INVALID_ARGS;
}

}  // namespace rtp_llm
