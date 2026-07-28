#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"

#include <memory>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"
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
    }
    return ErrorInfo(ErrorCode::UNKNOWN_ERROR, "unknown block transfer status");
}

}  // namespace

PerRankBlockTransferEngine::PerRankBlockTransferEngine(std::vector<GroupSetPtr> group_sets,
                                                       DeviceHostCopyOptions    device_host_options):
    group_sets_(std::move(group_sets)),
    device_host_executor_(std::make_unique<DeviceHostTransferExecutor>(std::move(device_host_options))),
    host_disk_executor_(std::make_unique<HostDiskTransferExecutor>()) {}

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

    if (desc.source_tier == Tier::DEVICE && desc.target_tier == Tier::HOST) {
        return device_host_executor_->execute(desc, *group_set);
    }
    if (desc.source_tier == Tier::HOST && desc.target_tier == Tier::DEVICE) {
        return device_host_executor_->execute(desc, *group_set);
    }
    if (desc.source_tier == Tier::HOST && desc.target_tier == Tier::DISK) {
        return host_disk_executor_->hostToDisk(desc, *group_set);
    }
    return host_disk_executor_->diskToHost(desc, *group_set);
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

    const bool device_host = (desc.source_tier == Tier::DEVICE && desc.target_tier == Tier::HOST)
                             || (desc.source_tier == Tier::HOST && desc.target_tier == Tier::DEVICE);
    if (device_host) {
        const auto host_pool = group_set->hostPool();
        if (host_pool == nullptr || !host_pool->validBlock(desc.host_block)) {
            RTP_LLM_LOG_WARNING("device-host request has invalid host block group_set=%zu", desc.group_set_id);
            return TransferStatus::INVALID_ARGS;
        }
        if (desc.device_blocks.size() != group_set->groupIds().size()) {
            RTP_LLM_LOG_WARNING("device-host request device block count %zu != group_set count %zu group_set=%zu",
                                desc.device_blocks.size(),
                                group_set->groupIds().size(),
                                desc.group_set_id);
            return TransferStatus::INVALID_ARGS;
        }
        bool has_device_block = false;
        for (size_t member_index = 0; member_index < desc.device_blocks.size(); ++member_index) {
            const BlockIdxType block = desc.device_blocks[member_index];
            if (isNullBlockIdx(block)) {
                continue;
            }
            const DeviceBlockPoolPtr& pool = group_set->devicePools()[member_index];
            if (pool == nullptr || !pool->validBlock(block)) {
                RTP_LLM_LOG_WARNING("invalid device block %d for group_id=%zu member_index=%zu",
                                    block,
                                    group_set->groupIds()[member_index],
                                    member_index);
                return TransferStatus::INVALID_ARGS;
            }
            has_device_block = true;
        }
        return has_device_block ? TransferStatus::OK : TransferStatus::INVALID_ARGS;
    }

    const bool host_disk = (desc.source_tier == Tier::HOST && desc.target_tier == Tier::DISK)
                           || (desc.source_tier == Tier::DISK && desc.target_tier == Tier::HOST);
    if (host_disk) {
        const auto host_pool = group_set->hostPool();
        const auto disk_pool = group_set->diskPool();
        if (host_pool == nullptr || disk_pool == nullptr || !host_pool->validBlock(desc.host_block)
            || !disk_pool->validBlock(desc.disk_block)) {
            RTP_LLM_LOG_WARNING("invalid host-disk request group_set=%zu", desc.group_set_id);
            return TransferStatus::INVALID_ARGS;
        }
        return TransferStatus::OK;
    }

    RTP_LLM_LOG_WARNING(
        "unsupported transfer tier pair source=%s target=%s", tierName(desc.source_tier), tierName(desc.target_tier));
    return TransferStatus::INVALID_ARGS;
}

}  // namespace rtp_llm
