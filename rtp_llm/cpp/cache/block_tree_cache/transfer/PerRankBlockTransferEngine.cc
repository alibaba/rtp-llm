#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"

#include <condition_variable>
#include <mutex>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/HostDiskTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/HostBlockPool.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

PerRankBlockTransferEngine::PerRankBlockTransferEngine(std::vector<GroupSetPtr> group_sets,
                                                       DeviceHostCopyOptions    device_host_options):
    group_sets_(std::move(group_sets)),
    device_host_executor_(std::make_unique<DeviceHostTransferExecutor>(std::move(device_host_options))),
    host_disk_executor_(std::make_unique<HostDiskTransferExecutor>()) {}

PerRankBlockTransferEngine::~PerRankBlockTransferEngine() = default;

// ---- TransferHandle ----

struct TransferHandle::State {
    explicit State(uint64_t id): request_id(id) {}

    uint64_t       request_id{0};
    bool           done{false};
    TransferStatus status{TransferStatus::OK};

    std::vector<TransferCompletionCallback> callbacks;

    mutable std::mutex      mutex;
    std::condition_variable cv;
};

TransferHandle TransferHandle::completed(TransferStatus status, uint64_t request_id) {
    auto state    = std::make_shared<TransferHandle::State>(request_id);
    state->status = status;
    state->done   = true;
    return TransferHandle(std::move(state));
}

uint64_t TransferHandle::requestId() const {
    return state_ ? state_->request_id : 0;
}

void TransferHandle::wait() const {
    auto state = state_;
    if (!state) {
        return;
    }

    std::unique_lock<std::mutex> lock(state->mutex);
    state->cv.wait(lock, [&state] { return state->done; });
}

bool TransferHandle::done() const {
    auto state = state_;
    if (!state) {
        return false;
    }

    std::lock_guard<std::mutex> lock(state->mutex);
    return state->done;
}

TransferStatus TransferHandle::status() const {
    auto state = state_;
    if (!state) {
        RTP_LLM_LOG_WARNING("invalid transfer handle");
        return TransferStatus::INVALID_ARGS;
    }

    wait();
    std::lock_guard<std::mutex> lock(state->mutex);
    return state->status;
}

void TransferHandle::onComplete(TransferCompletionCallback callback) const {
    auto state = state_;
    if (!state || !callback) {
        return;
    }

    TransferStatus completed_status = TransferStatus::OK;
    bool           run_now          = false;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->done) {
            completed_status = state->status;
            run_now          = true;
        } else {
            state->callbacks.push_back(std::move(callback));
        }
    }

    if (run_now) {
        callback(completed_status);
    }
}

TransferHandle PerRankBlockTransferEngine::submit(const TransferDescriptor& desc) {
    const uint64_t request_id = next_request_id_.fetch_add(1);
    return TransferHandle::completed(execute(desc), request_id);
}

TransferStatus PerRankBlockTransferEngine::execute(const TransferDescriptor& desc) {
    const GroupSet*      group  = nullptr;
    const TransferStatus status = validateRequest(desc, group);
    if (status != TransferStatus::OK) {
        return status;
    }

    if (desc.source_tier == Tier::DEVICE && desc.target_tier == Tier::HOST) {
        return device_host_executor_->execute(desc, *group);
    }
    if (desc.source_tier == Tier::HOST && desc.target_tier == Tier::DEVICE) {
        return device_host_executor_->execute(desc, *group);
    }
    if (desc.source_tier == Tier::HOST && desc.target_tier == Tier::DISK) {
        return host_disk_executor_->hostToDisk(desc, *group);
    }
    return host_disk_executor_->diskToHost(desc, *group);
}

TransferStatus PerRankBlockTransferEngine::validateRequest(const TransferDescriptor& desc,
                                                           const GroupSet*&          group) const {
    if (desc.group_set_id >= group_sets_.size()) {
        RTP_LLM_LOG_WARNING("invalid group_set_id=%zu", desc.group_set_id);
        return TransferStatus::INVALID_ARGS;
    }
    const GroupSetPtr& group_ptr = group_sets_[desc.group_set_id];
    if (group_ptr == nullptr) {
        RTP_LLM_LOG_WARNING("null group set=%zu", desc.group_set_id);
        return TransferStatus::INVALID_ARGS;
    }
    group = group_ptr.get();

    const bool device_host = (desc.source_tier == Tier::DEVICE && desc.target_tier == Tier::HOST)
                             || (desc.source_tier == Tier::HOST && desc.target_tier == Tier::DEVICE);
    if (device_host) {
        const auto host_pool = group->hostPool();
        if (host_pool == nullptr || !host_pool->validBlock(desc.host_block)) {
            RTP_LLM_LOG_WARNING("device-host request has invalid host block group=%zu", desc.group_set_id);
            return TransferStatus::INVALID_ARGS;
        }
        if (desc.device_blocks.size() != group->groupIds().size()) {
            RTP_LLM_LOG_WARNING("device-host request device block count %zu != group count %zu group_set=%zu",
                                desc.device_blocks.size(),
                                group->groupIds().size(),
                                desc.group_set_id);
            return TransferStatus::INVALID_ARGS;
        }
        bool has_device_block = false;
        for (size_t local_group_index = 0; local_group_index < desc.device_blocks.size(); ++local_group_index) {
            const BlockIdxType block = desc.device_blocks[local_group_index];
            if (isNullBlockIdx(block)) {
                continue;
            }
            const DeviceBlockPoolPtr& pool = group->devicePools()[local_group_index];
            if (pool == nullptr || !pool->validBlock(block)) {
                RTP_LLM_LOG_WARNING("invalid device block %d for local_group=%zu", block, local_group_index);
                return TransferStatus::INVALID_ARGS;
            }
            has_device_block = true;
        }
        return has_device_block ? TransferStatus::OK : TransferStatus::INVALID_ARGS;
    }

    const bool host_disk = (desc.source_tier == Tier::HOST && desc.target_tier == Tier::DISK)
                           || (desc.source_tier == Tier::DISK && desc.target_tier == Tier::HOST);
    if (host_disk) {
        const auto host_pool = group->hostPool();
        const auto disk_pool = group->diskPool();
        if (host_pool == nullptr || disk_pool == nullptr || !host_pool->validBlock(desc.host_block)
            || !disk_pool->validBlock(desc.disk_block)) {
            RTP_LLM_LOG_WARNING("invalid host-disk request group=%zu", desc.group_set_id);
            return TransferStatus::INVALID_ARGS;
        }
        return TransferStatus::OK;
    }

    RTP_LLM_LOG_WARNING(
        "unsupported transfer tier pair source=%s target=%s", tierName(desc.source_tier), tierName(desc.target_tier));
    return TransferStatus::INVALID_ARGS;
}

}  // namespace rtp_llm
