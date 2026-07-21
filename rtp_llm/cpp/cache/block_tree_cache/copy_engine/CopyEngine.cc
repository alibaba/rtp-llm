#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/CopyEngine.h"

#include <condition_variable>
#include <mutex>
#include <utility>

#include "rtp_llm/cpp/cache/block_tree_cache/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/DeviceHostTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/HostDiskTransferExecutor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/HostBlockPool.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

CopyEngine::CopyEngine(std::vector<ComponentGroupPtr>                component_groups,
                       std::shared_ptr<const std::vector<Component>> component_registry,
                       DeviceHostCopyOptions                         device_host_options):
    component_groups_(std::move(component_groups)),
    component_registry_(std::move(component_registry)),
    device_host_executor_(std::make_unique<DeviceHostTransferExecutor>(std::move(device_host_options))),
    host_disk_executor_(std::make_unique<HostDiskTransferExecutor>()) {}

CopyEngine::~CopyEngine() = default;

// ---- TransferHandle ----

struct TransferHandle::State {
    explicit State(uint64_t id): request_id(id) {}

    uint64_t   request_id{0};
    bool       done{false};
    CopyStatus status{CopyStatus::OK};

    std::vector<CopyCompletionCallback> callbacks;

    mutable std::mutex      mutex;
    std::condition_variable cv;
};

TransferHandle TransferHandle::completed(CopyStatus status, uint64_t request_id) {
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

CopyStatus TransferHandle::status() const {
    auto state = state_;
    if (!state) {
        RTP_LLM_LOG_WARNING("invalid transfer handle");
        return CopyStatus::INVALID_ARGS;
    }

    wait();
    std::lock_guard<std::mutex> lock(state->mutex);
    return state->status;
}

void TransferHandle::onComplete(CopyCompletionCallback callback) const {
    auto state = state_;
    if (!state || !callback) {
        return;
    }

    CopyStatus completed_status = CopyStatus::OK;
    bool       run_now          = false;
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

// ---- CopyEngine: submit / execute ----

TransferHandle CopyEngine::submit(const TransferDescriptor& desc) {
    const uint64_t request_id = next_request_id_.fetch_add(1);
    return TransferHandle::completed(execute(desc), request_id);
}

CopyStatus CopyEngine::execute(const TransferDescriptor& desc) {
    const ComponentGroup* group  = nullptr;
    const CopyStatus      status = validateRequest(desc, group);
    if (status != CopyStatus::OK) {
        return status;
    }

    if (desc.source_tier == Tier::DEVICE && desc.target_tier == Tier::HOST) {
        return device_host_executor_->execute(desc, *group, *component_registry_);
    }
    if (desc.source_tier == Tier::HOST && desc.target_tier == Tier::DEVICE) {
        return device_host_executor_->execute(desc, *group, *component_registry_);
    }
    if (desc.source_tier == Tier::HOST && desc.target_tier == Tier::DISK) {
        return host_disk_executor_->hostToDisk(desc, *group);
    }
    return host_disk_executor_->diskToHost(desc, *group);
}

CopyStatus CopyEngine::validateRequest(const TransferDescriptor& desc, const ComponentGroup*& group) const {
    if (desc.component_group_id < 0 || static_cast<size_t>(desc.component_group_id) >= component_groups_.size()) {
        RTP_LLM_LOG_WARNING("invalid component_group_id=%d", desc.component_group_id);
        return CopyStatus::INVALID_ARGS;
    }
    const ComponentGroupPtr& group_ptr = component_groups_[static_cast<size_t>(desc.component_group_id)];
    if (group_ptr == nullptr || component_registry_ == nullptr) {
        RTP_LLM_LOG_WARNING("null component group=%d", desc.component_group_id);
        return CopyStatus::INVALID_ARGS;
    }
    group = group_ptr.get();

    const bool device_host = (desc.source_tier == Tier::DEVICE && desc.target_tier == Tier::HOST)
                             || (desc.source_tier == Tier::HOST && desc.target_tier == Tier::DEVICE);
    if (device_host) {
        const auto host_pool = group->hostPool();
        if (host_pool == nullptr || !host_pool->validBlock(desc.host_block)) {
            RTP_LLM_LOG_WARNING("device-host request has invalid host block group=%d", desc.component_group_id);
            return CopyStatus::INVALID_ARGS;
        }
        if (desc.device_blocks.size() != group->componentIndices().size()) {
            RTP_LLM_LOG_WARNING("device-host request device block count %zu != component count %zu group=%d",
                                desc.device_blocks.size(),
                                group->componentIndices().size(),
                                desc.component_group_id);
            return CopyStatus::INVALID_ARGS;
        }
        bool has_device_block = false;
        for (size_t component_idx = 0; component_idx < desc.device_blocks.size(); ++component_idx) {
            const BlockIdxType block = desc.device_blocks[component_idx];
            if (isNullBlockIdx(block)) {
                continue;
            }
            const DeviceBlockPoolPtr& pool = group->devicePools()[component_idx];
            if (pool == nullptr || !pool->validBlock(block)) {
                RTP_LLM_LOG_WARNING("invalid device block %d for component=%zu", block, component_idx);
                return CopyStatus::INVALID_ARGS;
            }
            has_device_block = true;
        }
        return has_device_block ? CopyStatus::OK : CopyStatus::INVALID_ARGS;
    }

    const bool host_disk = (desc.source_tier == Tier::HOST && desc.target_tier == Tier::DISK)
                           || (desc.source_tier == Tier::DISK && desc.target_tier == Tier::HOST);
    if (host_disk) {
        const auto host_pool = group->hostPool();
        const auto disk_pool = group->diskPool();
        if (host_pool == nullptr || disk_pool == nullptr || !host_pool->validBlock(desc.host_block)
            || !disk_pool->validBlock(desc.disk_block)) {
            RTP_LLM_LOG_WARNING("invalid host-disk request group=%d", desc.component_group_id);
            return CopyStatus::INVALID_ARGS;
        }
        return CopyStatus::OK;
    }

    RTP_LLM_LOG_WARNING(
        "unsupported transfer tier pair source=%s target=%s", tierName(desc.source_tier), tierName(desc.target_tier));
    return CopyStatus::INVALID_ARGS;
}

}  // namespace rtp_llm
