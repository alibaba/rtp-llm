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

CopyEngine::CopyEngine(const std::vector<ComponentGroupPtr>& component_groups,
                       const std::vector<Component>&         components,
                       DeviceHostCopyOptions                 device_host_options):
    device_host_executor_(std::make_unique<DeviceHostTransferExecutor>(std::move(device_host_options))),
    host_disk_executor_(std::make_unique<HostDiskTransferExecutor>()) {
    buildGroupLayouts(component_groups, components);
}

CopyEngine::~CopyEngine() = default;

bool CopyEngine::isDeviceHostTransfer(Tier source_tier, Tier target_tier) {
    return (source_tier == Tier::DEVICE && target_tier == Tier::HOST)
           || (source_tier == Tier::HOST && target_tier == Tier::DEVICE);
}

void CopyEngine::buildGroupLayouts(const std::vector<ComponentGroupPtr>& component_groups,
                                   const std::vector<Component>&         components) {
    group_layouts_.reserve(component_groups.size());
    for (size_t group_index = 0; group_index < component_groups.size(); ++group_index) {
        const auto&         group = component_groups[group_index];
        ResolvedGroupLayout layout;

        layout.component_group_id = group->component_group_id;
        // Nullable: device-only / host-disabled configs are legal. Per-path pool checks happen at submit.
        // Hold shared ownership so the cached schema keeps its pools alive independently of the group.
        layout.host_pool = group->hostPool();
        layout.disk_pool = group->diskPool();

        const auto& pools                 = group->devicePools();
        bool        device_host_layout_ok = true;
        layout.components.reserve(group->component_indices.size());
        for (int component_index : group->component_indices) {
            if (component_index < 0 || static_cast<size_t>(component_index) >= components.size()) {
                RTP_LLM_LOG_WARNING("invalid component_index=%d group=%d",
                                    component_index,
                                    layout.component_group_id);
                device_host_layout_ok = false;
                break;
            }
            const auto& component = components[static_cast<size_t>(component_index)];
            if (component.component_group_id != layout.component_group_id) {
                RTP_LLM_LOG_WARNING("component[%d] belongs to group %d, expected %d",
                                    component_index,
                                    component.component_group_id,
                                    layout.component_group_id);
                device_host_layout_ok = false;
                break;
            }
            if (component.device_pool_index < 0 || static_cast<size_t>(component.device_pool_index) >= pools.size()) {
                RTP_LLM_LOG_WARNING("invalid device_pool_index=%d component=%d group=%d",
                                    component.device_pool_index,
                                    component_index,
                                    layout.component_group_id);
                device_host_layout_ok = false;
                break;
            }
            const auto& pool = pools[static_cast<size_t>(component.device_pool_index)];
            if (!pool) {
                RTP_LLM_LOG_WARNING("null device pool %d component=%d group=%d",
                                    component.device_pool_index,
                                    component_index,
                                    layout.component_group_id);
                device_host_layout_ok = false;
                break;
            }

            ResolvedComponentLayout comp_layout;
            comp_layout.component_index   = component_index;
            comp_layout.device_pool_index = component.device_pool_index;
            comp_layout.device_pool       = pool;
            comp_layout.layer_slots       = component.memory_block_layer_tag_slots;
            layout.components.push_back(std::move(comp_layout));
        }

        layout.layout_bytes           = computeLayoutsBlockSize(layout.components);
        layout.has_device_host_layout = device_host_layout_ok && layout.host_pool != nullptr
                                        && !layout.components.empty() && layoutHasAnyLayerSlot(layout.components);
        group_layouts_.push_back(std::move(layout));
    }
}

const ResolvedGroupLayout* CopyEngine::layoutFor(int component_group_id) const {
    if (component_group_id < 0) {
        return nullptr;
    }
    for (const auto& layout : group_layouts_) {
        if (layout.component_group_id == component_group_id) {
            return &layout;
        }
    }
    return nullptr;
}

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
    auto           state      = std::make_shared<TransferHandle::State>(request_id);

    completeRequest(state, execute(desc));
    return TransferHandle(std::move(state));
}

void CopyEngine::completeRequest(const std::shared_ptr<TransferHandle::State>& state, CopyStatus status) {
    std::vector<CopyCompletionCallback> callbacks;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->status = status;
        state->done   = true;
        callbacks.swap(state->callbacks);
    }

    state->cv.notify_all();

    for (const auto& callback : callbacks) {
        callback(status);
    }
}

CopyStatus CopyEngine::execute(const TransferDescriptor& desc) {
    if (desc.component_group_id < 0) {
        RTP_LLM_LOG_WARNING("missing component_group_id");
        return CopyStatus::INVALID_ARGS;
    }
    if (desc.source_tier == Tier::NONE || desc.target_tier == Tier::NONE || desc.source_tier == desc.target_tier) {
        RTP_LLM_LOG_WARNING("invalid transfer tier pair source=%s target=%s",
                            tierName(desc.source_tier),
                            tierName(desc.target_tier));
        return CopyStatus::INVALID_ARGS;
    }

    const ResolvedGroupLayout* layout = layoutFor(desc.component_group_id);
    if (!layout) {
        RTP_LLM_LOG_WARNING("unknown component_group_id=%d", desc.component_group_id);
        return CopyStatus::INVALID_ARGS;
    }

    if (isDeviceHostTransfer(desc.source_tier, desc.target_tier)) {
        if (!layout->has_device_host_layout) {
            RTP_LLM_LOG_WARNING("missing device-host layout group=%d", desc.component_group_id);
            return CopyStatus::INVALID_ARGS;
        }
    } else if ((desc.source_tier == Tier::HOST && desc.target_tier == Tier::DISK)
               || (desc.source_tier == Tier::DISK && desc.target_tier == Tier::HOST)) {
        if (!layout->host_pool) {
            RTP_LLM_LOG_WARNING("missing host_pool for host-disk transfer group=%d", desc.component_group_id);
            return CopyStatus::INVALID_ARGS;
        }
        if (!layout->disk_pool) {
            RTP_LLM_LOG_WARNING("missing disk_pool for host-disk transfer group=%d", desc.component_group_id);
            return CopyStatus::INVALID_ARGS;
        }
    } else {
        RTP_LLM_LOG_WARNING("unsupported transfer tier pair source=%s target=%s",
                            tierName(desc.source_tier),
                            tierName(desc.target_tier));
        return CopyStatus::INVALID_ARGS;
    }

    if (desc.source_tier == Tier::DEVICE && desc.target_tier == Tier::HOST) {
        if (isNullBlockIdx(desc.host_block)) {
            RTP_LLM_LOG_WARNING("D2H descriptor has invalid host target block");
            return CopyStatus::INVALID_ARGS;
        }
        if (desc.device_blocks.size() != layout->components.size()) {
            RTP_LLM_LOG_WARNING("D2H device_blocks(%zu) does not match components(%zu)",
                                desc.device_blocks.size(),
                                layout->components.size());
            return CopyStatus::INVALID_ARGS;
        }
        return device_host_executor_->execute(desc, *layout);
    }

    if (desc.source_tier == Tier::HOST && desc.target_tier == Tier::DEVICE) {
        if (isNullBlockIdx(desc.host_block) || desc.device_blocks.empty()) {
            RTP_LLM_LOG_WARNING("H2D descriptor has invalid source or target block");
            return CopyStatus::INVALID_ARGS;
        }
        if (desc.device_blocks.size() != layout->components.size()) {
            RTP_LLM_LOG_WARNING("H2D device_blocks(%zu) does not match components(%zu)",
                                desc.device_blocks.size(),
                                layout->components.size());
            return CopyStatus::INVALID_ARGS;
        }
        return device_host_executor_->execute(desc, *layout);
    }

    if (desc.source_tier == Tier::HOST && desc.target_tier == Tier::DISK) {
        if (isNullBlockIdx(desc.host_block) || isNullBlockIdx(desc.disk_block)) {
            RTP_LLM_LOG_WARNING("H2Disk descriptor has invalid source or target block");
            return CopyStatus::INVALID_ARGS;
        }
        return host_disk_executor_->execute(desc, *layout);
    }

    if (desc.source_tier == Tier::DISK && desc.target_tier == Tier::HOST) {
        if (isNullBlockIdx(desc.disk_block) || isNullBlockIdx(desc.host_block)) {
            RTP_LLM_LOG_WARNING("Disk2H descriptor has invalid source or target block");
            return CopyStatus::INVALID_ARGS;
        }
        return host_disk_executor_->execute(desc, *layout);
    }

    RTP_LLM_LOG_WARNING("unsupported transfer tier pair source=%s target=%s",
                        tierName(desc.source_tier),
                        tierName(desc.target_tier));
    return CopyStatus::INVALID_ARGS;
}

size_t CopyEngine::computeHostBlockSize(const std::vector<MemoryBlockLayerTagSlot>& slots) {
    size_t total = 0;
    for (const auto& slot : slots) {
        total += slot.stride_bytes;
    }
    return total;
}

}  // namespace rtp_llm
