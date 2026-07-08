#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/CopyEngine.h"

#include <condition_variable>
#include <cstring>
#include <mutex>
#include <utility>

#include <torch/torch.h>

#include "rtp_llm/cpp/cache/block_tree_cache/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/HostBlockPool.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm {

CopyEngine::CopyEngine(const std::vector<ComponentGroupPtr>& component_groups,
                       const std::vector<Component>&         components):
    components_(components) {
    component_groups_.reserve(component_groups.size());
    for (const auto& group : component_groups) {
        component_groups_.push_back(group);
    }
}

CopyResult CopyEngine::makeCopyResult(uint64_t request_id,
                                      CopyStatus status,
                                      size_t completed_entries,
                                      size_t failed_entries) {
    CopyResult result;
    result.request_id        = request_id;
    result.status            = status;
    result.completed_entries = completed_entries;
    result.failed_entries    = failed_entries;
    return result;
}

bool CopyEngine::isDeviceHostTransfer(Tier source_tier, Tier target_tier) {
    return (source_tier == Tier::DEVICE && target_tier == Tier::HOST)
           || (source_tier == Tier::HOST && target_tier == Tier::DEVICE);
}

bool CopyEngine::validAllocatedHostBlock(HostBlockPool& host_pool, BlockIdxType host_block) {
    return !isNullBlockIdx(host_block) && host_block > 0 && host_pool.isAllocated(host_block);
}

bool CopyEngine::validAllocatedDiskBlock(DiskBlockPool& disk_pool, BlockIdxType disk_block) {
    return !isNullBlockIdx(disk_block) && disk_block > 0 && disk_pool.isAllocated(disk_block);
}

bool CopyEngine::validDeviceBlock(DeviceBlockPool& device_pool, BlockIdxType device_block) {
    return !isNullBlockIdx(device_block) && device_block > 0
           && static_cast<size_t>(device_block) <= device_pool.totalBlocksNum();
}

const char* CopyEngine::blockIOStatusName(BlockIOStatus status) {
    switch (status) {
        case BlockIOStatus::OK:
            return "OK";
        case BlockIOStatus::INVALID_BLOCK:
            return "INVALID_BLOCK";
        case BlockIOStatus::INVALID_SIZE:
            return "INVALID_SIZE";
        case BlockIOStatus::ALIGNMENT_ERROR:
            return "ALIGNMENT_ERROR";
        case BlockIOStatus::IO_ERROR:
            return "IO_ERROR";
        case BlockIOStatus::PARTIAL_FAILURE:
            return "PARTIAL_FAILURE";
    }
    return "UNKNOWN";
}

CopyStatus CopyEngine::blockIOStatusToCopyStatus(BlockIOStatus status) {
    switch (status) {
        case BlockIOStatus::OK:
            return CopyStatus::OK;
        case BlockIOStatus::INVALID_BLOCK:
        case BlockIOStatus::INVALID_SIZE:
        case BlockIOStatus::ALIGNMENT_ERROR:
            return CopyStatus::INVALID_ARGS;
        case BlockIOStatus::IO_ERROR:
            return CopyStatus::DISK_IO_ERROR;
        case BlockIOStatus::PARTIAL_FAILURE:
            return CopyStatus::PARTIAL_FAILURE;
    }
    return CopyStatus::DISK_IO_ERROR;
}

bool CopyEngine::hasAnyLayerSlot(const std::vector<ResolvedComponentLayout>& layouts) {
    for (const auto& layout : layouts) {
        if (!layout.layer_slots.empty()) {
            return true;
        }
    }
    return false;
}

size_t CopyEngine::computeLayoutsBlockSize(const std::vector<ResolvedComponentLayout>& layouts) {
    size_t total = 0;
    for (const auto& layout : layouts) {
        for (const auto& slot : layout.layer_slots) {
            total += slot.stride_bytes;
        }
    }
    return total;
}

void CopyEngine::executeDeviceHostCopyTiles(const std::vector<DeviceHostCopyTile>& tiles, bool device_to_host) {
    std::vector<torch::Tensor> dst_buffers;
    std::vector<torch::Tensor> src_buffers;
    auto                       cpu_device = torch::Device(torch::kCPU);
    auto                       gpu_device = torch::Device(torch::kCUDA);
    auto                       byte_tensor = [](void* addr, size_t bytes, torch::Device device) {
        return torch::from_blob(
            addr, {static_cast<int64_t>(bytes)}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
    };

    for (const auto& tile : tiles) {
        if (tile.bytes == 0) {
            continue;
        }
        // DeviceBlockPool is always CUDA-backed, so every device tile is a GPU<->CPU copy.
        if (device_to_host) {
            dst_buffers.push_back(byte_tensor(tile.host_addr, tile.bytes, cpu_device));
            src_buffers.push_back(byte_tensor(tile.device_addr, tile.bytes, gpu_device));
        } else {
            dst_buffers.push_back(byte_tensor(tile.device_addr, tile.bytes, gpu_device));
            src_buffers.push_back(byte_tensor(tile.host_addr, tile.bytes, cpu_device));
        }
    }

    if (!dst_buffers.empty()) {
        MultiCopyParams mc{dst_buffers, src_buffers};
        execNoBlockCopy(mc);
    }
}

CopyStatus CopyEngine::resolveGroupPools(int component_group_id, ResolvedGroupLayout* out) const {
    if (!out) {
        RTP_LLM_LOG_WARNING("CopyEngine::resolveGroupPools: null output layout");
        return CopyStatus::INVALID_ARGS;
    }
    *out = ResolvedGroupLayout{};

    if (component_group_id < 0 || static_cast<size_t>(component_group_id) >= component_groups_.size()) {
        RTP_LLM_LOG_WARNING("CopyEngine::resolveGroupPools: invalid component_group_id=%d", component_group_id);
        return CopyStatus::INVALID_ARGS;
    }

    const auto& group = component_groups_[static_cast<size_t>(component_group_id)];
    if (!group) {
        RTP_LLM_LOG_WARNING("CopyEngine::resolveGroupPools: null component group %d", component_group_id);
        return CopyStatus::INVALID_ARGS;
    }
    if (group->component_group_id != component_group_id) {
        RTP_LLM_LOG_WARNING("CopyEngine::resolveGroupPools: group id mismatch vector_index=%d group_id=%d",
                            component_group_id,
                            group->component_group_id);
        return CopyStatus::INVALID_ARGS;
    }

    out->component_group_id = component_group_id;
    out->host_pool          = group->hostPool().get();
    out->disk_pool          = group->diskPool().get();
    return CopyStatus::OK;
}

CopyStatus CopyEngine::resolveGroupLayout(int component_group_id, ResolvedGroupLayout* out) const {
    auto status = resolveGroupPools(component_group_id, out);
    if (status != CopyStatus::OK) {
        return status;
    }

    const auto& group = component_groups_[static_cast<size_t>(component_group_id)];
    const auto& pools = group->devicePools();
    out->components.reserve(group->component_indices.size());

    for (int component_index : group->component_indices) {
        if (component_index < 0 || static_cast<size_t>(component_index) >= components_.size()) {
            RTP_LLM_LOG_WARNING("CopyEngine::resolveGroupLayout: invalid component_index=%d group=%d",
                                component_index,
                                component_group_id);
            return CopyStatus::INVALID_ARGS;
        }

        const auto& component = components_[static_cast<size_t>(component_index)];
        if (component.component_group_id != component_group_id) {
            RTP_LLM_LOG_WARNING("CopyEngine::resolveGroupLayout: component[%d] belongs to group %d, expected %d",
                                component_index,
                                component.component_group_id,
                                component_group_id);
            return CopyStatus::INVALID_ARGS;
        }
        if (component.device_pool_index < 0 || static_cast<size_t>(component.device_pool_index) >= pools.size()) {
            RTP_LLM_LOG_WARNING("CopyEngine::resolveGroupLayout: invalid device_pool_index=%d component=%d group=%d",
                                component.device_pool_index,
                                component_index,
                                component_group_id);
            return CopyStatus::INVALID_ARGS;
        }
        const auto& pool = pools[static_cast<size_t>(component.device_pool_index)];
        if (!pool) {
            RTP_LLM_LOG_WARNING("CopyEngine::resolveGroupLayout: null device pool %d component=%d group=%d",
                                component.device_pool_index,
                                component_index,
                                component_group_id);
            return CopyStatus::INVALID_ARGS;
        }

        ResolvedComponentLayout layout;
        layout.component_index    = component_index;
        layout.device_pool_index  = component.device_pool_index;
        layout.device_pool        = pool.get();
        layout.layer_slots        = component.memory_block_layer_tag_slots;
        out->components.push_back(std::move(layout));
    }

    return CopyStatus::OK;
}

CopyStatus CopyEngine::validateDeviceHostLayout(const ResolvedGroupLayout& layout) const {
    if (!layout.host_pool) {
        RTP_LLM_LOG_WARNING("CopyEngine::validateDeviceHostLayout: missing host_pool group=%d",
                            layout.component_group_id);
        return CopyStatus::INVALID_ARGS;
    }
    if (layout.components.empty() || !hasAnyLayerSlot(layout.components)) {
        RTP_LLM_LOG_WARNING("CopyEngine::validateDeviceHostLayout: missing components group=%d",
                            layout.component_group_id);
        return CopyStatus::INVALID_ARGS;
    }
    return CopyStatus::OK;
}

// ---- TransferHandle ----

struct TransferHandle::State {
    explicit State(uint64_t id): request_id(id) {
        result.request_id = id;
    }

    uint64_t request_id{0};
    bool     done{false};

    CopyResult                          result;
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

CopyResult TransferHandle::result() const {
    auto state = state_;
    if (!state) {
        RTP_LLM_LOG_WARNING("TransferHandle::result: invalid transfer handle");
        CopyResult result;
        result.status = CopyStatus::INVALID_ARGS;
        return result;
    }

    wait();
    std::lock_guard<std::mutex> lock(state->mutex);
    return state->result;
}

void TransferHandle::onComplete(CopyCompletionCallback callback) const {
    auto state = state_;
    if (!state || !callback) {
        return;
    }

    CopyResult completed_result;
    bool       run_now = false;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->done) {
            completed_result = state->result;
            run_now          = true;
        } else {
            state->callbacks.push_back(std::move(callback));
        }
    }

    if (run_now) {
        callback(completed_result);
    }
}

// ---- CopyEngine: submit / execute ----

TransferHandle CopyEngine::submit(const TransferDescriptor& desc) {
    const uint64_t request_id = next_request_id_.fetch_add(1);
    auto           state      = std::make_shared<TransferHandle::State>(request_id);

    auto result = execute(desc, request_id);
    completeRequest(state, std::move(result));
    return TransferHandle(std::move(state));
}

void CopyEngine::completeRequest(const std::shared_ptr<TransferHandle::State>& state, CopyResult result) {
    std::vector<CopyCompletionCallback> callbacks;
    CopyResult                          completed_result;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->result = std::move(result);
        state->done   = true;
        callbacks.swap(state->callbacks);
        completed_result = state->result;
    }

    state->cv.notify_all();

    for (const auto& callback : callbacks) {
        callback(completed_result);
    }
}

CopyResult CopyEngine::execute(const TransferDescriptor& desc, uint64_t request_id) {
    if (desc.component_group_id < 0) {
        RTP_LLM_LOG_WARNING("CopyEngine::execute: missing component_group_id");
        return makeCopyResult(request_id, CopyStatus::INVALID_ARGS, 0, 0);
    }
    if (desc.source_tier == Tier::NONE || desc.target_tier == Tier::NONE || desc.source_tier == desc.target_tier) {
        RTP_LLM_LOG_WARNING("CopyEngine::execute: invalid transfer tier pair source=%s target=%s",
                            tierName(desc.source_tier),
                            tierName(desc.target_tier));
        return makeCopyResult(request_id, CopyStatus::INVALID_ARGS, 0, 0);
    }

    ResolvedGroupLayout layout;
    if (isDeviceHostTransfer(desc.source_tier, desc.target_tier)) {
        auto validation = resolveGroupLayout(desc.component_group_id, &layout);
        if (validation != CopyStatus::OK) {
            return makeCopyResult(request_id, validation, 0, 1);
        }
        validation = validateDeviceHostLayout(layout);
        if (validation != CopyStatus::OK) {
            return makeCopyResult(request_id, validation, 0, 1);
        }
    } else if (desc.source_tier == Tier::HOST && desc.target_tier == Tier::DISK) {
        auto validation = resolveGroupPools(desc.component_group_id, &layout);
        if (validation != CopyStatus::OK) {
            return makeCopyResult(request_id, validation, 0, 1);
        }
        if (!layout.host_pool) {
            RTP_LLM_LOG_WARNING("CopyEngine::execute: missing host_pool for host-disk transfer");
            return makeCopyResult(request_id, CopyStatus::INVALID_ARGS, 0, 1);
        }
        if (!layout.disk_pool) {
            RTP_LLM_LOG_WARNING("CopyEngine::execute: missing disk_pool for host-disk transfer");
            return makeCopyResult(request_id, CopyStatus::INVALID_ARGS, 0, 1);
        }
    } else if (desc.source_tier == Tier::DISK && desc.target_tier == Tier::HOST) {
        auto validation = resolveGroupPools(desc.component_group_id, &layout);
        if (validation != CopyStatus::OK) {
            return makeCopyResult(request_id, validation, 0, 1);
        }
        if (!layout.host_pool) {
            RTP_LLM_LOG_WARNING("CopyEngine::execute: missing host_pool for disk-host transfer");
            return makeCopyResult(request_id, CopyStatus::INVALID_ARGS, 0, 1);
        }
        if (!layout.disk_pool) {
            RTP_LLM_LOG_WARNING("CopyEngine::execute: missing disk_pool for disk-host transfer");
            return makeCopyResult(request_id, CopyStatus::INVALID_ARGS, 0, 1);
        }
    } else {
        RTP_LLM_LOG_WARNING("CopyEngine::execute: unsupported transfer tier pair source=%s target=%s",
                            tierName(desc.source_tier),
                            tierName(desc.target_tier));
        return makeCopyResult(request_id, CopyStatus::INVALID_ARGS, 0, 1);
    }

    if (desc.source_tier == Tier::DEVICE && desc.target_tier == Tier::HOST) {
        if (isNullBlockIdx(desc.host_block)) {
            RTP_LLM_LOG_WARNING("CopyEngine::execute: D2H descriptor has invalid host target block");
            return makeCopyResult(request_id, CopyStatus::INVALID_ARGS, 0, 1);
        }
        if (desc.device_blocks.size() != layout.components.size()) {
            RTP_LLM_LOG_WARNING("CopyEngine::execute: D2H device_blocks(%zu) does not match components(%zu)",
                                desc.device_blocks.size(),
                                layout.components.size());
            return makeCopyResult(request_id, CopyStatus::INVALID_ARGS, 0, 1);
        }
        auto status = deviceToHost(desc.device_blocks, desc.host_block, layout);
        return status == CopyStatus::OK ? makeCopyResult(request_id, CopyStatus::OK, 1, 0) :
                                          makeCopyResult(request_id, status, 0, 1);
    }

    if (desc.source_tier == Tier::HOST && desc.target_tier == Tier::DEVICE) {
        if (isNullBlockIdx(desc.host_block) || desc.device_blocks.empty()) {
            RTP_LLM_LOG_WARNING("CopyEngine::execute: H2D descriptor has invalid source or target block");
            return makeCopyResult(request_id, CopyStatus::INVALID_ARGS, 0, 1);
        }
        if (desc.device_blocks.size() != layout.components.size()) {
            RTP_LLM_LOG_WARNING("CopyEngine::execute: H2D device_blocks(%zu) does not match components(%zu)",
                                desc.device_blocks.size(),
                                layout.components.size());
            return makeCopyResult(request_id, CopyStatus::INVALID_ARGS, 0, 1);
        }
        auto status = hostToDevice(desc.host_block, desc.device_blocks, layout);
        return status == CopyStatus::OK ? makeCopyResult(request_id, CopyStatus::OK, 1, 0) :
                                          makeCopyResult(request_id, status, 0, 1);
    }

    if (desc.source_tier == Tier::HOST && desc.target_tier == Tier::DISK) {
        if (isNullBlockIdx(desc.host_block) || isNullBlockIdx(desc.disk_block)) {
            RTP_LLM_LOG_WARNING("CopyEngine::execute: H2Disk descriptor has invalid source or target block");
            return makeCopyResult(request_id, CopyStatus::INVALID_ARGS, 0, 1);
        }
        auto status = hostToDisk(desc.host_block, desc.disk_block, *layout.host_pool, *layout.disk_pool);
        return status == CopyStatus::OK ? makeCopyResult(request_id, CopyStatus::OK, 1, 0) :
                                          makeCopyResult(request_id, status, 0, 1);
    }

    if (desc.source_tier == Tier::DISK && desc.target_tier == Tier::HOST) {
        if (isNullBlockIdx(desc.disk_block) || isNullBlockIdx(desc.host_block)) {
            RTP_LLM_LOG_WARNING("CopyEngine::execute: Disk2H descriptor has invalid source or target block");
            return makeCopyResult(request_id, CopyStatus::INVALID_ARGS, 0, 1);
        }
        auto status = diskToHost(desc.disk_block, desc.host_block, *layout.host_pool, *layout.disk_pool);
        return status == CopyStatus::OK ? makeCopyResult(request_id, CopyStatus::OK, 1, 0) :
                                          makeCopyResult(request_id, status, 0, 1);
    }

    RTP_LLM_LOG_WARNING("CopyEngine::execute: unsupported transfer tier pair source=%s target=%s",
                        tierName(desc.source_tier),
                        tierName(desc.target_tier));
    return makeCopyResult(request_id, CopyStatus::INVALID_ARGS, 0, 0);
}

// ---- Tier-pair primitives ----

size_t CopyEngine::computeHostBlockSize(const std::vector<MemoryBlockLayerTagSlot>& slots) {
    size_t total = 0;
    for (const auto& slot : slots) {
        total += slot.stride_bytes;
    }
    return total;
}

CopyStatus CopyEngine::deviceToHost(const std::vector<BlockIdxType>& device_blocks,
                                    BlockIdxType                     host_block,
                                    const ResolvedGroupLayout&       layout) const {
    auto& host_pool = *layout.host_pool;
    if (!validAllocatedHostBlock(host_pool, host_block)) {
        RTP_LLM_LOG_WARNING("CopyEngine::deviceToHost: invalid or unallocated host block %d", host_block);
        return CopyStatus::INVALID_ARGS;
    }

    void* host_base = host_pool.blockBuffer(host_block).addr;
    if (!host_base) {
        RTP_LLM_LOG_WARNING("CopyEngine::deviceToHost: null host address for block %d", host_block);
        return CopyStatus::DEVICE_IO_ERROR;
    }

    const size_t required_host_bytes = computeLayoutsBlockSize(layout.components);
    if (required_host_bytes != host_pool.payloadBytes()) {
        RTP_LLM_LOG_WARNING("CopyEngine::deviceToHost: component layout bytes %zu != host payload bytes %zu",
                            required_host_bytes,
                            host_pool.payloadBytes());
        return CopyStatus::INVALID_ARGS;
    }

    std::vector<DeviceHostCopyTile> copy_tiles;
    std::vector<HostZeroTile>       zero_tiles;

    size_t host_offset = 0;
    for (size_t component_idx = 0; component_idx < layout.components.size(); ++component_idx) {
        const auto& component    = layout.components[component_idx];
        const auto  device_block = device_blocks[component_idx];

        DeviceBlockPool* device_pool = nullptr;
        if (!isNullBlockIdx(device_block)) {
            device_pool = component.device_pool;
            if (!device_pool) {
                RTP_LLM_LOG_WARNING("CopyEngine::deviceToHost: invalid device_pool_index %d",
                                    component.device_pool_index);
                return CopyStatus::INVALID_ARGS;
            }
            if (!validDeviceBlock(*device_pool, device_block)) {
                RTP_LLM_LOG_WARNING("CopyEngine::deviceToHost: invalid or unallocated device block %d", device_block);
                return CopyStatus::INVALID_ARGS;
            }
        }

        for (const auto& slot : component.layer_slots) {
            if (slot.stride_bytes == 0) {
                RTP_LLM_LOG_WARNING("CopyEngine::deviceToHost: zero-sized layer slot layer=%d", slot.layer_id);
                return CopyStatus::INVALID_ARGS;
            }

            auto* slot_host_addr = static_cast<uint8_t*>(host_base) + host_offset;
            if (isNullBlockIdx(device_block)) {
                zero_tiles.push_back(HostZeroTile{slot_host_addr, slot.stride_bytes});
                host_offset += slot.stride_bytes;
                continue;
            }

            // DeviceBlockPool is always CUDA-backed, so the tile is device-side unconditionally.
            auto   buffers           = device_pool->blockBuffers(slot.layer_id, device_block);
            size_t slot_device_bytes = 0;
            for (const auto& buffer : buffers) {
                if (!buffer.addr || buffer.bytes == 0) {
                    RTP_LLM_LOG_WARNING("CopyEngine::deviceToHost: null device buffer layer=%d block=%d",
                                        slot.layer_id,
                                        device_block);
                    return CopyStatus::DEVICE_IO_ERROR;
                }
                copy_tiles.push_back(DeviceHostCopyTile{slot_host_addr + slot_device_bytes, buffer.addr, buffer.bytes});
                slot_device_bytes += buffer.bytes;
            }

            if (slot_device_bytes != slot.stride_bytes) {
                RTP_LLM_LOG_WARNING("CopyEngine::deviceToHost: device bytes %zu != slot stride %zu layer=%d block=%d",
                                    slot_device_bytes,
                                    slot.stride_bytes,
                                    slot.layer_id,
                                    device_block);
                return CopyStatus::INVALID_ARGS;
            }
            host_offset += slot.stride_bytes;
        }
    }

    for (const auto& zero_tile : zero_tiles) {
        std::memset(zero_tile.host_addr, 0, zero_tile.bytes);
    }
    executeDeviceHostCopyTiles(copy_tiles, /*device_to_host=*/true);

    RTP_LLM_LOG_DEBUG("CopyEngine::deviceToHost: packed %zu components into host_block=%d",
                      layout.components.size(),
                      host_block);
    return CopyStatus::OK;
}

CopyStatus CopyEngine::hostToDevice(BlockIdxType                     host_block,
                                    const std::vector<BlockIdxType>& device_blocks,
                                    const ResolvedGroupLayout&       layout) const {
    auto& host_pool = *layout.host_pool;
    if (!validAllocatedHostBlock(host_pool, host_block)) {
        RTP_LLM_LOG_WARNING("CopyEngine::hostToDevice: invalid or unallocated host block %d", host_block);
        return CopyStatus::INVALID_ARGS;
    }

    const void* host_base = host_pool.blockBuffer(host_block).addr;
    if (!host_base) {
        RTP_LLM_LOG_WARNING("CopyEngine::hostToDevice: null host address for block %d", host_block);
        return CopyStatus::DEVICE_IO_ERROR;
    }

    const size_t required_host_bytes = computeLayoutsBlockSize(layout.components);
    if (required_host_bytes != host_pool.payloadBytes()) {
        RTP_LLM_LOG_WARNING("CopyEngine::hostToDevice: component layout bytes %zu != host payload bytes %zu",
                            required_host_bytes,
                            host_pool.payloadBytes());
        return CopyStatus::INVALID_ARGS;
    }

    std::vector<DeviceHostCopyTile> copy_tiles;

    size_t host_offset = 0;
    for (size_t component_idx = 0; component_idx < layout.components.size(); ++component_idx) {
        const auto& component    = layout.components[component_idx];
        const auto  device_block = device_blocks[component_idx];

        DeviceBlockPool* device_pool = nullptr;
        if (!isNullBlockIdx(device_block)) {
            device_pool = component.device_pool;
            if (!device_pool) {
                RTP_LLM_LOG_WARNING("CopyEngine::hostToDevice: invalid device_pool_index %d",
                                    component.device_pool_index);
                return CopyStatus::INVALID_ARGS;
            }
            if (!validDeviceBlock(*device_pool, device_block)) {
                RTP_LLM_LOG_WARNING("CopyEngine::hostToDevice: invalid or unallocated device block %d", device_block);
                return CopyStatus::INVALID_ARGS;
            }
        }

        for (const auto& slot : component.layer_slots) {
            if (slot.stride_bytes == 0) {
                RTP_LLM_LOG_WARNING("CopyEngine::hostToDevice: zero-sized layer slot layer=%d", slot.layer_id);
                return CopyStatus::INVALID_ARGS;
            }

            auto* slot_host_addr = const_cast<uint8_t*>(static_cast<const uint8_t*>(host_base) + host_offset);
            if (isNullBlockIdx(device_block)) {
                host_offset += slot.stride_bytes;
                continue;
            }

            // DeviceBlockPool is always CUDA-backed, so the tile is device-side unconditionally.
            auto   buffers           = device_pool->blockBuffers(slot.layer_id, device_block);
            size_t slot_device_bytes = 0;
            for (const auto& buffer : buffers) {
                if (!buffer.addr || buffer.bytes == 0) {
                    RTP_LLM_LOG_WARNING("CopyEngine::hostToDevice: null device buffer layer=%d block=%d",
                                        slot.layer_id,
                                        device_block);
                    return CopyStatus::DEVICE_IO_ERROR;
                }
                copy_tiles.push_back(DeviceHostCopyTile{slot_host_addr + slot_device_bytes, buffer.addr, buffer.bytes});
                slot_device_bytes += buffer.bytes;
            }

            if (slot_device_bytes != slot.stride_bytes) {
                RTP_LLM_LOG_WARNING("CopyEngine::hostToDevice: device bytes %zu != slot stride %zu layer=%d block=%d",
                                    slot_device_bytes,
                                    slot.stride_bytes,
                                    slot.layer_id,
                                    device_block);
                return CopyStatus::INVALID_ARGS;
            }
            host_offset += slot.stride_bytes;
        }
    }

    executeDeviceHostCopyTiles(copy_tiles, /*device_to_host=*/false);

    RTP_LLM_LOG_DEBUG(
        "CopyEngine::hostToDevice: unpacked host_block=%d into %zu components", host_block, device_blocks.size());
    return CopyStatus::OK;
}

CopyStatus CopyEngine::hostToDisk(BlockIdxType   host_block,
                                  BlockIdxType   disk_block,
                                  HostBlockPool& host_pool,
                                  DiskBlockPool& disk_pool) const {
    if (!validAllocatedHostBlock(host_pool, host_block)) {
        RTP_LLM_LOG_WARNING("CopyEngine::hostToDisk: invalid or unallocated host block %d", host_block);
        return CopyStatus::INVALID_ARGS;
    }
    if (!validAllocatedDiskBlock(disk_pool, disk_block)) {
        RTP_LLM_LOG_WARNING("CopyEngine::hostToDisk: invalid or unallocated disk block %d", disk_block);
        return CopyStatus::INVALID_ARGS;
    }
    if (host_pool.payloadBytes() != disk_pool.payloadBytes()) {
        RTP_LLM_LOG_WARNING("CopyEngine::hostToDisk: host payload bytes %zu != disk payload bytes %zu",
                            host_pool.payloadBytes(),
                            disk_pool.payloadBytes());
        return CopyStatus::INVALID_ARGS;
    }

    const void*  host_base = host_pool.blockBuffer(host_block).addr;
    if (!host_base) {
        RTP_LLM_LOG_WARNING("CopyEngine::hostToDisk: null host buffer");
        return CopyStatus::DISK_IO_ERROR;
    }
    const size_t bytes     = host_pool.payloadBytes();
    const auto   status    = disk_pool.write(disk_block, host_base, bytes);
    if (status != BlockIOStatus::OK) {
        RTP_LLM_LOG_WARNING("CopyEngine::hostToDisk: write failed, host=%d, disk=%d, status=%s",
                            host_block,
                            disk_block,
                            blockIOStatusName(status));
        return blockIOStatusToCopyStatus(status);
    }
    return CopyStatus::OK;
}

CopyStatus CopyEngine::diskToHost(BlockIdxType   disk_block,
                                  BlockIdxType   host_block,
                                  HostBlockPool& host_pool,
                                  DiskBlockPool& disk_pool) const {
    if (!validAllocatedHostBlock(host_pool, host_block)) {
        RTP_LLM_LOG_WARNING("CopyEngine::diskToHost: invalid or unallocated host block %d", host_block);
        return CopyStatus::INVALID_ARGS;
    }
    if (!validAllocatedDiskBlock(disk_pool, disk_block)) {
        RTP_LLM_LOG_WARNING("CopyEngine::diskToHost: invalid or unallocated disk block %d", disk_block);
        return CopyStatus::INVALID_ARGS;
    }
    if (host_pool.payloadBytes() != disk_pool.payloadBytes()) {
        RTP_LLM_LOG_WARNING("CopyEngine::diskToHost: host payload bytes %zu != disk payload bytes %zu",
                            host_pool.payloadBytes(),
                            disk_pool.payloadBytes());
        return CopyStatus::INVALID_ARGS;
    }

    void*        host_base = host_pool.blockBuffer(host_block).addr;
    if (!host_base) {
        RTP_LLM_LOG_WARNING("CopyEngine::diskToHost: null host buffer");
        return CopyStatus::DISK_IO_ERROR;
    }
    const size_t bytes     = host_pool.payloadBytes();
    const auto   status    = disk_pool.read(disk_block, host_base, bytes);
    if (status != BlockIOStatus::OK) {
        RTP_LLM_LOG_WARNING("CopyEngine::diskToHost: read failed, disk=%d, host=%d, status=%s",
                            disk_block,
                            host_block,
                            blockIOStatusName(status));
        return blockIOStatusToCopyStatus(status);
    }
    return CopyStatus::OK;
}

}  // namespace rtp_llm
