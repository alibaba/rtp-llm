#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/CopyEngine.h"

#include <condition_variable>
#include <cstring>
#include <utility>

#include <torch/torch.h>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm {

namespace {

CopyResult makeCopyResult(uint64_t request_id,
                          CopyStatusCode status,
                          size_t completed_entries,
                          size_t failed_entries,
                          std::string error_message = "") {
    CopyResult result;
    result.request_id      = request_id;
    result.status          = status;
    result.completed_entries = completed_entries;
    result.failed_entries    = failed_entries;
    result.error_message   = std::move(error_message);
    return result;
}

bool isDeviceHostTransfer(Tier source_tier, Tier target_tier) {
    return (source_tier == Tier::DEVICE && target_tier == Tier::HOST)
           || (source_tier == Tier::HOST && target_tier == Tier::DEVICE);
}

bool hasValidHostBlock(const TransferEntry& entry) {
    return !isNullBlockIdx(entry.host_block);
}

bool hasValidDiskBlock(const TransferEntry& entry) {
    return !isNullBlockIdx(entry.disk_block);
}

bool validAllocatedHostBlock(HostBlockPool& host_pool, BlockIdxType host_block) {
    return !isNullBlockIdx(host_block) && host_block > 0 && host_pool.isAllocated(host_block);
}

bool validAllocatedDiskBlock(DiskBlockPool& disk_pool, BlockIdxType disk_block) {
    return !isNullBlockIdx(disk_block) && disk_block > 0 && disk_pool.isAllocated(disk_block);
}

const char* blockIOStatusName(BlockIOStatus status) {
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

}  // namespace

struct TransferHandle::State {
    explicit State(uint64_t id): request_id(id) {
        result.request_id = id;
    }

    uint64_t request_id{0};
    bool     done{false};

    CopyResult                         result;
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
        return makeCopyResult(0, CopyStatusCode::INVALID_DESCRIPTOR, 0, 0, "invalid transfer handle");
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

CopyEngine::CopyEngine(CopyEngineTransferResources resources): resources_(std::move(resources)) {}

TransferHandle CopyEngine::submit(const TransferDescriptor& desc, TransferSubmitOptions options) {
    const uint64_t request_id = next_request_id_.fetch_add(1);
    auto           state      = std::make_shared<TransferHandle::State>(request_id);

    auto result = executeTransfer(desc, options, request_id);
    completeRequest(state, std::move(result));
    return TransferHandle(std::move(state));
}

void CopyEngine::setTransferResources(CopyEngineTransferResources resources) {
    std::lock_guard<std::mutex> lock(resources_mutex_);
    resources_ = std::move(resources);
}

std::vector<MemoryBlockLayerTagSlot> CopyEngine::resolveLayerSlots(int component_group_id) const {
    std::function<std::vector<MemoryBlockLayerTagSlot>(int)> resolver;
    {
        std::lock_guard<std::mutex> lock(resources_mutex_);
        resolver = resources_.layer_slots_resolver;
    }
    return resolver ? resolver(component_group_id) : std::vector<MemoryBlockLayerTagSlot>{};
}

std::shared_ptr<HostBlockPool> CopyEngine::resolveHostPool(int component_group_id) const {
    std::function<std::shared_ptr<HostBlockPool>(int)> resolver;
    {
        std::lock_guard<std::mutex> lock(resources_mutex_);
        resolver = resources_.host_pool_resolver;
    }
    return resolver ? resolver(component_group_id) : nullptr;
}

std::shared_ptr<DiskBlockPool> CopyEngine::resolveDiskPool(int component_group_id) const {
    std::function<std::shared_ptr<DiskBlockPool>(int)> resolver;
    {
        std::lock_guard<std::mutex> lock(resources_mutex_);
        resolver = resources_.disk_pool_resolver;
    }
    return resolver ? resolver(component_group_id) : nullptr;
}

DeviceBufferResolver CopyEngine::resolveDeviceBufferResolver() const {
    std::lock_guard<std::mutex> lock(resources_mutex_);
    return resources_.device_buffer_resolver;
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

CopyResult CopyEngine::executeTransfer(const TransferDescriptor& desc,
                                       const TransferSubmitOptions& options,
                                       uint64_t request_id) {
    if (desc.component_group_id < 0) {
        return makeCopyResult(request_id, CopyStatusCode::INVALID_DESCRIPTOR, 0, 0, "missing component_group_id");
    }

    if (desc.source_tier == Tier::NONE || desc.target_tier == Tier::NONE || desc.source_tier == desc.target_tier) {
        return makeCopyResult(request_id, CopyStatusCode::INVALID_DESCRIPTOR, 0, 0, "invalid transfer tier pair");
    }

    const size_t entry_count = desc.entries.size();
    if (entry_count == 0) {
        return makeCopyResult(request_id, CopyStatusCode::SIZE_MISMATCH, 0, 0, "transfer descriptor has no entries");
    }

    if (isDeviceHostTransfer(desc.source_tier, desc.target_tier)) {
        auto slots     = resolveLayerSlots(desc.component_group_id);
        auto resolver  = resolveDeviceBufferResolver();
        auto host_pool = resolveHostPool(desc.component_group_id);
        if (slots.empty() || !resolver || !host_pool) {
            return makeCopyResult(
                request_id, CopyStatusCode::INVALID_DESCRIPTOR, 0, 0, "missing device-host transfer resources");
        }

        if (desc.source_tier == Tier::DEVICE) {
            size_t completed_entries = 0;
            for (size_t entry_id = 0; entry_id < entry_count; ++entry_id) {
                const auto& entry = desc.entries[entry_id];
                if (!hasValidHostBlock(entry)) {
                    const size_t failed_entries = entry_count - completed_entries;
                    return makeCopyResult(request_id,
                                          CopyStatusCode::INVALID_BLOCK,
                                          completed_entries,
                                          failed_entries,
                                          "D2H descriptor has invalid host target block");
                }
                if (!deviceToHost(entry.device_blocks, entry.host_block, slots, resolver, *host_pool)) {
                    const size_t failed_entries = entry_count - completed_entries;
                    return makeCopyResult(request_id,
                                          completed_entries > 0 && options.require_all_or_none ?
                                              CopyStatusCode::PARTIAL_FAILURE :
                                              CopyStatusCode::DEVICE_IO_ERROR,
                                          completed_entries,
                                          failed_entries,
                                          "D2H copy failed");
                }
                ++completed_entries;
            }
            return makeCopyResult(request_id, CopyStatusCode::OK, completed_entries, 0);
        }

        size_t completed_entries = 0;
        for (size_t entry_id = 0; entry_id < entry_count; ++entry_id) {
            const auto& entry = desc.entries[entry_id];
            if (!hasValidHostBlock(entry) || entry.device_blocks.empty()) {
                const size_t failed_entries = entry_count - completed_entries;
                return makeCopyResult(request_id,
                                      CopyStatusCode::INVALID_BLOCK,
                                      completed_entries,
                                      failed_entries,
                                      "H2D descriptor has invalid source or target block");
            }
            if (entry.device_blocks.size() != slots.size()) {
                const size_t failed_entries = entry_count - completed_entries;
                return makeCopyResult(request_id,
                                      CopyStatusCode::SIZE_MISMATCH,
                                      completed_entries,
                                      failed_entries,
                                      "H2D target block count mismatch");
            }
            if (!hostToDevice(entry.host_block, entry.device_blocks, slots, resolver, *host_pool)) {
                const size_t failed_entries = entry_count - completed_entries;
                return makeCopyResult(request_id,
                                      completed_entries > 0 && options.require_all_or_none ?
                                          CopyStatusCode::PARTIAL_FAILURE :
                                          CopyStatusCode::DEVICE_IO_ERROR,
                                      completed_entries,
                                      failed_entries,
                                      "H2D copy failed");
            }
            ++completed_entries;
        }
        return makeCopyResult(request_id, CopyStatusCode::OK, completed_entries, 0);
    }

    if (desc.source_tier == Tier::HOST && desc.target_tier == Tier::DISK) {
        auto host_pool = resolveHostPool(desc.component_group_id);
        auto disk_pool = resolveDiskPool(desc.component_group_id);
        if (!host_pool || !disk_pool) {
            return makeCopyResult(request_id,
                                  CopyStatusCode::INVALID_DESCRIPTOR,
                                  0,
                                  entry_count,
                                  "missing host-disk transfer resources");
        }

        size_t completed_entries = 0;
        for (size_t entry_id = 0; entry_id < entry_count; ++entry_id) {
            const auto& entry = desc.entries[entry_id];
            if (!hasValidHostBlock(entry) || !hasValidDiskBlock(entry)) {
                const size_t failed_entries = entry_count - completed_entries;
                return makeCopyResult(request_id,
                                      CopyStatusCode::INVALID_BLOCK,
                                      completed_entries,
                                      failed_entries,
                                      "H2Disk descriptor has invalid source or target block");
            }
            if (!hostToDisk(entry.host_block, entry.disk_block, *host_pool, *disk_pool)) {
                const size_t failed_entries = entry_count - completed_entries;
                return makeCopyResult(request_id,
                                      completed_entries > 0 && options.require_all_or_none ?
                                          CopyStatusCode::PARTIAL_FAILURE :
                                          CopyStatusCode::DISK_IO_ERROR,
                                      completed_entries,
                                      failed_entries,
                                      "H2Disk copy failed");
            }
            ++completed_entries;
        }
        return makeCopyResult(request_id, CopyStatusCode::OK, completed_entries, 0);
    }

    if (desc.source_tier == Tier::DISK && desc.target_tier == Tier::HOST) {
        auto host_pool = resolveHostPool(desc.component_group_id);
        auto disk_pool = resolveDiskPool(desc.component_group_id);
        if (!host_pool || !disk_pool) {
            return makeCopyResult(request_id,
                                  CopyStatusCode::INVALID_DESCRIPTOR,
                                  0,
                                  entry_count,
                                  "missing disk-host transfer resources");
        }

        size_t completed_entries = 0;
        for (size_t entry_id = 0; entry_id < entry_count; ++entry_id) {
            const auto& entry = desc.entries[entry_id];
            if (!hasValidDiskBlock(entry) || !hasValidHostBlock(entry)) {
                const size_t failed_entries = entry_count - completed_entries;
                return makeCopyResult(request_id,
                                      CopyStatusCode::INVALID_BLOCK,
                                      completed_entries,
                                      failed_entries,
                                      "Disk2H descriptor has invalid source or target block");
            }
            if (!diskToHost(entry.disk_block, entry.host_block, *host_pool, *disk_pool)) {
                const size_t failed_entries = entry_count - completed_entries;
                return makeCopyResult(request_id,
                                      completed_entries > 0 && options.require_all_or_none ?
                                          CopyStatusCode::PARTIAL_FAILURE :
                                          CopyStatusCode::DISK_IO_ERROR,
                                      completed_entries,
                                      failed_entries,
                                      "Disk2H copy failed");
            }
            ++completed_entries;
        }
        return makeCopyResult(request_id, CopyStatusCode::OK, completed_entries, 0);
    }

    return makeCopyResult(request_id, CopyStatusCode::INVALID_DESCRIPTOR, 0, 0, "unsupported transfer tier pair");
}

size_t CopyEngine::computeHostBlockSize(const std::vector<MemoryBlockLayerTagSlot>& slots) {
    size_t total = 0;
    for (const auto& slot : slots) {
        total += slot.stride_bytes;
    }
    return total;
}

bool CopyEngine::deviceToHost(const std::vector<BlockIdxType>&            device_blocks,
                              BlockIdxType                                host_block,
                              const std::vector<MemoryBlockLayerTagSlot>& slots,
                              const DeviceBufferResolver&                 resolver,
                              HostBlockPool&                              host_pool) {
    if (!validAllocatedHostBlock(host_pool, host_block)) {
        RTP_LLM_LOG_WARNING("CopyEngine::deviceToHost: invalid or unallocated host block %d", host_block);
        return false;
    }
    if (device_blocks.size() != slots.size()) {
        RTP_LLM_LOG_WARNING(
            "CopyEngine::deviceToHost: device_blocks(%zu) != slots(%zu)", device_blocks.size(), slots.size());
        return false;
    }

    void* host_base = host_pool.blockBuffer(host_block).addr;
    if (!host_base) {
        RTP_LLM_LOG_WARNING("CopyEngine::deviceToHost: null host address for block %d", host_block);
        return false;
    }

    bool use_cuda_copy = false;
    for (size_t i = 0; i < slots.size(); ++i) {
        if (!isNullBlockIdx(device_blocks[i])) {
            BlockInfo info = resolver(slots[i].layer_id, device_blocks[i]);
            if (info.is_cuda) {
                use_cuda_copy = true;
            }
            break;
        }
    }

    if (use_cuda_copy) {
        std::vector<torch::Tensor> dst_buffers;
        std::vector<torch::Tensor> src_buffers;
        auto                       cpu_opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
        auto                       gpu_opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);

        size_t byte_off = 0;
        for (size_t i = 0; i < slots.size(); ++i) {
            const auto& slot         = slots[i];
            const auto  device_block = device_blocks[i];

            if (isNullBlockIdx(device_block)) {
                std::memset(static_cast<uint8_t*>(host_base) + byte_off, 0, slot.stride_bytes);
                byte_off += slot.stride_bytes;
                continue;
            }

            BlockInfo device_info = resolver(slot.layer_id, device_block);
            if (!device_info.addr || device_info.size_bytes == 0) {
                RTP_LLM_LOG_WARNING("CopyEngine::deviceToHost: null device buffer at slot %zu "
                                    "(layer=%d, block=%d)",
                                    i,
                                    slot.layer_id,
                                    device_block);
                return false;
            }

            const size_t copy_bytes = std::min(device_info.size_bytes, slot.stride_bytes);
            src_buffers.push_back(torch::from_blob(device_info.addr, {static_cast<int64_t>(copy_bytes)}, gpu_opts));
            dst_buffers.push_back(torch::from_blob(
                static_cast<uint8_t*>(host_base) + byte_off, {static_cast<int64_t>(copy_bytes)}, cpu_opts));
            if (copy_bytes < slot.stride_bytes) {
                std::memset(
                    static_cast<uint8_t*>(host_base) + byte_off + copy_bytes, 0, slot.stride_bytes - copy_bytes);
            }
            byte_off += slot.stride_bytes;
        }

        if (!dst_buffers.empty()) {
            MultiCopyParams mc{dst_buffers, src_buffers};
            execNoBlockCopy(mc);
        }
    } else {
        size_t byte_off = 0;
        for (size_t i = 0; i < slots.size(); ++i) {
            const auto& slot         = slots[i];
            const auto  device_block = device_blocks[i];

            if (isNullBlockIdx(device_block)) {
                std::memset(static_cast<uint8_t*>(host_base) + byte_off, 0, slot.stride_bytes);
                byte_off += slot.stride_bytes;
                continue;
            }

            BlockInfo device_info = resolver(slot.layer_id, device_block);
            if (!device_info.addr || device_info.size_bytes == 0) {
                RTP_LLM_LOG_WARNING("CopyEngine::deviceToHost: null device buffer at slot %zu "
                                    "(layer=%d, block=%d)",
                                    i,
                                    slot.layer_id,
                                    device_block);
                return false;
            }

            const size_t copy_bytes = std::min(device_info.size_bytes, slot.stride_bytes);
            std::memcpy(static_cast<uint8_t*>(host_base) + byte_off, device_info.addr, copy_bytes);

            if (copy_bytes < slot.stride_bytes) {
                std::memset(
                    static_cast<uint8_t*>(host_base) + byte_off + copy_bytes, 0, slot.stride_bytes - copy_bytes);
            }
            byte_off += slot.stride_bytes;
        }
    }

    RTP_LLM_LOG_DEBUG("CopyEngine::deviceToHost: packed %zu slots into host_block=%d", slots.size(), host_block);
    return true;
}

bool CopyEngine::hostToDevice(BlockIdxType                                host_block,
                              const std::vector<BlockIdxType>&            device_blocks,
                              const std::vector<MemoryBlockLayerTagSlot>& slots,
                              const DeviceBufferResolver&                 resolver,
                              HostBlockPool&                              host_pool) {
    if (!validAllocatedHostBlock(host_pool, host_block)) {
        RTP_LLM_LOG_WARNING("CopyEngine::hostToDevice: invalid or unallocated host block %d", host_block);
        return false;
    }
    if (device_blocks.size() != slots.size()) {
        RTP_LLM_LOG_WARNING(
            "CopyEngine::hostToDevice: device_blocks(%zu) != slots(%zu)", device_blocks.size(), slots.size());
        return false;
    }

    const void* host_base = host_pool.blockBuffer(host_block).addr;
    if (!host_base) {
        RTP_LLM_LOG_WARNING("CopyEngine::hostToDevice: null host address for block %d", host_block);
        return false;
    }

    bool use_cuda_copy = false;
    for (size_t i = 0; i < slots.size(); ++i) {
        if (!isNullBlockIdx(device_blocks[i])) {
            BlockInfo info = resolver(slots[i].layer_id, device_blocks[i]);
            if (info.is_cuda) {
                use_cuda_copy = true;
            }
            break;
        }
    }

    if (use_cuda_copy) {
        std::vector<torch::Tensor> dst_buffers;
        std::vector<torch::Tensor> src_buffers;
        auto                       cpu_opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
        auto                       gpu_opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);

        size_t byte_off = 0;
        for (size_t i = 0; i < slots.size(); ++i) {
            const auto& slot         = slots[i];
            const auto  device_block = device_blocks[i];

            if (isNullBlockIdx(device_block)) {
                byte_off += slot.stride_bytes;
                continue;
            }

            BlockInfo device_info = resolver(slot.layer_id, device_block);
            if (!device_info.addr || device_info.size_bytes == 0) {
                RTP_LLM_LOG_WARNING("CopyEngine::hostToDevice: null device buffer at slot %zu "
                                    "(layer=%d, block=%d)",
                                    i,
                                    slot.layer_id,
                                    device_block);
                return false;
            }

            const size_t copy_bytes = std::min(device_info.size_bytes, slot.stride_bytes);
            src_buffers.push_back(torch::from_blob(
                const_cast<void*>(static_cast<const void*>(static_cast<const uint8_t*>(host_base) + byte_off)),
                {static_cast<int64_t>(copy_bytes)},
                cpu_opts));
            dst_buffers.push_back(torch::from_blob(device_info.addr, {static_cast<int64_t>(copy_bytes)}, gpu_opts));

            byte_off += slot.stride_bytes;
        }

        if (!dst_buffers.empty()) {
            MultiCopyParams mc{dst_buffers, src_buffers};
            execNoBlockCopy(mc);
        }
    } else {
        size_t byte_off = 0;
        for (size_t i = 0; i < slots.size(); ++i) {
            const auto& slot         = slots[i];
            const auto  device_block = device_blocks[i];

            if (isNullBlockIdx(device_block)) {
                byte_off += slot.stride_bytes;
                continue;
            }

            BlockInfo device_info = resolver(slot.layer_id, device_block);
            if (!device_info.addr || device_info.size_bytes == 0) {
                RTP_LLM_LOG_WARNING("CopyEngine::hostToDevice: null device buffer at slot %zu "
                                    "(layer=%d, block=%d)",
                                    i,
                                    slot.layer_id,
                                    device_block);
                return false;
            }

            const size_t copy_bytes = std::min(device_info.size_bytes, slot.stride_bytes);
            std::memcpy(device_info.addr, static_cast<const uint8_t*>(host_base) + byte_off, copy_bytes);

            byte_off += slot.stride_bytes;
        }
    }

    RTP_LLM_LOG_DEBUG(
        "CopyEngine::hostToDevice: unpacked host_block=%d into %zu device blocks", host_block, device_blocks.size());
    return true;
}

bool CopyEngine::hostToDisk(BlockIdxType   host_block,
                            BlockIdxType   disk_block,
                            HostBlockPool& host_pool,
                            DiskBlockPool& disk_pool) {
    if (!validAllocatedHostBlock(host_pool, host_block)) {
        RTP_LLM_LOG_WARNING("CopyEngine::hostToDisk: invalid or unallocated host block %d", host_block);
        return false;
    }
    if (!validAllocatedDiskBlock(disk_pool, disk_block)) {
        RTP_LLM_LOG_WARNING("CopyEngine::hostToDisk: invalid or unallocated disk block %d", disk_block);
        return false;
    }

    const void*  host_base = host_pool.blockBuffer(host_block).addr;
    const size_t bytes     = std::min(host_pool.strideBytes(), disk_pool.strideBytes());
    const auto   status    = disk_pool.write(disk_block, host_base, bytes);
    if (status != BlockIOStatus::OK) {
        RTP_LLM_LOG_WARNING("CopyEngine::hostToDisk: write failed, host=%d, disk=%d, status=%s",
                            host_block,
                            disk_block,
                            blockIOStatusName(status));
        return false;
    }
    return true;
}

bool CopyEngine::diskToHost(BlockIdxType   disk_block,
                            BlockIdxType   host_block,
                            HostBlockPool& host_pool,
                            DiskBlockPool& disk_pool) {
    if (!validAllocatedHostBlock(host_pool, host_block)) {
        RTP_LLM_LOG_WARNING("CopyEngine::diskToHost: invalid or unallocated host block %d", host_block);
        return false;
    }
    if (!validAllocatedDiskBlock(disk_pool, disk_block)) {
        RTP_LLM_LOG_WARNING("CopyEngine::diskToHost: invalid or unallocated disk block %d", disk_block);
        return false;
    }

    void*        host_base = host_pool.blockBuffer(host_block).addr;
    const size_t bytes     = std::min(host_pool.strideBytes(), disk_pool.strideBytes());
    const auto   status    = disk_pool.read(disk_block, host_base, bytes);
    if (status != BlockIOStatus::OK) {
        RTP_LLM_LOG_WARNING("CopyEngine::diskToHost: read failed, disk=%d, host=%d, status=%s",
                            disk_block,
                            host_block,
                            blockIOStatusName(status));
        return false;
    }
    return true;
}

}  // namespace rtp_llm
