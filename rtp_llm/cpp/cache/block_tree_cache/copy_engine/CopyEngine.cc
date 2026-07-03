#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/CopyEngine.h"

#include <cstring>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

CopyEngine::CopyEngine(std::shared_ptr<HostBlockPool> host_pool, std::shared_ptr<DiskBlockPool> disk_pool):
    host_pool_(std::move(host_pool)), disk_pool_(std::move(disk_pool)) {}

size_t CopyEngine::computeHostBlockSize(const std::vector<MemoryBlockLayerTagSlot>& slots) {
    size_t total = 0;
    for (const auto& slot : slots) {
        total += slot.stride_bytes;
    }
    return total;
}

// ---- Device ↔ Host ----

bool CopyEngine::deviceToHost(const std::vector<BlockIdxType>&            device_blocks,
                              BlockIdxType                                host_block,
                              const std::vector<MemoryBlockLayerTagSlot>& slots,
                              const DeviceBufferResolver&                 resolver) {
    if (!host_pool_ || !host_pool_->validBlock(host_block)) {
        RTP_LLM_LOG_WARNING("CopyEngine::deviceToHost: invalid host block %d", host_block);
        return false;
    }
    if (device_blocks.size() != slots.size()) {
        RTP_LLM_LOG_WARNING(
            "CopyEngine::deviceToHost: device_blocks(%zu) != slots(%zu)", device_blocks.size(), slots.size());
        return false;
    }

    void* host_base = host_pool_->blockAddr(host_block);
    if (!host_base) {
        RTP_LLM_LOG_WARNING("CopyEngine::deviceToHost: null host address for block %d", host_block);
        return false;
    }

    size_t byte_off = 0;
    for (size_t i = 0; i < slots.size(); ++i) {
        const auto& slot         = slots[i];
        const auto  device_block = device_blocks[i];

        if (isNullBlockIdx(device_block)) {
            // Skip this slot — zero-fill
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

        // Zero-fill remaining if device buffer is smaller than stride
        if (copy_bytes < slot.stride_bytes) {
            std::memset(static_cast<uint8_t*>(host_base) + byte_off + copy_bytes, 0, slot.stride_bytes - copy_bytes);
        }

        byte_off += slot.stride_bytes;
    }

    RTP_LLM_LOG_DEBUG("CopyEngine::deviceToHost: packed %zu slots into host_block=%d, total_bytes=%zu",
                      slots.size(),
                      host_block,
                      byte_off);
    return true;
}

bool CopyEngine::hostToDevice(BlockIdxType                                host_block,
                              const std::vector<BlockIdxType>&            device_blocks,
                              const std::vector<MemoryBlockLayerTagSlot>& slots,
                              const DeviceBufferResolver&                 resolver) {
    if (!host_pool_ || !host_pool_->validBlock(host_block)) {
        RTP_LLM_LOG_WARNING("CopyEngine::hostToDevice: invalid host block %d", host_block);
        return false;
    }
    if (device_blocks.size() != slots.size()) {
        RTP_LLM_LOG_WARNING(
            "CopyEngine::hostToDevice: device_blocks(%zu) != slots(%zu)", device_blocks.size(), slots.size());
        return false;
    }

    const void* host_base = host_pool_->blockAddr(host_block);
    if (!host_base) {
        RTP_LLM_LOG_WARNING("CopyEngine::hostToDevice: null host address for block %d", host_block);
        return false;
    }

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

    RTP_LLM_LOG_DEBUG("CopyEngine::hostToDevice: unpacked host_block=%d into %zu device blocks, "
                      "total_bytes=%zu",
                      host_block,
                      device_blocks.size(),
                      byte_off);
    return true;
}

// ---- Host ↔ Disk ----

bool CopyEngine::hostToDisk(BlockIdxType host_block, int32_t disk_slot) {
    if (!host_pool_ || !host_pool_->validBlock(host_block)) {
        RTP_LLM_LOG_WARNING("CopyEngine::hostToDisk: invalid host block %d", host_block);
        return false;
    }
    if (!disk_pool_ || !disk_pool_->validSlot(disk_slot)) {
        RTP_LLM_LOG_WARNING("CopyEngine::hostToDisk: invalid disk slot %d", disk_slot);
        return false;
    }

    const void*  host_base = host_pool_->blockAddr(host_block);
    const size_t bytes     = std::min(host_pool_->blockSizeBytes(), disk_pool_->blockSizeBytes());

    bool ok = disk_pool_->write(disk_slot, host_base, bytes);
    if (ok) {
        RTP_LLM_LOG_DEBUG(
            "CopyEngine::hostToDisk: wrote host_block=%d → disk_slot=%d, bytes=%zu", host_block, disk_slot, bytes);
    } else {
        RTP_LLM_LOG_WARNING("CopyEngine::hostToDisk: write failed, host=%d, disk=%d", host_block, disk_slot);
    }
    return ok;
}

bool CopyEngine::diskToHost(int32_t disk_slot, BlockIdxType host_block) {
    if (!host_pool_ || !host_pool_->validBlock(host_block)) {
        RTP_LLM_LOG_WARNING("CopyEngine::diskToHost: invalid host block %d", host_block);
        return false;
    }
    if (!disk_pool_ || !disk_pool_->validSlot(disk_slot)) {
        RTP_LLM_LOG_WARNING("CopyEngine::diskToHost: invalid disk slot %d", disk_slot);
        return false;
    }

    void*        host_base = host_pool_->blockAddr(host_block);
    const size_t bytes     = std::min(host_pool_->blockSizeBytes(), disk_pool_->blockSizeBytes());

    bool ok = disk_pool_->read(disk_slot, host_base, bytes);
    if (ok) {
        RTP_LLM_LOG_DEBUG(
            "CopyEngine::diskToHost: read disk_slot=%d → host_block=%d, bytes=%zu", disk_slot, host_block, bytes);
    } else {
        RTP_LLM_LOG_WARNING("CopyEngine::diskToHost: read failed, disk=%d, host=%d", disk_slot, host_block);
    }
    return ok;
}

}  // namespace rtp_llm
