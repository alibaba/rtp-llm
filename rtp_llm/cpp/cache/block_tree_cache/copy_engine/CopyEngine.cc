#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/CopyEngine.h"

#include <cstring>

#include <torch/torch.h>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/NoBlockCopy.h"

namespace rtp_llm {

size_t CopyEngine::computeHostBlockSize(const std::vector<MemoryBlockLayerTagSlot>& slots) {
    size_t total = 0;
    for (const auto& slot : slots) {
        total += slot.stride_bytes;
    }
    return total;
}

// ---- Device <-> Host ----

bool CopyEngine::deviceToHost(const std::vector<BlockIdxType>&            device_blocks,
                              BlockIdxType                                host_block,
                              const std::vector<MemoryBlockLayerTagSlot>& slots,
                              const DeviceBufferResolver&                 resolver,
                              BlockPool&                                  host_pool) {
    if (host_block < 1 || host_block > static_cast<BlockIdxType>(host_pool.totalBlocksNum())) {
        RTP_LLM_LOG_WARNING("CopyEngine::deviceToHost: invalid host block %d", host_block);
        return false;
    }
    if (device_blocks.size() != slots.size()) {
        RTP_LLM_LOG_WARNING(
            "CopyEngine::deviceToHost: device_blocks(%zu) != slots(%zu)", device_blocks.size(), slots.size());
        return false;
    }

    void* host_base = host_pool.convertIndexToAddr(0, host_block).kv_addr;
    if (!host_base) {
        RTP_LLM_LOG_WARNING("CopyEngine::deviceToHost: null host address for block %d", host_block);
        return false;
    }

    // Detect whether we need real CUDA copy by checking the first valid device block.
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
        // Real CUDA D2H path: build tensor pairs and call execNoBlockCopy.
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

            // Zero-fill remaining host region if device buffer is smaller than stride
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
        // CPU memcpy path (test/mock mode: is_cuda == false).
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
                              BlockPool&                                  host_pool) {
    if (host_block < 1 || host_block > static_cast<BlockIdxType>(host_pool.totalBlocksNum())) {
        RTP_LLM_LOG_WARNING("CopyEngine::hostToDevice: invalid host block %d", host_block);
        return false;
    }
    if (device_blocks.size() != slots.size()) {
        RTP_LLM_LOG_WARNING(
            "CopyEngine::hostToDevice: device_blocks(%zu) != slots(%zu)", device_blocks.size(), slots.size());
        return false;
    }

    const void* host_base = host_pool.convertIndexToAddr(0, host_block).kv_addr;
    if (!host_base) {
        RTP_LLM_LOG_WARNING("CopyEngine::hostToDevice: null host address for block %d", host_block);
        return false;
    }

    // Detect whether we need real CUDA copy by checking the first valid device block.
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
        // Real CUDA H2D path: build tensor pairs and call execNoBlockCopy.
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
        // CPU memcpy path (test/mock mode: is_cuda == false).
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

// ---- Host <-> Disk ----

bool CopyEngine::hostToDisk(BlockIdxType   host_block,
                            int32_t        disk_slot,
                            BlockPool&     host_pool,
                            DiskBlockPool& disk_pool) {
    if (host_block < 1 || host_block > static_cast<BlockIdxType>(host_pool.totalBlocksNum())) {
        RTP_LLM_LOG_WARNING("CopyEngine::hostToDisk: invalid host block %d", host_block);
        return false;
    }
    if (!disk_pool.validSlot(disk_slot)) {
        RTP_LLM_LOG_WARNING("CopyEngine::hostToDisk: invalid disk slot %d", disk_slot);
        return false;
    }

    const void*  host_base      = host_pool.convertIndexToAddr(0, host_block).kv_addr;
    const size_t host_blk_bytes = host_pool.getTotalSizeBytes() / host_pool.totalBlocksNum();
    const size_t bytes          = std::min(host_blk_bytes, disk_pool.blockSizeBytes());

    bool ok = disk_pool.write(disk_slot, host_base, bytes);
    if (ok) {
        RTP_LLM_LOG_DEBUG(
            "CopyEngine::hostToDisk: wrote host_block=%d -> disk_slot=%d, bytes=%zu", host_block, disk_slot, bytes);
    } else {
        RTP_LLM_LOG_WARNING("CopyEngine::hostToDisk: write failed, host=%d, disk=%d", host_block, disk_slot);
    }
    return ok;
}

bool CopyEngine::diskToHost(int32_t        disk_slot,
                            BlockIdxType   host_block,
                            BlockPool&     host_pool,
                            DiskBlockPool& disk_pool) {
    if (host_block < 1 || host_block > static_cast<BlockIdxType>(host_pool.totalBlocksNum())) {
        RTP_LLM_LOG_WARNING("CopyEngine::diskToHost: invalid host block %d", host_block);
        return false;
    }
    if (!disk_pool.validSlot(disk_slot)) {
        RTP_LLM_LOG_WARNING("CopyEngine::diskToHost: invalid disk slot %d", disk_slot);
        return false;
    }

    void*        host_base      = host_pool.convertIndexToAddr(0, host_block).kv_addr;
    const size_t host_blk_bytes = host_pool.getTotalSizeBytes() / host_pool.totalBlocksNum();
    const size_t bytes          = std::min(host_blk_bytes, disk_pool.blockSizeBytes());

    bool ok = disk_pool.read(disk_slot, host_base, bytes);
    if (ok) {
        RTP_LLM_LOG_DEBUG(
            "CopyEngine::diskToHost: read disk_slot=%d -> host_block=%d, bytes=%zu", disk_slot, host_block, bytes);
    } else {
        RTP_LLM_LOG_WARNING("CopyEngine::diskToHost: read failed, disk=%d, host=%d", disk_slot, host_block);
    }
    return ok;
}

}  // namespace rtp_llm
