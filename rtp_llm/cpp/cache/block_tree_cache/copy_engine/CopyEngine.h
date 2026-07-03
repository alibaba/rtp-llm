#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/TransferDescriptor.h"
#include "rtp_llm/cpp/cache/block_tree_cache/TreeNode.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/DiskBlockPool.h"

namespace rtp_llm {

// Function that resolves a (layer_id, device_block_idx) to a raw buffer pointer + size.
// For GPU: wraps BlockPool::convertIndexToBuffer.
// For testing: wraps a simple CPU buffer map.
using DeviceBufferResolver = std::function<BlockInfo(int layer_id, BlockIdxType device_block_idx)>;

// CopyEngine: stateless data-movement utility between Device/Host/Disk tiers.
//
// D2H (deviceToHost): packs multiple device blocks into one host block,
//   using MemoryBlockLayerTagSlot layout to compute byte offsets.
// H2D (hostToDevice): unpacks one host block into multiple device blocks.
// H2Disk (hostToDisk): writes host block to a disk slot.
// Disk2H (diskToHost): reads disk slot into a host block.
//
// Pool ownership: CopyEngine does NOT own any pool.  BlockPool (HOST) and
// DiskBlockPool are owned by BlockTreeCache (L2/L3 tier resources) and
// passed in as needed.  CopyEngine only performs byte-level copies.
class CopyEngine {
public:
    CopyEngine()  = default;
    ~CopyEngine() = default;

    // ---- Device <-> Host ----

    // Pack device blocks into host block using layer tag slot layout.
    bool deviceToHost(const std::vector<BlockIdxType>&            device_blocks,
                      BlockIdxType                                host_block,
                      const std::vector<MemoryBlockLayerTagSlot>& slots,
                      const DeviceBufferResolver&                 resolver,
                      BlockPool&                                  host_pool);

    // Unpack host block into device blocks (reverse of deviceToHost).
    bool hostToDevice(BlockIdxType                                host_block,
                      const std::vector<BlockIdxType>&            device_blocks,
                      const std::vector<MemoryBlockLayerTagSlot>& slots,
                      const DeviceBufferResolver&                 resolver,
                      BlockPool&                                  host_pool);

    // ---- Host <-> Disk ----

    // Write host block to disk slot.
    bool hostToDisk(BlockIdxType host_block, int32_t disk_slot, BlockPool& host_pool, DiskBlockPool& disk_pool);

    // Read disk slot into host block.
    bool diskToHost(int32_t disk_slot, BlockIdxType host_block, BlockPool& host_pool, DiskBlockPool& disk_pool);

    // ---- Static helpers ----

    // Compute the total packed host block size for a set of slots.
    static size_t computeHostBlockSize(const std::vector<MemoryBlockLayerTagSlot>& slots);
};

using CopyEnginePtr = std::shared_ptr<CopyEngine>;

}  // namespace rtp_llm
