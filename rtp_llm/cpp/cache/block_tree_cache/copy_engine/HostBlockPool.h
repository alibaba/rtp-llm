#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "rtp_llm/cpp/cache/BlockRefCounter.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"

namespace rtp_llm {

// Simple CPU memory block pool.
// Manages a contiguous pinned/unpinned CPU buffer divided into fixed-size blocks.
// Used as the Host (L2) tier in BlockTreeCache for packing device blocks.
class HostBlockPool {
public:
    HostBlockPool(size_t block_size_bytes, size_t block_count, bool use_pinned_memory = false);
    ~HostBlockPool();

    bool init();

    // Allocate one block, returns block index or NULL_BLOCK_IDX if full.
    BlockIdxType malloc();

    // Free a block back to the pool.
    void free(BlockIdxType block_idx);

    // Reference counting (compatible with BlockPool semantics).
    void blockCacheReference(BlockIdxType block_idx);
    void blockCacheFree(BlockIdxType block_idx);
    void requestReference(BlockIdxType block_idx);
    void requestFree(BlockIdxType block_idx);

    // Get the raw buffer pointer for a block.
    void*  blockAddr(BlockIdxType block_idx) const;
    size_t blockSizeBytes() const {
        return block_size_bytes_;
    }
    size_t totalBlocks() const {
        return block_count_;
    }
    size_t freeBlocks() const;
    bool   validBlock(BlockIdxType block_idx) const;

    // Get a BlockInfo for a block (for CopyEngine interop).
    BlockInfo blockInfo(BlockIdxType block_idx) const;

private:
    size_t                 block_size_bytes_;
    size_t                 block_count_;
    bool                   use_pinned_memory_;
    void*                  base_ptr_{nullptr};
    std::vector<uint8_t>   backing_buffer_;  // non-pinned backing storage
    mutable std::mutex     mutex_;
    std::set<BlockIdxType> free_blocks_;
    BlockRefCounter        block_cache_ref_counter_;
    BlockRefCounter        request_ref_counter_;
    bool                   initialized_{false};
};

using HostBlockPoolPtr = std::shared_ptr<HostBlockPool>;

}  // namespace rtp_llm
