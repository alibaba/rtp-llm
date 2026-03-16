#pragma once

#include <cstdint>
#include <vector>

namespace rtp_llm {

/// Token-level virtual block sharding for Context Parallelism.
///
/// A virtual block groups `cp_size` physical blocks (one per rank) to hold
/// `block_size * cp_size` consecutive tokens.  Within a virtual block, tokens
/// are interleaved across ranks at token granularity:
///
///   target_rank(position) = (position % virtual_block_size) % cp_size
///
/// Each rank stores exactly `block_size` tokens per virtual block in one
/// physical block.  The local offset within that physical block is:
///
///   local_block_offset(position) = (position % virtual_block_size) / cp_size
///
/// This mapping is independent of per-request quantities so that prefix cache
/// keys remain stable across requests.
class CPSlotMapper {
public:
    CPSlotMapper();
    CPSlotMapper(int cp_rank, int cp_size, int block_size);

    bool isSharded() const {
        return cp_size_ > 1;
    }

    int cpRank() const {
        return cp_rank_;
    }
    int cpSize() const {
        return cp_size_;
    }
    int blockSize() const {
        return block_size_;
    }
    int virtualBlockSize() const {
        return virtual_block_size_;
    }

    int  targetRank(int position) const;
    bool isOwned(int position) const;
    int  localBlockOffset(int position) const;

    int virtualBlockCount(int seq_len) const;
    int localBlockCount(int seq_len) const;

    int64_t computeSlot(int position, int physical_block_id) const;

    std::vector<int> ownedPositions(int start, int end) const;

    // Translate actual seq_len to an effective value that, when divided by
    // block_size, yields localBlockCount(actual_seq_len).  Use this when
    // feeding seq_len into an allocator that divides by block_size internally.
    int effectiveSeqLenForAlloc(int actual_seq_len) const;

    // Block-level helpers used by prefix cache and allocation.
    // With virtual blocks, every rank owns every virtual block (one physical
    // block per virtual block), so these simply produce 0..count-1 ranges.
    std::vector<int> ownedBlockIndices(int total_virtual_blocks) const;
    int              ownedBlockCount(int total_virtual_blocks) const;

private:
    int cp_rank_            = 0;
    int cp_size_            = 1;
    int block_size_         = 1;
    int virtual_block_size_ = 1;
};

}  // namespace rtp_llm
