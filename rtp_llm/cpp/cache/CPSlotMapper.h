#pragma once

namespace rtp_llm {

/// Page-level virtual block sharding for Context Parallelism.
///
/// Entire blocks are assigned to ranks: target_rank(pos) = (pos / block_size) % cp_size.
/// Each rank stores only blocks where block_idx % cp_size == cp_rank.
/// Virtual block size is block_size * cp_size (used for cache key grouping).
///
/// Sharded when cp_size > 1.  The default constructor (cp_size=1) gives
/// passthrough behaviour identical to "no CP".
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

    // Translate actual seq_len to an effective value that, when divided by
    // block_size, yields localBlockCount(actual_seq_len).  Use this when
    // feeding seq_len into an allocator that divides by block_size internally.
    int effectiveSeqLenForAlloc(int actual_seq_len) const;

private:
    int cp_rank_            = 0;
    int cp_size_            = 1;
    int block_size_         = 1;
    int virtual_block_size_ = 1;
};

}  // namespace rtp_llm
