#include "rtp_llm/cpp/cache/CPSlotMapper.h"

namespace rtp_llm {

CPSlotMapper::CPSlotMapper(): cp_rank_(0), cp_size_(1), block_size_(1), virtual_block_size_(1) {}

CPSlotMapper::CPSlotMapper(int cp_rank, int cp_size, int block_size):
    cp_rank_(cp_rank), cp_size_(cp_size), block_size_(block_size), virtual_block_size_(block_size * cp_size) {}

int CPSlotMapper::localBlockCount(int seq_len) const {
    // All CP ranks keep the same block count = ceil(total_blocks / cp_size).
    // rank0 is the controller: it allocates blocks and broadcasts block_ids
    // to all ranks.  Using a uniform count simplifies KV cache management —
    // ranks with fewer "real" data blocks simply have unused trailing blocks.
    int total_blocks = (seq_len + block_size_ - 1) / block_size_;
    return (total_blocks + cp_size_ - 1) / cp_size_;
}

int CPSlotMapper::effectiveSeqLenForAlloc(int actual_seq_len) const {
    // Translate to a seq_len that, when the allocator divides by block_size,
    // yields localBlockCount(actual_seq_len).
    return localBlockCount(actual_seq_len) * block_size_;
}

}  // namespace rtp_llm
