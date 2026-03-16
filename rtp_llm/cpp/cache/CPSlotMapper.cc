#include "rtp_llm/cpp/cache/CPSlotMapper.h"

#include <algorithm>

namespace rtp_llm {

CPSlotMapper::CPSlotMapper(): cp_rank_(0), cp_size_(1), block_size_(1), virtual_block_size_(1) {}

CPSlotMapper::CPSlotMapper(int cp_rank, int cp_size, int block_size):
    cp_rank_(cp_rank), cp_size_(cp_size), block_size_(block_size), virtual_block_size_(block_size * cp_size) {}

int CPSlotMapper::targetRank(int position) const {
    if (!isSharded()) {
        return 0;
    }
    return (position % virtual_block_size_) % cp_size_;
}

bool CPSlotMapper::isOwned(int position) const {
    if (!isSharded()) {
        return true;
    }
    return targetRank(position) == cp_rank_;
}

int CPSlotMapper::localBlockOffset(int position) const {
    if (!isSharded()) {
        return position % block_size_;
    }
    return (position % virtual_block_size_) / cp_size_;
}

int CPSlotMapper::virtualBlockCount(int seq_len) const {
    if (!isSharded()) {
        return (seq_len + block_size_ - 1) / block_size_;
    }
    return (seq_len + virtual_block_size_ - 1) / virtual_block_size_;
}

int CPSlotMapper::localBlockCount(int seq_len) const {
    return virtualBlockCount(seq_len);
}

int64_t CPSlotMapper::computeSlot(int position, int physical_block_id) const {
    int offset = localBlockOffset(position);
    return static_cast<int64_t>(physical_block_id) * block_size_ + offset;
}

std::vector<int> CPSlotMapper::ownedPositions(int start, int end) const {
    if (!isSharded()) {
        std::vector<int> all;
        all.reserve(end - start);
        for (int i = start; i < end; ++i) {
            all.push_back(i);
        }
        return all;
    }
    std::vector<int> owned;
    owned.reserve((end - start + cp_size_ - 1) / cp_size_);
    for (int i = start; i < end; ++i) {
        if (isOwned(i)) {
            owned.push_back(i);
        }
    }
    return owned;
}

int CPSlotMapper::effectiveSeqLenForAlloc(int actual_seq_len) const {
    if (!isSharded()) {
        return actual_seq_len;
    }
    return localBlockCount(actual_seq_len) * block_size_;
}

std::vector<int> CPSlotMapper::ownedBlockIndices(int total_virtual_blocks) const {
    std::vector<int> indices(total_virtual_blocks);
    for (int i = 0; i < total_virtual_blocks; ++i) {
        indices[i] = i;
    }
    return indices;
}

int CPSlotMapper::ownedBlockCount(int total_virtual_blocks) const {
    return total_virtual_blocks;
}

}  // namespace rtp_llm
