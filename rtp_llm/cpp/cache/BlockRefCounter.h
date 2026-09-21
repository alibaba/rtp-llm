#pragma once

#include <cstdint>
#include <vector>
#include <limits>
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

class BlockRefCounter {
public:
    BlockRefCounter() {}
    BlockRefCounter(int block_nums) {
        init(block_nums);
    }

    void init(int block_nums) {
        RTP_LLM_CHECK(block_nums > 0);
        ref_counter.assign(block_nums, 0);
        total_block_nums_ = block_nums - 1;
        busy_block_num_ = 0;
    }

    int getRefCounter(int block_index) const {
        RTP_LLM_CHECK(block_index > 0);
        return ref_counter.at(block_index);
    }

    // The pool serializes updates. Dense, bounded IDs need no hash lookup.
    void updateRefCounter(int block_index, int delta) {
        RTP_LLM_CHECK(block_index > 0 && static_cast<size_t>(block_index) < ref_counter.size());
        auto& counter = ref_counter[block_index];
        const int64_t next = static_cast<int64_t>(counter) + delta;
        RTP_LLM_CHECK(next >= 0 && next <= std::numeric_limits<int>::max());
        if (counter == 0 && next != 0) {
            ++busy_block_num_;
        } else if (counter != 0 && next == 0) {
            --busy_block_num_;
        }
        counter = static_cast<int>(next);
    }

    void incrementRefCounter(const std::vector<int>& block_indices) {
        for (int index : block_indices) {
            updateRefCounter(index, 1);
        }
    }

    void decrementRefCounter(const std::vector<int>& block_indices) {
        decrementRefCounterImpl<false>(block_indices);
    }

    std::vector<int> decrementRefCounterWithFreeInfo(const std::vector<int>& block_indices) {
        auto free_block = decrementRefCounterImpl<true>(block_indices);
        return free_block;
    }

    uint32_t busyBlockNum() const {
        return busy_block_num_;
    }

    uint32_t freeBlockNum() const {
        return total_block_nums_ - busy_block_num_;
    }

private:
    template<bool with_free_info>
    std::vector<int> decrementRefCounterImpl(const std::vector<int>& block_indices) {
        std::vector<int> free_blocks;
        if constexpr (with_free_info) {
            free_blocks.reserve(block_indices.size());
        }

        for (int index : block_indices) {
            RTP_LLM_CHECK(index > 0 && static_cast<size_t>(index) < ref_counter.size());
            auto& counter = ref_counter[index];
            if (counter == 0) {
                RTP_LLM_FAIL("block:%d decrease zero ref count.", index);
                return {};
            } else {
                counter--;
                if (counter == 0) {
                    if constexpr (with_free_info) {
                        free_blocks.push_back(index);
                    }
                    busy_block_num_--;
                }
            }
        }
        return free_blocks;
    }

private:
    std::vector<int>             ref_counter;
    uint32_t                     busy_block_num_ = 0;
    uint32_t                     total_block_nums_ = 0;
};

}  // namespace rtp_llm
