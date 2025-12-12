#pragma once

#include <cassert>
#include <unordered_map>
#include <vector>
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

class BlockRefCounter {
public:
    BlockRefCounter() {}
    BlockRefCounter(int block_nums) {
        init(block_nums);
    }

    void init(int block_nums) {
        ref_counter.clear();
        for (int i = 1; i < block_nums; ++i) {
            ref_counter[i] = 0;
        }
        busy_block_num_ = 0;
    }

    int getRefCounter(int block_index) const {
        return ref_counter.at(block_index);
    }

    void incrementRefCounter(const std::vector<int>& block_indices) {
        for (int index : block_indices) {
            auto& counter = ref_counter[index];
            counter++;
            if (counter == 1) {
                busy_block_num_++;
            }
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

private:
    template<bool with_free_info>
    std::vector<int> decrementRefCounterImpl(const std::vector<int>& block_indices) {
        std::vector<int> free_blocks;
        if constexpr (with_free_info) {
            free_blocks.reserve(block_indices.size());
        }

        for (int index : block_indices) {
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

    std::unordered_map<int, int> ref_counter;
    uint32_t                     busy_block_num_ = 0;
};

}  // namespace rtp_llm
