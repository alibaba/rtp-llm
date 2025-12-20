#pragma once

#include <vector>
#include <unordered_map>
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
        total_block_nums_ = block_nums - 1;
        for (int i = 1; i < block_nums; ++i) {
            ref_counter[i] = 0;
        }
        busy_block_num_ = 0;
    }

    int getRefCounter(int block_index) const {
        return ref_counter.at(block_index);
    }

    int getRefCounterUnchecked(int block_index) const {
        auto it = ref_counter.find(block_index);
        return it != ref_counter.end() ? it->second : 0;
    }

    void incrementRefCounter(const std::vector<int>& block_indices) {
        for (int index : block_indices) {
            ref_counter[index]++;
            if (ref_counter[index] == 1) {
                busy_block_num_++;
            }
        }
    }

    void decrementRefCounter(const std::vector<int>& block_indices) {
        for (int index : block_indices) {
            if (ref_counter[index] == 0) {
                RTP_LLM_FAIL("block:%d decrease zero ref count.", index);
                return;
            } else {
                ref_counter[index]--;
                if (ref_counter[index] == 0) {
                    busy_block_num_--;
                }
            }
        }
    }

    uint32_t busyBlockNum() const {
        return busy_block_num_;
    }

    uint32_t freeBlockNum() const {
        return total_block_nums_ - busy_block_num_;
    }

private:
    std::unordered_map<int, int> ref_counter;
    uint32_t                     busy_block_num_ = 0;
    uint32_t                     total_block_nums_;
};

}  // namespace rtp_llm
