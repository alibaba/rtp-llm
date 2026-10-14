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

    void init(int block_nums, bool sparse = false) {
        sparse_ = sparse;
        ref_counter.clear();
        total_block_nums_ = block_nums - 1;
        for (int i = 1; !sparse_ && i < block_nums; ++i) {
            ref_counter[i] = 0;
        }
        busy_block_num_ = 0;
    }

    int getRefCounter(int block_index) const {
        auto it = ref_counter.find(block_index);
        return it == ref_counter.end() ? 0 : it->second;
    }

    void incrementRefCounter(const std::vector<int>& block_indices) {
        for (int index : block_indices) {
            auto& count = ref_counter[index];
            if (++count == 1) {
                busy_block_num_++;
            }
        }
    }

    void decrementRefCounter(const std::vector<int>& block_indices) {
        for (int index : block_indices) {
            auto it = ref_counter.find(index);
            if (it == ref_counter.end() || it->second == 0) {
                RTP_LLM_FAIL("block:%d decrease zero ref count.", index);
                return;
            } else {
                if (--it->second == 0) {
                    busy_block_num_--;
                    if (sparse_) {
                        ref_counter.erase(it);
                    }
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
    bool                         sparse_ = false;
    std::unordered_map<int, int> ref_counter;
    uint32_t                     busy_block_num_ = 0;
    uint32_t                     total_block_nums_;
};

}  // namespace rtp_llm
