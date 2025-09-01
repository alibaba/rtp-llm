#pragma once

#include <cassert>
#include <unordered_map>
#include <vector>

namespace rtp_llm {

class BlockRefCounter {
public:
    BlockRefCounter() {}
    BlockRefCounter(int block_nums) {
        for (int i = 1; i < block_nums; ++i) {
            ref_counter[i] = 0;
        }
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
        }
    }

    void decrementRefCounter(const std::vector<int>& block_indices) {
        for (int index : block_indices) {
            if (ref_counter[index] == 0) {
                RTP_LLM_FAIL("block:%d decrease zero ref count.", index);
                return;
            } else {
                ref_counter[index]--;
            }
        }
    }

private:
    std::unordered_map<int, int> ref_counter;
};

}  // namespace rtp_llm
