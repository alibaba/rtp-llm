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
        // assert(block_index < block_nums);
        return ref_counter.at(block_index);
    }

    void incrementRefCounter(const std::vector<int>& block_indices) {
        for (int index : block_indices) {
            ref_counter[index]++;
        }
    }

    void decrementRefCounter(const std::vector<int>& block_indices) {
        for (int index : block_indices) {
            if (ref_counter[index] == 0) {
                RTP_LLM_FAIL("decrease zero ref count.");
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
