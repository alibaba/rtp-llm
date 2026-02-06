#pragma once

#include <utility>

namespace rtp_llm {

inline std::pair<int, int> getCachedTokenBlockSwapIdx(int cur_seq_length, int nxt_seq_length, int seq_size_per_block) {
    // check if generate tokens has cached tokens
    // we only checked tokens except first and last tokens
    // if first token is cached token, don't need to swap
    // if last token is cached token, it will be swapped in the next step, so skip here
    if ((cur_seq_length + 1) % seq_size_per_block <= (nxt_seq_length + seq_size_per_block - 1) % seq_size_per_block) {
        return {0, 0};
    }

    int base_block_idx       = cur_seq_length / seq_size_per_block;
    int cached_token_offset  = seq_size_per_block - cur_seq_length % seq_size_per_block - 1;
    int cached_src_block_idx = base_block_idx + cached_token_offset;
    int cached_des_block_idx = nxt_seq_length / seq_size_per_block - 1;

    return {cached_src_block_idx, cached_des_block_idx};
}

inline std::pair<int, int> getFinalTokenBlockSwapIdx(int cur_seq_length, int nxt_seq_length, int seq_size_per_block) {
    int base_block_idx   = cur_seq_length / seq_size_per_block;
    int accept_token_num = nxt_seq_length - cur_seq_length;
    int src_block_idx    = base_block_idx + accept_token_num - 1;
    int des_block_idx    = (nxt_seq_length - 1) / seq_size_per_block;

    return {src_block_idx, des_block_idx};
}

}  // namespace rtp_llm