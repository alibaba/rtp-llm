#include "rtp_llm/cpp/normal_engine/ContextParallelUtils.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <numeric>
#include <vector>

using namespace std;

namespace rtp_llm {

std::vector<int> generateZigZagShuffleIndices(int num_padded_input_tokens, int cp_size) {
    std::vector<int> shuffle_indices(num_padded_input_tokens);

    // Sequential indices for short sequences, no need to do zigzag
    if (num_padded_input_tokens <= cp_size) {
        std::iota(shuffle_indices.begin(), shuffle_indices.end(), 0);
        return shuffle_indices;
    }

    // Zig-zag pattern for long sequences
    // pair_size = num_input_tokens / (2 * cp_size)
    const int chunk_num = cp_size * 2;  // cp_size * 2
    RTP_LLM_CHECK_WITH_INFO(
        num_padded_input_tokens % chunk_num == 0,
        "num_padded_input_tokens must be multiple of (cp_size * 2), got num_padded_input_tokens=%d, cp_size=%d",
        num_padded_input_tokens,
        cp_size);

    const int pair_size = num_padded_input_tokens / chunk_num;

    // Direct calculation: O(n) with optimal cache locality
    // Zig-zag: alternately take groups from start (forward) and end (backward) of entire sequence
    for (int i = 0; i < num_padded_input_tokens; ++i) {
        const int pair_idx    = i / pair_size;  // which group in output (0,1,2,3,...)
        const int pair_offset = i % pair_size;  // offset within group
        const int half_pos    = pair_idx >> 1;  // pair_idx / 2

        // Zig-zag: even group indices from start, odd group indices from end
        int target_idx;
        if (pair_idx & 1) {
            // Odd group index: take from end, counting backwards
            // pair_idx=1, half_pos=0: take last group from entire sequence
            // pair_idx=3, half_pos=1: take 2nd-to-last group from entire sequence
            target_idx = num_padded_input_tokens - pair_size * (half_pos + 1) + pair_offset;
        } else {
            // Even group index: take from start, counting forwards
            // pair_idx=0, half_pos=0: take first group from entire sequence
            // pair_idx=2, half_pos=1: take second group from entire sequence
            target_idx = half_pos * pair_size + pair_offset;
        }
        shuffle_indices[i] = target_idx;
    }

    return shuffle_indices;
}

bool contextParallelLoadBalanceSplit(const std::vector<int>& total_input_tokens,
                                     std::vector<int>&       input_tokens,
                                     std::vector<int>&       shuffle_indices,
                                     int                     cp_rank,
                                     int                     cp_size,
                                     int                     cp_chunk_size,
                                     int                     cp_padding_size) {
    const int input_token_size      = static_cast<int>(total_input_tokens.size());
    const int padded_seq_token_size = input_token_size + cp_padding_size;
    RTP_LLM_CHECK(cp_rank >= 0 && cp_rank < cp_size);
    // Generate zig-zag shuffle indices
    const auto zigzag_indices = generateZigZagShuffleIndices(padded_seq_token_size, cp_size);

    // Calculate this rank's chunk range in the shuffled sequence
    const int start_pos = cp_rank * cp_chunk_size;
    const int end_pos   = start_pos + cp_chunk_size;

    // Validate range
    if (start_pos >= padded_seq_token_size) {
        return false;
    }

    // Copy this rank's chunk using shuffled indices
    for (int i = 0, j = start_pos; j < end_pos && i < cp_chunk_size; ++i, ++j) {
        const int src_idx = zigzag_indices[j];
        if (src_idx < input_token_size) {  // Skip padding tokens
            input_tokens[i] = total_input_tokens[src_idx];
        }
        shuffle_indices[i] = src_idx;  // include padding tokens
    }

    return true;
}

}  // namespace rtp_llm