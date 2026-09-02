#pragma once

#include <vector>

namespace rtp_llm {

inline std::vector<int> draftPrefillCaptureSeqLens(const std::vector<int>& decode_batch_sizes, int num_tokens_per_bs) {
    std::vector<int> seq_lens;
    seq_lens.reserve(decode_batch_sizes.size());
    for (int batch_size : decode_batch_sizes) {
        seq_lens.push_back(batch_size * num_tokens_per_bs);
    }
    return seq_lens;
}

}  // namespace rtp_llm
