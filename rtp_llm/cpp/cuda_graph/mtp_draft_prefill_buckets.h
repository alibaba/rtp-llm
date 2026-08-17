#pragma once

#include <cstddef>
#include <vector>

namespace rtp_llm {

// Capture buckets for MTP draft prefill, expressed as sequence lengths.
//
// One graph per batch size from 1 to max_bs makes the capture count -- and the memory it
// holds -- grow linearly with CONCURRENCY_LIMIT, which is what runs the decode role out of
// device memory once the limit is raised. Replay does not need an exact match:
// tryGetRealGraphPrefillSeqLen() takes the first captured length >= the request via
// lower_bound, and prepareInputs() zero-fills the padded batch slots, so bucketing costs
// padded compute on a replay and nothing else. Bucket the same way decode does, keeping
// max_bs itself so the largest batch still has an exact graph.
//
// Returns an empty vector for degenerate configs (max_bs == 0 or num_tokens_per_bs <= 0)
// so capturePrefill() registers no graph and replay falls back to eager, instead of
// capturing a zero-length graph.
inline std::vector<int> mtpDraftPrefillCaptureSeqLens(size_t max_bs, int num_tokens_per_bs) {
    std::vector<int> seq_lens;
    if (max_bs == 0 || num_tokens_per_bs <= 0) {
        return seq_lens;
    }
    const int max_batch_size = static_cast<int>(max_bs);
    for (int bs = 1; bs < max_batch_size; bs *= 2) {
        seq_lens.push_back(bs * num_tokens_per_bs);
    }
    seq_lens.push_back(max_batch_size * num_tokens_per_bs);
    return seq_lens;
}

}  // namespace rtp_llm
