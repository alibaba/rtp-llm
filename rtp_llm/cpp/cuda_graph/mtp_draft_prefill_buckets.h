#pragma once

#include <cstddef>
#include <vector>

namespace rtp_llm {

// Capture sequence lengths for MTP draft prefill, one per batch size from 1 to max_bs.
//
// This MUST cover every batch size, not a sparse bucket set. The prefill graph is keyed by
// sequence length (current_real_graph_seq_len) and replay picks a key with lower_bound, but a
// captured graph also bakes in the per-batch attention layout (cu_seqlens / grid dims) of the
// batch it was captured with. Replaying a draft batch that has no exact key lands on a larger
// key's graph whose baked-in layout does not match the real cu_seqlens, which reads out of
// bounds -- observed as a decode-side CUDA illegal memory access at concurrency > 1 once the
// set was thinned to powers of two (batches 3/5/6/7 fell back to the next power and corrupted).
// draft seq_len is always k*num_tokens_per_bs for some 1<=k<=max_bs, so capturing every k gives
// each reachable batch an exact-layout graph.
//
// One graph per batch size makes the capture count grow linearly with CONCURRENCY_LIMIT; that is
// the intended cost. A memory-saving sparse set would need replay to pad the batch up to the
// captured layout (as decode does), which this prefill path does not do, so it is not safe here.
//
// Returns an empty vector for degenerate configs (max_bs == 0 or num_tokens_per_bs <= 0) so
// capturePrefill() registers no graph and replay falls back to eager, instead of capturing a
// zero-length graph.
inline std::vector<int> mtpDraftPrefillCaptureSeqLens(size_t max_bs, int num_tokens_per_bs) {
    std::vector<int> seq_lens;
    if (max_bs == 0 || num_tokens_per_bs <= 0) {
        return seq_lens;
    }
    const int max_batch_size = static_cast<int>(max_bs);
    for (int bs = 1; bs <= max_batch_size; ++bs) {
        seq_lens.push_back(bs * num_tokens_per_bs);
    }
    return seq_lens;
}

}  // namespace rtp_llm
