#include "rtp_llm/cpp/normal_engine/PrefillWarmupBatchSizing.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace rtp_llm {

PrefillWarmupBatchSizing calculatePrefillWarmupBatchSizing(size_t max_seq_len,
                                                           size_t configured_max_batch_tokens,
                                                           size_t max_context_batch_size) {
    if (max_seq_len == 0) {
        throw std::invalid_argument("prefill warmup max_seq_len must be positive, got 0");
    }

    size_t max_batch_tokens = configured_max_batch_tokens;
    if (max_batch_tokens == 0) {
        // Mirrors finalize_scheduler_config (rtp_llm/config/engine_config.py), which
        // normalises max_batch_tokens_size = max_context_batch_size * max_seq_len when
        // unset -- so through the Python server entry this branch is dead. It is NOT
        // equivalent: the max(1, ...) below also covers max_context_batch_size == 0,
        // which the Python side would normalise to a zero token budget. Kept for
        // entrypoints that build RuntimeConfig without the Python normalisation.
        if (max_context_batch_size > std::numeric_limits<size_t>::max() / max_seq_len) {
            throw std::overflow_error("prefill warmup token budget overflow: max_context_batch_size="
                                      + std::to_string(max_context_batch_size) + " * max_seq_len="
                                      + std::to_string(max_seq_len) + " exceeds size_t");
        }
        max_batch_tokens = std::max<size_t>(1, max_context_batch_size) * max_seq_len;
    }
    const size_t rounded_sequences =
        max_batch_tokens / max_seq_len + static_cast<size_t>(max_batch_tokens % max_seq_len != 0);
    size_t num_sequences = std::max<size_t>(1, rounded_sequences);
    if (max_context_batch_size > 0 && num_sequences > max_context_batch_size) {
        num_sequences    = max_context_batch_size;
        max_batch_tokens = num_sequences * max_seq_len;
    }
    return {max_batch_tokens, num_sequences};
}

}  // namespace rtp_llm
