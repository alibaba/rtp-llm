#pragma once

#include <cstddef>

namespace rtp_llm {

struct PrefillWarmupBatchSizing {
    size_t max_batch_tokens = 0;
    size_t num_sequences    = 0;
};

// Pure warmup batch-shape math, intentionally dependency-free. It lives with the
// engine because it sizes the synthetic prefill forward NormalEngine::prefillWarmUp
// builds -- it is not memory sizing, so it does not belong in
// rtp_llm/cpp/cache/RuntimeMemorySizing.
//
// max_context_batch_size == 0 means "unset", and the two places it is read treat that
// deliberately differently -- do not "unify" them:
//   * deriving a token budget (configured_max_batch_tokens == 0): counted as one
//     sequence, because multiplying by zero would ask for a zero-token warmup.
//   * capping the sequence count: no cap at all, because an unset context-batch limit
//     must not shrink an explicitly configured token budget. Capping to 1 here would
//     silently halve e.g. configured=65536 / max_seq_len=32768 down to one sequence.
//
// num_sequences is a *ceiling* of the token budget: the warmup input can exceed
// configured_max_batch_tokens by up to one full sequence. Deliberate -- warmup sizing
// prefers overestimating the peak (a slightly smaller KV cache) over under-measuring
// it (a runtime OOM). max_batch_tokens in the result is diagnostic output for the
// [PREFILL_WARMUP] log only; nothing sizes the warmup off it.
PrefillWarmupBatchSizing calculatePrefillWarmupBatchSizing(size_t max_seq_len,
                                                           size_t configured_max_batch_tokens,
                                                           size_t max_context_batch_size);
}  // namespace rtp_llm
