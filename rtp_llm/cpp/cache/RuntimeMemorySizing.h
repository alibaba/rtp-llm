#pragma once

#include <cstddef>
#include <cstdint>

namespace rtp_llm {

// Pure sizing math, intentionally dependency-free. safety_ratio and
// no_warmup_floor_bytes carry no product defaults on purpose: every caller
// passes them in from KVCacheConfig, whose fields own those defaults
// (kDefaultRuntimeMemorySafetyRatio / kDefaultRuntimeNoWarmupFloorMiB in
// rtp_llm/cpp/config/ConfigModules.h). Restating them here would duplicate a
// value this layer does not own.
struct RuntimeMemorySizingInput {
    bool   has_warmup               = false;
    size_t configured_reserve_bytes = 0;
    size_t warmup_required_bytes    = 0;
    size_t sampler_required_bytes   = 0;
    size_t total_gpu_bytes          = 0;
    double safety_ratio             = 0.0;
    size_t no_warmup_floor_bytes    = 0;
};

struct RuntimeMemorySizingResult {
    // total_gpu_bytes * safety_ratio. Added on top of the base requirement when a
    // warmup measurement exists; otherwise it is one more floor inside the max()
    // (the pre-warmup-feature "at least 5% of total GPU" minimum).
    size_t safety_ratio_bytes     = 0;
    size_t runtime_required_bytes = 0;
};

struct PrefillWarmupBatchSizing {
    size_t max_batch_tokens = 0;
    size_t num_sequences    = 0;
};

RuntimeMemorySizingResult calculateRuntimeMemorySizing(const RuntimeMemorySizingInput& input);

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
