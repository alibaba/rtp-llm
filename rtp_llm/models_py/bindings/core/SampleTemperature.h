#pragma once

#include <cstdint>

namespace rtp_llm {

// Temperature handling of the CUDA greedy sampler (CudaSampleOp.cc), shared so
// that every component that needs to know whether this round's temperature is
// folded into the force_topk candidate scaling — including upstream decisions
// that keep the logits buffer private / un-cloned — reads the *same* predicate
// and cannot drift from the sampler.
//
// The fold is unconditional: there is no environment switch. The gates in
// greedyTemperatureFoldedForRequest are correctness conditions, not tunables —
// each one marks a round where folding would change the numbers, so that round
// keeps the legacy full-vocabulary in-place pass.
//
// Equivalence the fold relies on: for T > 0 the map x -> x * inv(T) is strictly
// monotone, so top-k over `logits` selects exactly the candidate set top-k over
// the tempered row would select. Scaling those k values right before their
// softmax therefore yields the same distribution as tempering all V logits
// first, minus one [batch, vocab] read+write.

// Per-row logits scale that reproduces batchApplyTemperaturePenalty
// (sampling_penalty_kernels.cu) bit-for-bit: the kernel precomputes
// inv = 1.0f / (temperatures[i] + 1e-6f) and multiplies each logit by it, so
// multiplying the selected candidate values by the same factor keeps them
// identical to the tempered values the legacy pass handed to top-k.
// Note T == 1.0f maps to 0.999999x, not exactly 1.0f — that is precisely why
// the fold is not allowed for an all-T == 1 batch (see below).
inline float sampleTemperatureInvScale(float temperature) {
    return 1.0f / (temperature + 1e-6f);
}

// True when this round folds the temperature into the force_topk candidate
// scaling, i.e. the sampler leaves `logits` untempered in place and a caller
// that declared the buffer private may skip its defensive clone.
//
// Every fact below is a correctness gate (not a tunable): it marks a round in
// which folding would change the numbers, so that round keeps the legacy
// full-vocabulary in-place pass.
//  * force_topk <= 0 — the sampler softmaxes the full row, so there is no
//    narrower consumer for the temperature;
//  * has_do_sample_false_row — those rows are restored to untouched logits
//    before sampling and are never scored by temperature;
//  * has_cum_log_probs / has_return_original_all_probs — those consumers need
//    the full-width probability row, while the fold only produces compact
//    candidate probabilities;
//  * has_active_logits_penalty — repetition/presence/frequency/ngram steps are
//    applied *after* the temperature in the legacy chain and the penalizers are
//    elementwise on logits, so rep(x / T) != rep(x) / T: the fold would change
//    both the distribution and the candidate set;
//  * every row has T == 1 — the legacy chain applies no temperature at all,
//    while folding would inject the kernel's 1/(T + 1e-6) factor;
//  * any row has T == 0 (greedy row) — the scale would be a division by zero.
//
// `temperatures_host` is the batch-sized FP32 host array the sampler reads
// (GreedyParams::temperature). `force_topk` must be the *effective* width the
// sampler will use (0 when its own gates decline). Callers must pass the values
// the sampler will see; the sampler (greedyTemperatureInSelect in
// CudaSampleOp.cc) calls this function too, so the decision has one source of
// truth. It is the caller's job to additionally guarantee that the sampler's
// force_topk branch runs at all (force_topk > 0 passed here already encodes the
// sampler's own output-contract gates).
inline bool greedyTemperatureFoldedForRequest(int64_t      batch_size,
                                              const float* temperatures_host,
                                              int          force_topk,
                                              bool         has_do_sample_false_row,
                                              bool         has_cum_log_probs,
                                              bool         has_return_original_all_probs,
                                              bool         has_active_logits_penalty) {
    if (force_topk <= 0 || has_do_sample_false_row || has_cum_log_probs || has_return_original_all_probs
        || has_active_logits_penalty || batch_size <= 0 || temperatures_host == nullptr) {
        return false;
    }
    bool any_scaled = false;
    for (int64_t i = 0; i < batch_size; ++i) {
        const float temperature = temperatures_host[i];
        if (temperature == 0.0f) {
            return false;
        }
        any_scaled = any_scaled || temperature != 1.0f;
    }
    return any_scaled;
}

}  // namespace rtp_llm
