#pragma once

#include <atomic>
#include <cstdint>
#include <vector>

namespace rtp_llm {

// Per-request persistent state for the remaining-length predictor.
//
// Threading contract: every field except `predicted_total` is written and read
// only by the predictor's single worker thread. `predicted_total` is written
// by the worker and read by engine threads (aux_info fill, schedulers), hence
// atomic. GenerateStream is copy-constructed when a context stream becomes a
// NormalGenerateStream, so the struct provides value-copy semantics for the
// atomic member.
struct LengthPredictorState {
    // Prefill Once anchor: full output length predicted exactly once from the
    // last prompt-token hidden state. Later steps only count down from it.
    bool   anchor_ready = false;
    double anchor_total = 0.0;

    // Decode step (= generated token count) of the last GRU observation.
    // 0 means only the prefill point has been consumed.
    int64_t last_obs_step = 0;

    // [state_dim] fp32 GRU history state; empty until the prefill anchor.
    std::vector<float> gru_state;

    // Latest fused total-length estimate. Negative means "not available yet".
    // remaining(t) is derived as predicted_total - t at read time so exposure
    // stays a countdown between prediction points.
    std::atomic<double> predicted_total{-1.0};

    LengthPredictorState() = default;
    LengthPredictorState(const LengthPredictorState& other) {
        *this = other;
    }
    LengthPredictorState& operator=(const LengthPredictorState& other) {
        if (this != &other) {
            anchor_ready  = other.anchor_ready;
            anchor_total  = other.anchor_total;
            last_obs_step = other.last_obs_step;
            gru_state     = other.gru_state;
            predicted_total.store(other.predicted_total.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }
};

}  // namespace rtp_llm
