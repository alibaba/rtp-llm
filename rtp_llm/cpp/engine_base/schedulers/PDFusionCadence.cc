#include "rtp_llm/cpp/engine_base/schedulers/PDFusionCadence.h"

#include <algorithm>
#include <stdexcept>

namespace rtp_llm {

PDFusionCadence::PDFusionCadence(int64_t decode_prefill_ratio, bool adaptive, int64_t min_decode_share_pct):
    decode_prefill_ratio_(decode_prefill_ratio),
    adaptive_(adaptive),
    min_decode_share_pct_(min_decode_share_pct),
    decode_since_prefill_(decode_prefill_ratio) {
    if (!adaptive_ && (decode_prefill_ratio_ < 1 || decode_prefill_ratio_ > 1000000)) {
        throw std::invalid_argument("PDFUSION cadence requires DECODE_PREFILL_RATIO in [1, 1000000]");
    }
    if (adaptive_ && (min_decode_share_pct_ < 0 || min_decode_share_pct_ > 99)) {
        throw std::invalid_argument("PDFUSION_CADENCE_MIN_DECODE_SHARE_PCT must be in [0, 99]");
    }
}

PDFusionPhase PDFusionCadence::choose(bool prefill_ready, bool decode_ready) {
    if (!decode_ready) {
        decode_debt_scaled_ = 0;
        return prefill_ready ? PDFusionPhase::PREFILL : PDFusionPhase::IDLE;
    }
    const bool prefill_due = adaptive_ ? decode_debt_scaled_ <= 0 : decode_since_prefill_ >= decode_prefill_ratio_;
    return prefill_ready && prefill_due ? PDFusionPhase::PREFILL : PDFusionPhase::DECODE;
}

void PDFusionCadence::recordExecution(PDFusionPhase phase, int64_t elapsed_us) {
    if (phase == PDFusionPhase::IDLE) {
        return;
    }
    if (!adaptive_) {
        decode_since_prefill_ =
            phase == PDFusionPhase::PREFILL ? 0 : std::min(decode_since_prefill_ + 1, decode_prefill_ratio_);
    } else if (min_decode_share_pct_ > 0) {
        const auto model_us = std::clamp<int64_t>(elapsed_us, 0, 600000000);
        if (phase == PDFusionPhase::PREFILL) {
            decode_debt_scaled_ += model_us * min_decode_share_pct_;
        } else {
            const auto credit = model_us * (100 - min_decode_share_pct_);
            // An idle or Decode-only period can bank at most one Decode step.
            decode_debt_scaled_ = std::max(decode_debt_scaled_ - credit, -credit);
        }
    }
}

}  // namespace rtp_llm
