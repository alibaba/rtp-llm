#pragma once

#include <cstdint>

namespace rtp_llm {

enum class PDFusionPhase {
    IDLE,
    PREFILL,
    DECODE
};

// A single DP replica uses the same phase choice on all TP ranks via the
// normal input broadcast. Track completed model time on the scheduling rank.
class PDFusionCadence {
public:
    PDFusionCadence(int64_t decode_prefill_ratio, bool adaptive, int64_t min_decode_share_pct);

    PDFusionPhase choose(bool prefill_ready, bool decode_ready);
    void          recordExecution(PDFusionPhase phase, int64_t elapsed_us);
    bool          adaptive() const {
        return adaptive_;
    }

private:
    const int64_t decode_prefill_ratio_;
    const bool    adaptive_;
    const int64_t min_decode_share_pct_;
    int64_t       decode_since_prefill_;
    int64_t       decode_debt_scaled_ = 0;
};

}  // namespace rtp_llm
