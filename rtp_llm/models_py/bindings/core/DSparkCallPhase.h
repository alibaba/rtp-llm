#pragma once

#include <cstdint>

namespace rtp_llm {

// Explicit DSpARK draft-call identity. Proposal and commit are orthogonal to
// attention shape: both are prefill-style calls, but only commit carries
// target features and mutates the persistent feature KV. Keep this out of
// tensor-presence heuristics so CUDA-graph capture can select the right path.
enum class DSparkCallPhase : int32_t {
    NONE    = 0,
    PROPOSE = 1,
    COMMIT  = 2,
};

inline const char* dsparkCallPhaseName(DSparkCallPhase phase) {
    switch (phase) {
        case DSparkCallPhase::PROPOSE:
            return "propose";
        case DSparkCallPhase::COMMIT:
            return "commit";
        default:
            return "none";
    }
}

}  // namespace rtp_llm
