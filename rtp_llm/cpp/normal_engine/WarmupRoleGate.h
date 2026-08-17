#pragma once

#include "rtp_llm/cpp/config/RoleTypes.h"

namespace rtp_llm {

// Which roles run the startup warmup forward, and which of those let its measurement reach KV
// sizing. Split out of NormalEngine.cc so the role classification is testable on its own: getting
// it wrong silently changes how much KV cache a whole role class allocates, and the only other
// coverage is at the sizing layer, which sees the consequence rather than the mapping.

// PREFILL and DECODE size their KV cache from the measured warmup peak.
inline bool isPdSeparatedRole(RoleType role_type) {
    return role_type == RoleType::PREFILL || role_type == RoleType::DECODE;
}

// Roles that execute the warmup forward at all. PDFUSION is included: it keeps the
// pre-warmup-feature lazy-init timing and the post-forward available_bytes sample.
inline bool isWarmupRole(RoleType role_type) {
    return isPdSeparatedRole(role_type) || role_type == RoleType::PDFUSION;
}

// Whether the measurement taken during that forward may be used for sizing. False for PDFUSION:
// it runs the forward but always sizes with the no-warmup formula against the post-teardown pool,
// which is what keeps its sizing bit-for-bit identical to the pre-feature behavior. A role that
// does not run the warmup at all never produces a WarmUpResult, so this is only meaningful for
// isWarmupRole() roles.
inline bool warmupMeasurementTrustedForRole(RoleType role_type) {
    return isPdSeparatedRole(role_type);
}

}  // namespace rtp_llm
