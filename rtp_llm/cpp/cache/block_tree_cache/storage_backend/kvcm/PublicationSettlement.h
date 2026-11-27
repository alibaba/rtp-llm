#pragma once

namespace rtp_llm::kvcm {

// Source pin lifetime alone is not a destination-reclamation proof. A settled
// payload proof is admitted only after strict all-peer validation on the PACE
// path: every item completed or was explicitly rejected/cancelled before I/O.
enum class FailedPublicationAction { ABORT_UNCOMMITTED, RETAIN_SESSION };

inline FailedPublicationAction failedPublicationAction(bool payload_may_have_started,
                                                        bool finish_attempted,
                                                        bool payload_settled_safe = false) noexcept {
    if (finish_attempted || (payload_may_have_started && !payload_settled_safe))
        return FailedPublicationAction::RETAIN_SESSION;
    return FailedPublicationAction::ABORT_UNCOMMITTED;
}

inline bool allowsPublication(bool decode_enabled, bool decode_role, bool decode_completion) noexcept {
    return decode_role ? decode_enabled && decode_completion : !decode_completion;
}

}  // namespace rtp_llm::kvcm
