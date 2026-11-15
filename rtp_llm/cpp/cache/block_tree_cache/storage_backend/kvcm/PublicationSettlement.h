#pragma once

namespace rtp_llm::kvcm {

// Source safety is not a destination-reclamation proof. Keep this decision
// independent of the SDK error code and of the caller's source pin lifetime.
enum class FailedPublicationAction { ABORT_UNCOMMITTED, RETAIN_SESSION };

inline FailedPublicationAction failedPublicationAction(bool payload_may_have_started,
                                                        bool finish_attempted) noexcept {
    if (payload_may_have_started || finish_attempted)
        return FailedPublicationAction::RETAIN_SESSION;
    return FailedPublicationAction::ABORT_UNCOMMITTED;
}

inline bool allowsPublication(bool decode_enabled, bool decode_role, bool decode_completion) noexcept {
    return !decode_enabled || !decode_role || decode_completion;
}

}  // namespace rtp_llm::kvcm
