#pragma once

#include <algorithm>
#include <cstdint>

#include <torch/torch.h>

namespace rtp_llm {

enum class FusedPrefillReplayValidationStatus {
    kCompatible,
    kInvalidCapturedInputLengths,
    kInvalidReplayInputLengths,
    kInvalidCapturedPrefixLengths,
    kInvalidReplayPrefixLengths,
    kInputLengthExceedsCapture,
    kPrefixLengthExceedsCapture,
    kPrefixPresenceChanged,
};

struct FusedPrefillReplayValidationResult {
    FusedPrefillReplayValidationStatus status{FusedPrefillReplayValidationStatus::kCompatible};
    int                                captured_max_input_length{0};
    int                                replay_max_input_length{0};
    int                                captured_max_prefix_length{0};
    int                                replay_max_prefix_length{0};

    bool compatible() const {
        return status == FusedPrefillReplayValidationStatus::kCompatible;
    }
};

inline bool getNonNegativeHostLengthMax(const torch::Tensor& lengths, bool allow_empty, int& max_length) {
    max_length = 0;
    if (!lengths.defined() || !lengths.has_storage() || lengths.scalar_type() != torch::kInt32 || !lengths.is_cpu()
        || !lengths.is_contiguous() || lengths.dim() != 1) {
        return false;
    }
    if (lengths.numel() == 0) {
        return allow_empty;
    }

    const auto* values = lengths.data_ptr<int32_t>();
    for (int64_t i = 0; i < lengths.numel(); ++i) {
        if (values[i] < 0) {
            return false;
        }
        max_length = std::max(max_length, static_cast<int>(values[i]));
    }
    return true;
}

inline FusedPrefillReplayValidationResult
validateFusedPrefillReplayLengths(const torch::Tensor& captured_input_lengths,
                                  const torch::Tensor& captured_prefix_lengths,
                                  const torch::Tensor& replay_input_lengths,
                                  const torch::Tensor& replay_prefix_lengths) {
    FusedPrefillReplayValidationResult result;
    if (!getNonNegativeHostLengthMax(captured_input_lengths, false, result.captured_max_input_length)) {
        result.status = FusedPrefillReplayValidationStatus::kInvalidCapturedInputLengths;
        return result;
    }
    if (!getNonNegativeHostLengthMax(replay_input_lengths, false, result.replay_max_input_length)) {
        result.status = FusedPrefillReplayValidationStatus::kInvalidReplayInputLengths;
        return result;
    }
    if (!getNonNegativeHostLengthMax(captured_prefix_lengths, true, result.captured_max_prefix_length)) {
        result.status = FusedPrefillReplayValidationStatus::kInvalidCapturedPrefixLengths;
        return result;
    }
    if (!getNonNegativeHostLengthMax(replay_prefix_lengths, true, result.replay_max_prefix_length)) {
        result.status = FusedPrefillReplayValidationStatus::kInvalidReplayPrefixLengths;
        return result;
    }

    if ((result.captured_max_prefix_length > 0) != (result.replay_max_prefix_length > 0)) {
        result.status = FusedPrefillReplayValidationStatus::kPrefixPresenceChanged;
    } else if (result.replay_max_input_length > result.captured_max_input_length) {
        result.status = FusedPrefillReplayValidationStatus::kInputLengthExceedsCapture;
    } else if (result.replay_max_prefix_length > result.captured_max_prefix_length) {
        result.status = FusedPrefillReplayValidationStatus::kPrefixLengthExceedsCapture;
    }
    return result;
}

inline const char* fusedPrefillReplayValidationStatusName(FusedPrefillReplayValidationStatus status) {
    switch (status) {
        case FusedPrefillReplayValidationStatus::kCompatible:
            return "compatible";
        case FusedPrefillReplayValidationStatus::kInvalidCapturedInputLengths:
            return "invalid captured input_lengths";
        case FusedPrefillReplayValidationStatus::kInvalidReplayInputLengths:
            return "invalid replay input_lengths";
        case FusedPrefillReplayValidationStatus::kInvalidCapturedPrefixLengths:
            return "invalid captured prefix_lengths";
        case FusedPrefillReplayValidationStatus::kInvalidReplayPrefixLengths:
            return "invalid replay prefix_lengths";
        case FusedPrefillReplayValidationStatus::kInputLengthExceedsCapture:
            return "replay max input length exceeds capture";
        case FusedPrefillReplayValidationStatus::kPrefixLengthExceedsCapture:
            return "replay max prefix length exceeds capture";
        case FusedPrefillReplayValidationStatus::kPrefixPresenceChanged:
            return "replay changes prefix presence";
    }
    return "unknown";
}

}  // namespace rtp_llm
