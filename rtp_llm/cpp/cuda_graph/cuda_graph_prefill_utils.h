#pragma once

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace rtp_llm {

inline std::vector<int>
buildMtpDraftPrefillCaptureSequenceLengths(const std::vector<int>& decode_capture_batch_sizes,
                                           size_t                  max_bs,
                                           int                     num_tokens_per_bs) {
    if (max_bs == 0 || num_tokens_per_bs <= 0) {
        throw std::invalid_argument("MTP draft prefill capture dimensions must be positive");
    }
    if (max_bs > static_cast<size_t>(std::numeric_limits<int>::max())
        || max_bs > static_cast<size_t>(std::numeric_limits<int>::max() / num_tokens_per_bs)) {
        throw std::overflow_error("MTP draft prefill capture token capacity exceeds INT_MAX");
    }

    std::vector<int> result;
    result.reserve(decode_capture_batch_sizes.size());
    for (int batch_size : decode_capture_batch_sizes) {
        if (batch_size > 0 && static_cast<size_t>(batch_size) <= max_bs) {
            result.push_back(batch_size * num_tokens_per_bs);
        }
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

}  // namespace rtp_llm
