#pragma once

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>

#include "absl/status/status.h"

namespace rtp_llm {

inline absl::Status
validateTextTokensMaskLength(int64_t stream_id, const int* text_tokens_mask, int text_tokens_mask_length, int length) {
    if (text_tokens_mask != nullptr && text_tokens_mask_length != length) {
        std::ostringstream error_msg;
        error_msg << "stream [" << stream_id << "] text_tokens_mask length " << text_tokens_mask_length
                  << " does not match input length " << length;
        return absl::InvalidArgumentError(error_msg.str());
    }
    return absl::OkStatus();
}

inline void
normalizeMaskedTokenTypeIds(int* token_type_ids, const int* text_tokens_mask, int text_tokens_mask_length, int length) {
    if (token_type_ids == nullptr || text_tokens_mask == nullptr) {
        return;
    }
    const int checked_length = std::min(text_tokens_mask_length, length);
    for (int index = 0; index < checked_length; ++index) {
        if (text_tokens_mask[index] == 0) {
            token_type_ids[index] = 0;
        }
    }
}

}  // namespace rtp_llm
